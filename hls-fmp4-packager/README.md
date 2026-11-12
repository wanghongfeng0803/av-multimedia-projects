# fMP4 HLS 打包器

单码率、基于 **fragmented MP4（fMP4）** 的 HLS 打包器。ffmpeg 只负责转码并把
fragmented mp4 字节流写到管道；**切点规划、moof/mdat 重组、init 段、m3u8 与分片
元数据全部由本项目代码生成，不使用 `ffmpeg -f hls`**，切点完全可控。

## 需求对照

| 需求 | 实现 |
| --- | --- |
| 分片 2 秒 | ffmpeg 强制 2s GOP，自有状态机在 IDR 上收口，实际每片精确 2.000s（末片除外） |
| GOP 与分片边界对齐，切点只落 IDR | 转码端 `-g`/`-force_key_frames` 固定 IDR，打包端只在 `sample.sync` 样本上切 |
| 首片独立起播 | 第一片从第一个 IDR 开始，写入 `#EXT-X-INDEPENDENT-SEGMENTS`，init 段外置 |
| 断流后从续点继续，不从头 | `session.json` 记录输入侧消费位置，重连用 `-ss` 续点；时间线按秒锚点平移 |
| `EXT-X-DISCONTINUITY` 位置正确 | 断流续接、编码变化的「新编码会话第一片」前插入，旧残片不挂标记 |
| 自己写 m3u8 并校验 | `packager/manifest.py` 手写 HLS v7 播放列表并内置规范校验 |
| 输出每片 start_time/时长/序列号 | `segments.json` + 命令行表格 |
| 单码率 | 单组转码参数，无 master variant 切换 |
| init 段单独生成，各片共用 | `init.mp4` 外置；编码变化时版本化为 `initN.mp4` 并在列表里 `EXT-X-MAP` 切换 |
| ffplay 冷启动 ≤ 3s | 实测约 0.4–1.1s |

## 架构

```
输入文件
  │  ffmpeg（libx264 固定 2s GOP + AAC，-f mp4 输出 fragmented mp4 到管道）
  ▼
FragmentedMp4Reader（packager/mp4reader.py）
  │  增量解析 ftyp/moov（stsd/timescale/trex）与 moof/mdat（tfdt/trun 样本）
  ▼
HlsPackager 状态机（packager/pipeline.py）
  │  ① 时间戳按「秒锚点」平移到全局连续时间线
  │  ② 只在视频 IDR 且累计时长达标时切（CUT_RATIO=0.96，容忍一帧误差）
  │  ③ 断流检测 → 残片收口 → -ss 续点重拉 → DISCONTINUITY
  │  ④ stsd 变化 → 版本化新 init + EXT-X-MAP
  ▼
Fmp4Writer（packager/fmp4writer.py）   Manifest（packager/manifest.py）
  init: ftyp+moov(mvex/trex)              master.m3u8（手写 + 自校验）
  seg : styp+sidx+moof+mdat               segments.json
```

| 模块 | 职责 |
| --- | --- |
| `packager/boxes.py` | ISO-BMFF 盒子读写原语（大端、64 位 size、FullBox、容器遍历） |
| `packager/mp4reader.py` | fMP4 字节流增量解析，输出 moov 事件与样本事件 |
| `packager/fmp4writer.py` | 自构 init 段与媒体段（tfhd/tfdt/trun/data_offset/sidx） |
| `packager/pipeline.py` | ffmpeg 管道、切点状态机、断流重连、时间戳平移、编码变化检测 |
| `packager/manifest.py` | m3u8 渲染与规范校验、分片元数据时间线校验 |
| `packager/session.py` | `session.json` 持久化，支撑跨进程 `--resume` |
| `hls_packager.py` | 命令行入口 |
| `validate.py` | 独立校验：盒子层 + ffprobe 双路验证 |
| `tests/e2e_test.py` | 端到端验收（5 场景 + 冷启动计时） |

## 分片策略与 IDR 对齐原理

1. **转码端固定 GOP**：`-g 2*fps -keyint_min 2*fps -sc_threshold 0
   -force_key_frames expr:gte(t,n_forced*2)`，保证每 2 秒恰好一个 IDR，
   且不被场景检测插入额外关键帧。源本身 GOP 不规则也会被重新规整。
2. **打包端只认 IDR**：样本进入队列后，状态机只在 `sync=True` 的视频样本上
   评估切割；当该 IDR 与当前分片首帧的时间跨度 ≥ `2s × 0.96` 时收口，IDR
   本身留给下一片作为首帧。因此**每个分片首帧必为 IDR，切点与 GOP 边界严格对齐**。
3. **音频跟随视频切点**：音频样本按 PTS 不超过视频切点边界归入同一分片，
   视频/音频各一个 `traf`，共享同一个 `mdat`，各自的 `trun.data_offset`
   指向本 track 在 mdat 内的起点。
4. 首片从流中第一个 IDR 起，配合外置 init 段与 `#EXT-X-INDEPENDENT-SEGMENTS`，
   播放器无需任何前置分片即可解码起播。

## fMP4 产物结构

- `init.mp4`：`ftyp` + `moov`（含 `mvhd/trak/mdia/minf/stbl/stsd` 编码配置
  与 `mvex/trex`），不含任何样本，可被所有分片复用。
- `segN.m4s`：`styp` + `sidx` + `moof` + `mdat`。
  - `moof/mfhd` 写 HLS 分片序号；每个 `traf` 含 `tfhd`（default-base-is-moof）、
    `tfdt`（baseMediaDecodeTime）、`trun`（每样本 duration/size/flags/cto 与
    data_offset）。
  - 首样本 `sample_flags` 置 `0x02000000`（sample_depends_on=2，同步样本），
    其余置非同步标志。
  - 所有盒子 size 字段为含 8 字节头的总长度，`data_offset` 经两遍构建精确回填。

## 断流续传机制

- **输入侧记账**：每次 ffmpeg 拉流记录 `-ss` 起点与本进程最大视频输入 PTS，
  得到「输入文件侧」已消费秒数，写入 `session.json`。
- **续点重拉**：进程退出（自然 EOF 之外的任何中断）后，用
  `ffmpeg -ss <consumed_seconds> -i <input>` 从断点重新拉流，配合退避重试
  （`--max-reconnect`、`--reconnect-backoff`）。
- **时间戳不回退**：全局时间线以**秒**记账（与 timescale 解耦）。每次拉流以
  该 track 首样本建立锚点，`全局秒 = 锚点秒 + (本样本dts-锚点dts)/timescale`，
  再按当前 timescale 换算回整数 dts；重拉重叠部分钳制到时间线末尾。这样即使
  重连后帧率/timescale 变化（25000→30000），时间戳依然单调连续。
- **跨进程 `--resume`**：进程被杀死后重新带 `--resume` 启动，从 `session.json`
  恢复已落盘分片、序列号、时间线与续点，已完成分片不会重打。注意 JSON 对象键
  会把整数 track_id 转成字符串，加载时统一归一回 int。

### 为什么 DISCONTINUITY 挂在新会话第一片

断流前已解析的残片仍属于**旧编码会话**，不能给它挂 DISCONTINUITY；标记统一在
残片收口之后置位，由新编码会话的第一个分片携带。`#EXT-X-DISCONTINUITY` 告诉
播放器后续分片的编码时间戳/编码参数可能不连续，需要重置解码时间线。

## 编码参数变化与多 init

fMP4 的解码配置（SPS/PPS、esds 等）在 init 段的 `stsd` 里，媒体分片只放裸
样本。一旦编码参数变化：

- 旧分片仍引用旧 init，**不能覆盖** `init.mp4`，否则旧片会用新 SPS/PPS 解码而花屏；
- 打包器把新配置写成版本化的 `init1.mp4`、`init2.mp4`…，并在变化分片前同时输出
  `#EXT-X-DISCONTINUITY` 与新的 `#EXT-X-MAP:URI="init1.mp4"`，每个分片在
  `segments.json` 里记录自己引用哪份 init。

> 说明：本工具始终用 libx264 重编码，真实部署中输出 profile/分辨率固定时，
> 输入变化通常不会改变 SPS/PPS，只需要 DISCONTINUITY；多 MAP 机制用于输出参数
> 确实变化的场景。另：ffmpeg 命令行的 HLS demuxer 对单个 playlist 内 fMP4 的
> `EXT-X-MAP` 重载支持不完整（其已知限制），但产物按 RFC 8216 合规，Safari/
> hls.js/ExoPlayer 等原生客户端支持；校验时按 init 分组合并解码逐组验证。

## m3u8 与自校验

`packager/manifest.py` 手写并校验（HLS 版本 7）：

- 必备标签 `#EXTM3U/VERSION/TARGETDURATION/MAP`，VERSION≥7；
- `EXTINF` 精度 ≤3 位小数、四舍五入不超过 TARGETDURATION，与分片实际样本时长一致；
- `EXTINF` 与分片 URI 一一配对，序号连续、时间戳单调不回退、每片首帧 IDR；
- DISCONTINUITY 与 EXT-X-MAP 的位置/顺序正确。

`validate.py` 在此之外再做盒子层结构检查（盒子顺序、mfhd 序号、trun
data_offset 落在 mdat 内、tfdt 单调、首样本同步标志）与 ffprobe 交叉验证
（按 init 分组统计 IDR 数 == 分片数、ffprobe 能解析 m3u8）。

## 用法

```bash
# 基本打包
python hls_packager.py input.mp4 output_dir/

# 显式指定 2 秒分片
python hls_packager.py input.mp4 output_dir/ --segment-duration 2

# 独立校验
python validate.py output_dir/

# 模拟断流（消费到 4.5s 杀掉 ffmpeg，自动续点重拉）
python hls_packager.py input.mp4 out/ --simulate-disconnect 4.5

# 断流后换输入（制造编码参数变化，触发多 init + EXT-X-MAP）
python hls_packager.py a.mp4 out/ --simulate-disconnect 5 --reconnect-input b.mp4

# 跨进程断点续传（先制造崩溃，再 resume）
python hls_packager.py a.mp4 out/ --simulate-disconnect 3 --max-reconnect 0
python hls_packager.py a.mp4 out/ --resume
```

输出：

```
output_dir/
├── master.m3u8      # 手写 HLS v7 播放列表
├── init.mp4         # 初始化段（编码变化时另有 init1.mp4 ...）
├── seg0.m4s ...     # 媒体分片
├── segments.json    # 每片 sequence_number/start_time/duration/init_file/...
└── session.json     # 续传状态
```

`segments.json` 每条记录：

```json
{
  "sequence_number": 0,
  "filename": "seg0.m4s",
  "start_time": 0.0,
  "duration": 2.0,
  "discontinuity": false,
  "first_frame_idr": true,
  "size_bytes": 562455,
  "init_file": "init.mp4"
}
```

## 验收结果（Windows + ffmpeg 9.0.2）

`python tests/e2e_test.py` 覆盖五场景，全部通过：

| 场景 | 分片 | DISCONTINUITY | 结果 |
| --- | --- | --- | --- |
| 正常打包 | 6 | 无 | 总时长 12.000s，整 playlist 解码零错误 |
| 进程内断流续传 | 6 | 分片 3 前 | 时间线连续，解码零错误 |
| 跨进程 `--resume` | 5 | 分片 1 前 | 不从头，序列号/时间线续接 |
| 编码参数变化（720p→480p） | 7 | 分片 3 前 | init.mp4→init1.mp4，两组各自解码零错误 |
| 无音频源 | 3 | 无 | 单视频 track 正常 |

- ffprobe：关键帧间隔全部 2.0s±0.12s，每片首帧 IDR；
- 整段 `ffmpeg -f null` 解码无 corrupt/error（无花屏、声画同步）；
- ffplay 冷启动到出画面实测 **约 0.4–1.1s**，满足 ≤3s 验收线。

## 限制与说明

- 单码率输出；多码率 master playlist 不在本题范围。
- 依赖 ffmpeg/ffprobe/ffplay 在 PATH 中，仅用标准库实现 MP4/m3u8 逻辑，无第三方 Python 依赖。
- 测试源由 `tests/e2e_test.py` 用 lavfi 现场生成，产物落在 `tests/work/`（已 gitignore）。
