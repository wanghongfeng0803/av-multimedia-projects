# ABR-Playlist-Generation（vhls）

零三方依赖的 HLS 打包器：ffmpeg 只负责重编码切片，时间轴规划、
m3u8 渲染与全部时序校验都用 Python 标准库实现。支持**单码率**与
**多码率（ABR）**两种模式，核心目标是**切片与播放列表共用同一套
时间轴，EXTINF 与分片真实时长严格一致，主列表标注可独立核对**。

## 设计约束（本项目要解决的问题）

单码率：

1. **分片时长必须与 EXTINF 声明一致**（含浮点舍入），实测偏差超容差
   整体失败，不产出对不上的列表；
2. **媒体序列号连续**，从 0 起逐片递增，写盘前以计划为基准交叉校验；
3. **EXT-X-TARGETDURATION 与各分片时长上限对得上**——取各片实测时长
   上限的整数秒向上取整，不是估算值；
4. **末片不足目标时长时单独标出且时长准确**，EXTINF 如实写真实长度，
   绝不补齐成目标时长；
5. **列表写入与切片切割共用同一套时间轴**，从结构上消除累积误差漂移。

多码率（ABR）：

6. **主列表 BANDWIDTH/RESOLUTION 与各子流真实值一致**——BANDWIDTH
   取该路所有分片「文件字节数 × 8 ÷ 实测时长」的峰值（RFC 8216 峰值
   分片码率口径），RESOLUTION 取分片视频流的 ffprobe 实测宽高，两者
   都可用磁盘上的分片独立重算核对，不是请求值；
7. **各子列表分片对齐一致**——所有码率共享同一份时间轴计划，切点/
   片数/序列号由构造保证一致，不同码率之间不能多片/少片；切割完成后
   逐片复核各码率实测时长，偏差超容差即整体失败；
8. **旧单码率输出保持兼容**——`pack_hls` 与 `hls_pack.py` 行为不变，
   ABR 是独立的 `pack_abr` / `abr_pack.py` 入口。

## 一套时间轴是怎么做到零漂移的

常见实现的漂移来源：每片各自 `round(目标时长)` 一次浮点，N 片之后舍入
误差沿时间轴累积，末片起点悄悄偏移；切点又由 ffmpeg 按关键帧自行吸附，
列表声明与文件实际边界根本不是同一条时间线。

本包把时间轴收敛成一个不可变的 `TimelinePlan`（`vhls/timeline.py`），
切割器与播放列表都只读它：

| 阶段 | 模块 | 职责 |
| --- | --- | --- |
| 探测 | `vhls/probe.py` | ffprobe 逐帧 PTS/时长（VFR 同样成立）、音频存在性 |
| 规划 | `vhls/timeline.py` | 入口一次性浮点秒→**整数毫秒**，帧边界贪心分组，产出唯一时间轴 |
| 切割 | `vhls/cut.py` | 按计划 `-ss/-t` 重编码帧精确切片，片首强制 IDR；逐片独立 ffprobe 实测 |
| 列表 | `vhls/playlist.py` | EXTINF 写实测毫秒的 3 位小数字符串，校验通过后原子写入 |

三个关键口径：

- **整数毫秒时间单位**：浮点只在两个边界出现——入口的 PTS 换算与出口
  的 EXTINF 格式化，中间所有边界、比较、求和都是整数运算，不存在跨片
  浮点累加。毫秒量化误差有界（±0.5ms/帧，25/24fps 等整数帧长为 0），
  详见 `timeline.py` 模块 docstring。
- **切点只落真实帧边缘**：贪心累积帧时长，装不下下一帧才关片，每片是
  整数个完整帧，不会把一帧劈成两半；非末片相对目标最多偏差一帧。
- **实测取视频轨时长，不取 format duration**：MPEG-TS 复用器会在容器
  尾部做 PCR/音频帧对齐填充，实测 format duration 比视频轨长出约一个
  AAC 帧（~23ms），那段时间没有画面。HLS 按视频轨时间线呈现，EXTINF
  以视频流时长为准才能与帧边界计划对应（实测音频素材 format=2.023s
  而视频轨=2.000s）。

## 校验与失败口径

切割完成后、写盘之前，`validate_against_plan` 以**计划**为基准逐项交叉
校验（不能用结果自带的规格自证）：结果数量、序列号连续性、文件名、
分片存在且非空、每片实测 ≤ TARGETDURATION、时间区间为正。任一条不过
即抛错且不动磁盘；播放列表本身用「同目录临时文件 + `os.replace`」
原子落盘，进程失败不会留下半截 m3u8。

## 实测（25fps / 5.08s 含音轨，目标 2s）

```
#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:2
#EXT-X-MEDIA-SEQUENCE:0
#EXT-X-PLAYLIST-TYPE:VOD
#EXT-X-INDEPENDENT-SEGMENTS
#EXTINF:2.000,
seg_00000.ts
#EXTINF:2.000,
seg_00001.ts
#EXTINF:1.080,
seg_00002.ts
#EXT-X-ENDLIST
```

- 3 个分片，序列号 0/1/2 连续；前两片各 50 帧精确 2.000s；
- 末片 27 帧 1.080s 被标记为短尾片，EXTINF 如实写 `1.080` 而非补齐；
- 各片 EXTINF 之和 5.080s 与源时长**逐毫秒相等**；
- 每个分片可被 ffmpeg 独立解码（片首 IDR），按 m3u8 反向 remux 回
  单个文件时长与源一致。

## 多码率（ABR）模式

`pack_abr` / `abr_pack.py` 在单码率四步之上加一层编排：

```
输出目录/
  master.m3u8          # 主列表（EXT-X-STREAM-INF 指向各子列表）
  180p/index.m3u8      # 每路码率一个子目录 + 子列表
  180p/seg_00000.ts
  360p/index.m3u8
  360p/seg_00000.ts
```

- **共享时间轴**：所有码率共用同一份 `TimelinePlan`，切点/片数/序列号
  完全一致——「不同码率不能多片/少片」由构造保证，不靠事后对齐。
- **标注可独立核对**：主列表的 `BANDWIDTH` 是该路分片的峰值码率
  （`max(文件字节数 × 8 ÷ 实测时长)`），`RESOLUTION` 是分片视频流的
  ffprobe 实测宽高。任何核对者用磁盘上的分片即可重算，不依赖本项目
  的内部状态。源小于目标分辨率时不放大，RESOLUTION 如实写源分辨率。
- **跨码率对齐复核**：切割完成后逐片比对各码率实测时长，偏差超容差
  （默认 2×20ms）即整体失败，不写主列表。

```bash
python abr_pack.py 输入视频 输出目录 [--target 6] [--tolerance 20] \
                    [--ladder 360p:400k,480p:800k,720p:2500k]
```

- `--ladder`：码率阶梯，格式 `<名称>:<码率>`，逗号分隔；缺省按源高度
  自动生成（360p/480p/720p/1080p 中不超过源高度的档位，源更矮时回退
  到源分辨率单档）。

作为库调用：

```python
from vhls import pack_abr, Rendition

result = pack_abr("in.mp4", "out_abr", renditions=(
    Rendition(name="360p", height=360, video_bitrate=400_000),
    Rendition(name="720p", height=720, video_bitrate=2_500_000),
))
for v in result.variants:
    print(v.rendition.name, v.bandwidth, f"{v.width}x{v.height}")
```

## 用法（单码率）

```bash
python hls_pack.py 输入视频 输出目录 [--target 6] [--playlist index.m3u8] \
                       [--prefix seg] [--tolerance 20]
```

- `--target`：目标分片时长（秒），默认 6（Apple HLS 推荐值，与
  24/25/30fps 都是整数帧边界）；
- `--tolerance`：分片实测时长与计划值容差（毫秒），默认 20；
- 产物：`<prefix>_00000.ts`、`<prefix>_00001.ts` … + 播放列表。

作为库调用：

```python
from vhls import pack_hls

result = pack_hls("in.mp4", "out_dir", target_seconds=2.0)
print(result.playlist.target_duration, result.short_segments)
```

## 测试

```bash
python -m pytest tests/ -q
```

- `test_timeline.py` / `test_playlist.py`：纯逻辑，不依赖 ffmpeg，
  覆盖帧边界分组、VFR、整数毫秒无漂移、量化误差边界、EXTINF 舍入、
  TARGETDURATION 对齐、序列号连续性、短尾标记与原子写入；
- `test_abr.py`：ABR 纯逻辑，覆盖默认阶梯、主列表渲染、峰值码率
  计算、跨码率对齐复核的接受/拒绝路径；
- `test_integration_pack.py`：单码率真实 ffmpeg 端到端，独立解析
  m3u8 并对每片单独 ffprobe 交叉验证（环境缺 ffmpeg/ffprobe 时自动
  skip）；
- `test_integration_abr.py`：ABR 真实 ffmpeg 端到端，独立解析
  master.m3u8 的 BANDWIDTH/RESOLUTION 并对每片单独 ffprobe 重算核对，
  验证各路码率分片数与 EXTINF 逐片对齐。
