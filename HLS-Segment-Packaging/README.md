# HLS MPEG-TS 切片打包工具（HLS Segment Packaging）

把任意本地媒体文件按 **GOP（关键帧）边界**封装为 MPEG-TS（`.ts`）分片，
并生成符合 RFC 8216 的 M3U8 播放列表。**全程流拷贝、不重编码**，
分片策略与播放列表完全由本项目自己控制，不使用 `ffmpeg -f hls` 的成品输出。

## 功能特性

- **GOP 对齐切片**：切点只落在 IDR 关键帧上，每个分片首帧可独立解码，
  播放器任意 seek / 起播都不会花屏。
- **时间戳连续**：各分片保留全局连续的 PTS/DTS，跨分片首尾相接、
  无回退、无跳变；拼接后 DTS 严格单调。
- **规范的 M3U8**：自带 `EXT-X-VERSION`、`EXT-X-TARGETDURATION`、
  `EXT-X-MEDIA-SEQUENCE`、`EXT-X-PLAYLIST-TYPE`、`EXTINF`、
  `EXT-X-INDEPENDENT-SEGMENTS`、`EXT-X-ENDLIST` 等标签，
  `TARGETDURATION` 按实际最大片长向上取整，保证每片 `EXTINF` 都不超限。
- **点播 / 直播两种列表**：默认 VOD 全量列表；可指定滑动窗口，
  通过 `MEDIA-SEQUENCE` 表达窗口滚动。
- **处理高效**：一次 ffmpeg 调用 + `-c copy` 流拷贝，无解码/重编码，
  开销接近纯 IO；关键帧探测优先走只解复用的包级查询。
- **元数据落盘**：`segments.json` 记录每片的源内起点、输出时间轴起点、
  实测时长、序号，便于下游核对。
- **文字水印（可选）**：完全自研文字栅格化——手写 UTF-8 解码器、
  TrueType（.ttf/.ttc）解析器、扫描线超采样抗锯齿与 PNG 编码，零第三方
  依赖；自带 5x7 点阵字体作为字体全部缺失时的最终回退。叠加走
  `overlay` 过滤器，参数全部以独立 argv 传递、路径不进 filtergraph，
  杜绝注入；不指定水印时行为与纯流拷贝版本完全一致。
- **零第三方 Python 依赖**：仅用标准库，外部只依赖系统 `ffmpeg`/`ffprobe`。

## 目录结构

```
hls_packager/
├── __init__.py      # 公开 API
├── __main__.py      # python -m hls_packager 入口
├── cli.py           # argparse 命令行
├── errors.py        # 异常体系（环境/探测/切片/播放列表/水印）
├── binaries.py      # ffmpeg / ffprobe 可执行文件发现
├── process.py       # 子进程执行封装
├── models.py        # 不可变数据模型（StreamInfo/MediaInfo/SegmentInfo...）
├── probe.py         # ffprobe 探测 + 关键帧时间序列提取
├── planner.py       # GOP 对齐分片规划（纯函数，无 IO）
├── ts_writer.py     # segment muxer 一次成型切片 + 逐片复测（含水印分支）
├── m3u8.py          # M3U8 渲染与合规校验（纯函数）
├── packager.py      # 编排层：probe → plan → 切片 → 复测 → 播放列表
└── watermark/       # 文字水印：自研栅格化 + 安全叠加（仅依赖标准库）
    ├── utf8.py          # 手写 UTF-8 解码器（拒绝过长/代理/截断/越界）
    ├── bitmap_font.py   # 内置 5x7 点阵字体（零依赖最终回退）
    ├── ttf_font.py      # 自研 TrueType/TTC 解析（cmap/glyf/hmtx/loca...）
    ├── rasterizer.py    # 扫描线 + 4x 超采样抗锯齿字形栅格化
    ├── font_stack.py    # 字体发现与按码点回退链（显式→系统→点阵→豆腐块）
    ├── rasterize.py     # 排版绘制：码点序列 → RGBA 位图 → PNG
    ├── png_buf.py       # RGBA 位图与自研 PNG 编码
    ├── escaping.py      # 过滤器路径转义 + shell/argv 引号包裹（防注入）
    ├── overlay.py       # 九宫格定位 + ffmpeg overlay segment 命令构造
    └── config.py        # WatermarkConfig 不可变配置与校验
tests/               # pytest：纯逻辑单测 + ffmpeg 集成测试（含 72 个水印用例）
```

数据单向流动，模块之间只通过 `models.py` 里的不可变 dataclass 通信：

```
输入文件
  │  probe_media()        ffprobe：流信息 + 全部关键帧时间
  ▼
MediaInfo
  │  plan_segments()      纯函数：在关键帧中选最近切点
  ▼
[SegmentInfo 规划]
  │  write_all_segments() 一次 ffmpeg segment muxer，流拷贝
  ▼
[WrittenSegment 实测时长]
  │  _rebuild_timeline()  用实测时长重建连续时间轴
  ▼
[SegmentInfo 终态] ──► render_playlist() ──► playlist.m3u8
                   └► _write_metadata() ──► segments.json
```

## 核心设计与正确性保证

### 1. GOP 对齐算法（`planner.py`）

每个分片从当前边界 `current` 出发，理想切点为 `current + 目标时长`，
在其后的关键帧中选择**距理想切点最近**的一个；距离相同取**靠前**的关键帧。

- 所有切点必然是关键帧时刻，分片首帧即 IDR；
- **大 GOP 场景**（关键帧间隔 > 目标时长）下分片会比目标长——这是
  GOP 对齐的必然代价，`TARGETDURATION` 会按实际最大片长自动放大；
- 纯音频 / 无关键帧信息时退化为等长规划，过短的尾片自动并入前一片。

### 2. 为什么用 segment muxer 而不是逐片 `-ss/-t`

最初的逐进程方案（每片一个 `ffmpeg -ss X -t Y`）在 **B 帧重排**与
**AAC priming（编码器延迟采样）**下会过切：`-t` 按输出包起点计时，
音频 priming 会让容器多吞下一个 GOP 的关键帧，导致分片变长、混入额外 IDR。

ffmpeg 的 `segment` muxer 在解复用器内部按关键帧边界切分，正确处理
重排与 priming。本工具仍然**完全掌握切片策略**——切点通过
`-segment_times` 显式传入（来自自己的规划器），播放列表也由自己生成，
segment muxer 只负责"在给定时刻做流拷贝切割"，并未使用 ffmpeg 的
hls muxer 或其自带播放列表。

### 3. 时间戳连续性（`ts_writer.py` + `packager.py`）

- 切片**不启用** `-reset_timestamps`，各分片保留全局时间戳，相邻分片
  PTS/DTS 首尾相接；每个 TS 分片又都由 muxer 写入独立的 PAT/PMT，
  因此既能独立解析，又在时间轴上连续。
- 编排层用 ffprobe **逐片复测实际时长**，并以"前序实测时长之和"重建
  输出时间轴，使 `EXTINF` 之和与分片真实时长严格一致，避免帧级漂移累积。

### 4. M3U8 合规（`m3u8.py`）

渲染前做硬性自检，违规直接抛错而不是产出非法播放列表：

- `TARGETDURATION = ceil(max(分片时长))`，且断言每片不超过它；
- 分片序号必须连续；直播窗口的 `MEDIA-SEQUENCE` 与窗口首片严格对齐；
- `EXTINF` 统一 3 位小数浮点秒；VOD 列表带 `ENDLIST`，直播列表不带。

## 安装与环境

需要系统提供 `ffmpeg` 与 `ffprobe`（建议 4.4+）。可执行文件查找顺序：

1. 环境变量 `FFMPEG_BIN` / `FFPROBE_BIN`；
2. `PATH`；
3. Python 解释器同目录（便携部署）。

无需安装 Python 第三方库；运行测试才需要 pytest：

```bash
pip install -r requirements-dev.txt
```

## 命令行用法

```bash
# 基本：VOD 点播，目标分片 6 秒
python -m hls_packager input.mp4 output_dir/

# 指定目标分片时长（实际切点对齐最近关键帧）
python -m hls_packager input.mp4 output_dir/ --segment-duration 4

# 直播滑动窗口（保留最近 3 片）+ 自定义起始序号与文件名前缀
python -m hls_packager input.mp4 output_dir/ -d 6 -w 3 -s 100 -p chunk

# 允许写入非空输出目录（覆盖同名产物）
python -m hls_packager input.mp4 output_dir/ -y

# 叠加文字水印（视频轨重编码，音频仍流拷贝；不指定水印则保持流拷贝）
python -m hls_packager input.mp4 output_dir/ -d 6 -y \
  --watermark-text "直播水印 HLS 2026!" \
  --watermark-size 30 --watermark-color white \
  --watermark-anchor bottom-right

# 指定显式字体（可重复传入形成回退链）与九宫格位置
python -m hls_packager input.mp4 output_dir/ -y \
  --watermark-text "机密" --watermark-font C:/Windows/Fonts/msyh.ttc \
  --watermark-anchor top-left --watermark-margin-x 16 --watermark-margin-y 16
```

参数一览：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `input` | 输入媒体文件（mp4/mkv/mov/ts/m4a 等） | 必填 |
| `output_dir` | HLS 产物输出目录（自动创建） | 必填 |
| `-d/--segment-duration` | 目标分片时长（秒），切点对齐 GOP | 6 |
| `-w/--window-size` | 直播滑动窗口分片数；不指定即 VOD | 无 |
| `-s/--media-sequence` | 首片 `EXT-X-MEDIA-SEQUENCE` 序号 | 0 |
| `-p/--filename-prefix` | 分片文件名前缀 | seg |
| `-y/--overwrite` | 允许写入非空目录并覆盖同名产物 | 关 |
| `--watermark-text` | 水印文字（UTF-8，支持多字节/特殊字符）；不指定即不叠加 | 无 |
| `--watermark-font` | 显式字体文件，可重复传入按顺序回退 | 系统字体 |
| `--watermark-size` | 水印字号（像素） | 28 |
| `--watermark-color` | 文字颜色（`#RGB`/`#RRGGBB`/命名色） | white |
| `--watermark-opacity` | 不透明度 0.0~1.0 | 1.0 |
| `--watermark-anchor` | 九宫格锚点（如 `top-left`/`center`/`bottom-right`） | bottom-right |
| `--watermark-margin-x/y` | 水印距画面边缘的像素留白 | 24 / 24 |

退出码：`0` 成功；`2` 可预期的输入/环境/切片错误；`3` 未知内部错误。

### 文字水印的安全与回退设计

- **防注入两道防线**：① 子进程始终以 argv **列表**启动、绝不经过 shell，
  含空格/引号/分号/`$()`/反引号/中文的路径与文本作为独立元素原样传递；
  ② 水印 PNG 经第二个 `-i` + 流标签进入过滤器图，**路径永不拼进
  filtergraph 字符串**，从根上消除过滤器注入面。`escaping.py` 另提供
  filtergraph 两级转义与 POSIX/Windows 引号包裹（仅用于日志展示）。
- **字体回退链（逐码点决策）**：显式字体 → 系统 TrueType（CJK 字体优先，
  支持 .ttf/.ttc）→ 内置 5x7 点阵（仅 ASCII）→ 豆腐块占位。解析失败的
  字体被缓存跳过，绝不静默丢字。
- **GOP 承诺不破坏**：重编码时把规划切点同时喂给 `-force_key_frames`，
  并对图片输入加 `-loop 1`（否则单帧图片会经 `shortest` 截断整段视频），
  保证每片首帧仍是 IDR、时间戳全局连续、`segment_times` 精确生效。

## 作为库调用

```python
from hls_packager import package, plan_segments, render_playlist, PlaylistConfig
from hls_packager import WatermarkConfig, render_text_png

# 一键打包
result = package("input.mp4", "out/", target_duration=4.0)
for seg in result.segments:
    print(seg.sequence_number, seg.filename, seg.start_time, seg.duration)

# 带文字水印的打包（watermark=None 时与纯流拷贝行为完全一致）
wm = WatermarkConfig(
    text="内部资料 2026",
    font_size=30,
    color="#FF8800",
    anchor="bottom-right",
)
result = package("input.mp4", "out/", target_duration=6.0, watermark=wm, overwrite=True)

# 也可单独使用纯函数模块
planned = plan_segments(duration=30.0, target_duration=4.0,
                        keyframe_times=(0.0, 2.0, 4.0, ...))
text = render_playlist(planned, PlaylistConfig(target_duration=4))

# 只做文字栅格化（自行产出 RGBA PNG，不跑 ffmpeg）
rendered = render_text_png(WatermarkConfig(text="水印 ABC"), "watermark.png")
print(rendered.width, rendered.height, rendered.used_fallbacks)
```

## 产物说明

```
output_dir/
├── playlist.m3u8     # 播放列表
├── seg_0000.ts       # MPEG-TS 分片（188 字节定长包，同步字节 0x47，自带 PAT/PMT）
├── seg_0001.ts
├── ...
└── segments.json     # 机器可读元数据（源内起点 / 时间轴起点 / 实测时长 / 序号）
```

## 支持的编码

流拷贝进 MPEG-TS，无需重编码的常见编码：

- 视频：H.264/AVC、HEVC、MPEG-1/2；
- 音频：AAC、MP3、AC-3、E-AC-3、MP2、Opus。

主视频编码不支持时直接报错（避免静默产出只有声音的"视频"）；
个别不支持的音频轨道会跳过。字幕 / 数据流不打入 TS 分片。

## 运行测试

```bash
pytest
```

- `tests/test_planner.py`、`test_m3u8.py`、`test_probe.py`、
  `test_ts_writer.py`：纯逻辑单测，不依赖 ffmpeg；
- `tests/test_watermark_*.py`：水印单测（UTF-8 异常输入、字体回退、
  转义注入、PNG 渲染）+ ffmpeg 集成测试（首帧 IDR、水印可见、
  特殊字符路径、时间戳连续、CLI 端到端）；
- `tests/test_integration.py`、`test_cli.py`：用 ffmpeg 合成
  H.264/HEVC、纯音频、大 GOP 等素材做端到端校验，覆盖首帧 IDR、
  独立解码、GOP 对齐、全局时间戳连续、concat 单调、M3U8 合规、
  滑动窗口等口径。无 ffmpeg 时自动 skip。

## 已知限制

- 不带水印时只做**重封装**，不转码；源编码无法承载于 MPEG-TS 时会明确报错，
  不会偷偷转码。
- **带水印必然重编码视频轨**（像素级叠加无法流拷贝），仅支持 H.264/HEVC/
  MPEG-1/2 视频；音频仍流拷贝。纯音频输入不能叠加文字水印，会明确报错。
- 自研 TrueType 解析器只支持 TrueType 轮廓与 cmap format 0/4/12；
  OpenType/CFF（OTTO）、WOFF 会被跳过并走字体回退链，不影响主流程。
- 单码率输出，不生成多码率 master playlist。
- 直播模式产出的是"当前窗口快照"式 VOD 风格文件，不包含分片滚动
  清理与周期刷新守护进程；`MEDIA-SEQUENCE` 语义已就绪，可作为
  实时打包循环的构建块。
- 切点数量极多导致命令行超长时会明确报错并提示增大分片时长
  （Windows 命令行长度上限保护）。
