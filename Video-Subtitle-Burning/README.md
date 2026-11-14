# Video-Subtitle-Burning

从零实现 SRT 字幕的**状态机解析、毫秒级时间对齐与逐像素烧录**的命令行工具。
Python 侧仅依赖标准库（Pillow 为可选增强），视频编解码经系统 `ffmpeg`/`ffprobe`
完成，而**解析、时间轴调度、排版、alpha 混合全部自行实现**，不调用 ffmpeg 的
`subtitles` 滤镜。

## 设计总览

整条流水线分为四个自有模块，职责单一、各自可独立单测：

```
SRT 字节流
   │  srt.py        编码探测 + 换行归一化 + 四态状态机 + 时间码→毫秒
   ▼
Cue 列表（start_ms / end_ms / text，毫秒整数）
   │  timeline.py   帧时间轴 t=i/fps，半开区间 [start,end) 调度
   ▼
每帧的在场字幕
   │  render.py     瓦片光栅化（描边/阴影/底盒）+ 自写源覆盖 alpha 混合
   ▼
RGB24 帧缓冲 ──► ffmpeg 编码（pipeline.py 负责解码/编码双进程与音频直通）
```

### 1. 状态机解析（`subburn/srt.py`）

显式四状态行级状态机，**不假设文件守规矩**：

| 状态 | 含义 | 迁移 |
| --- | --- | --- |
| `IDLE` | 块外 | 数字行→`GOT_INDEX`；时间码行→直接 `GOT_TIME` |
| `GOT_INDEX` | 已收序号 | 时间码行→`GOT_TIME`（容忍中间空行） |
| `GOT_TIME` | 已收时间码 | 文本行→`GOT_TEXT`；新时间码→收尾后重开块 |
| `GOT_TEXT` | 收集文本中 | 空行→收尾；粘连的新块→告警后拆分 |

健壮性覆盖：

- **编码**：UTF-8/UTF-8 BOM、UTF-16 LE/BE（按 BOM 识别）、GB18030，
  全部失败时 latin-1 单字节兜底，解码永不抛异常；
- **换行**：CRLF、孤立 CR（老式 Mac）、LF 在入口统一归一化；
- **时间码**：逗号/小数点等价、缺小时 `MM:SS,mmm`、小数秒 1~3 位按
  **字符串位数**右侧补零（`.05`→50ms 而非 500ms）、分钟/秒 60 进制越界校验、
  剥离 `X1/X2/Y1/Y2` 定位信息；
- **脏数据**：缺失序号、多余空行、缺空行粘连块、错号、逆序区间（丢弃并告警）、
  零宽字幕（告警）。默认宽松恢复，`strict=True` 时任一异常直接抛 `SRTParseError`。

### 2. 时间对齐（`subburn/timeline.py`）

- 解码端用 `fps=R` 滤镜把可变帧率(VFR)归一化为恒定帧率，此后第 `i` 帧的展示
  时刻严格为 `t_i = i/R` 秒，时间戳由帧序号推导，不依赖容器 PTS；
- 字幕在**半开区间 `[start_ms, end_ms)`** 内有效：起点帧显示、终点帧不显示，
  相邻字幕在边界帧不会同帧闪现；
- 「入场游标 + 在场列表」增量推进，每条字幕只进、出各一次，
  总复杂度 O(字幕数 + 帧数)；非整数帧率（23.976 等）按浮点现算，不预取整，无累积漂移。

### 3. 像素渲染（`subburn/fonts.py` + `subburn/render.py`）

- 两级字体实现同一 `Font` 协议：内置 **5×7 位图字体**（零依赖保底，含 ASCII
  字母/数字/标点，未收录字符渲染空心方框）与可选 **Pillow TTF**（自动探测
  微软雅黑/黑体/PingFang/Noto 等，支持中文与任意字号）；
- 光栅化统一输出 **RGBA**，以 alpha 而非颜色标记墨迹，因此纯黑文字也能正确渲染；
- 瓦片支持描边（8 邻域逐层外扩）、半透明阴影、半透明底盒、多行居中排版，
  按文本缓存，同一字幕在其生命周期内只光栅化一次；
- 叠加用自写的源覆盖公式逐像素混合，含 alpha=0/255 快路径与越界裁剪：

  ```
  dst = src_rgb·src_a + dst_rgb·(1 − src_a)
  ```

### 4. ffmpeg 管道（`subburn/pipeline.py`）

解码进程输出 `rgb24` 裸流 → Python 按固定帧长读取、调度、就地混合 →
编码进程吃裸流回写 H.264（yuv420p）；音轨预先解为 m4a 再并入，无音轨自动跳过。
支持 CRF/preset、进度回调、MP4/MKV 输出。

## 环境要求

- Python 3.10+
- `ffmpeg` 与 `ffprobe` 在 PATH 中（也可用环境变量 `FFMPEG`/`FFPROBE` 指定）
- Pillow 为**可选**依赖：缺失时自动回退内置位图字体（仅支持 ASCII 字形）

## 快速开始

```bash
# 基本烧录：输出 <视频名>_subbed.mp4
python burn.py input.mp4 subs.srt

# 模块方式调用，指定输出/字体/质量
python -m subburn burn input.mp4 subs.srt -o out.mp4 --font-size 32 --crf 20

# 严格模式：SRT 格式异常直接报错；JSON 摘要；覆盖已有输出
python -m subburn burn input.mp4 subs.srt --strict --json -y
```

作为库使用：

```python
from subburn import parse_srt_file, burn_subtitles, load_font, Style

cues, warnings = parse_srt_file("subs.srt")          # 状态机解析
result = burn_subtitles(
    "input.mp4", "out.mp4", cues,
    font=load_font(size=32),                          # 不传 font 也会自动选字体
    style=Style(box=True, outline_width=2),
    progress_callback=lambda r: print(f"{r*100:.0f}%"),
)
print(result.frames_written, result.had_audio)
```

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest          # 74 个用例：解析/时间码/调度边界/混合数学/ffmpeg 端到端
python -m doctest subburn/srt.py
```

无 ffmpeg 的环境下，管道集成测试自动 skip，纯逻辑用例不受影响。

## 目录结构

```
subburn/
  srt.py        编码探测、换行归一化、时间码换算、四态状态机
  timeline.py   帧时间轴与半开区间字幕调度
  fonts.py      内置 5x7 位图字体 + 可选 Pillow TTF
  render.py     排版、瓦片缓存、RGBA 源覆盖混合
  pipeline.py   ffprobe 探测、解码/编码双进程、音频直通
  cli.py        命令行入口
tests/          单元测试 + ffmpeg 端到端集成测试
demo/           示例视频与 SRT
```

## 已知边界

- 位图字体路径（无 Pillow 时）不含 CJK 字形，中文显示为空心方框；
  安装 Pillow 并存在系统 TTF 即自动获得完整字符集；
- SRT 富文本标签（`<b>`、`{\\an8}` 等）按普通文本处理，不做样式解析；
- 输出编码固定为 H.264/yuv420p（容器可选 MP4/MKV）。
