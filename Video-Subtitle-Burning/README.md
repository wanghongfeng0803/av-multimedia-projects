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
- 瓦片支持描边（8 邻域逐层外扩）、半透明阴影、半透明底盒、多行排版，
  按文本缓存，同一字幕在其生命周期内只光栅化一次；
- 叠加用自写的源覆盖公式逐像素混合，含 alpha=0/255 快路径与越界裁剪：

  ```
  dst = src_rgb·src_a + dst_rgb·(1 − src_a)
  ```

### 3.5 字幕样式系统（`subburn/style.py`）

在渲染层之上新增一套可配置样式，数据描述与渲染逻辑分离，全部为冻结
dataclass / Enum，可哈希（直接作为瓦片缓存键的一部分）、可 JSON 往返。

**数据结构（样式参数 → 渲染映射）：**

| 子系统 | 类型 | 可配置项 | 渲染映射 |
| --- | --- | --- | --- |
| 字体 | `FontSpec` / `FontWeight` | 字体族回退链、字号（0=自动）、九档字重 | `FontRegistry` 解析族→字重→字体文件，选不到原生字重时合成粗体 |
| 颜色 | `Color` | 8-bit RGBA，透明度独立通道 | alpha 直接进入瓦片墨迹/底盒/阴影像素，复用源覆盖混合 |
| 位置 | `Position` / `Alignment` / `Margins` | 水平三档、垂直三档对齐 + 四边边距 | `anchor_origin` 计算锚点坐标，夹紧到画面内 |

颜色支持 `#rgb(a)` / `#rrggbbaa`、`rgb()/rgba()/hsl()/hsla()`、命名色、
整数、元组多种输入，并提供 **RGB↔HSL、RGB↔HSV** 的颜色空间转换（可往返）。

**边界情况处理：**

- **字体缺失回退**：族链逐个匹配 → 系统默认字体 → 内置位图字体，三级降级
  永不失败；损坏字体文件被跳过；无原生 bold 时位图/Pillow 均以合成粗体描厚；
- **颜色空间与越界**：HSL/HSV 色相自动取模、浮点通道按 0~1 解释；
  外部配置可用 `parse(clamp=True)` 把越界通道夹紧而非抛异常；
- **位置越界**：单边超画面先夹紧；对边边距之和占满画面时双双归零（避免负
  可用宽度把字幕推出画面）；字幕块比画面还大时退化为贴边，超出部分由混合
  层裁剪，保证至少左上角内容可见。

**兼容性**：旧版 `Style`（RGB 三元组 + 单一 `margin`）完整保留，
`SubtitleStyle.from_legacy()` 逐项翻译；默认样式的渲染结果与旧版**逐像素
等价**（有回归测试锁定零差异）。

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
from subburn import (
    parse_srt_file, burn_subtitles, SubtitleStyle, FontSpec,
    FontWeight, Color, Position, HAlign, VAlign, Margins,
)

cues, warnings = parse_srt_file("subs.srt")          # 状态机解析

# 新版样式：字体族回退链 + 字重、RGBA 颜色（含透明度）、对齐与四边边距
style = SubtitleStyle(
    font=FontSpec(families=("Arial", "微软雅黑"), size=32,
                  weight=FontWeight.BOLD),
    color=Color.parse("#ffdd55cc"),                   # 支持 hsl()/命名色等
    outline_color=Color(0, 0, 80),
    position=Position(HAlign.RIGHT, VAlign.TOP,
                      Margins(left=12, top=20, right=24, bottom=12)),
    box=True, box_color=Color.parse("hsla(220,80%,20%,0.45)"),
)
result = burn_subtitles("input.mp4", "out.mp4", cues, style=style,
                        progress_callback=lambda r: print(f"{r*100:.0f}%"))
print(result.frames_written, result.had_audio)
```

样式也可用 JSON 描述（字段与 `SubtitleStyle.to_dict()` 一致），命令行经
`--style style.json` 载入，并用单项参数覆盖：

```bash
python -m subburn burn input.mp4 subs.srt \
    --style style.json --alpha 200 --halign left --margin 24 -y

# 不写 JSON 也能直接调：字体族（可重复构成回退链）、字重、颜色、对齐、边距
python -m subburn burn input.mp4 subs.srt \
    --font-family Arial --font-family 微软雅黑 --font-weight 700 \
    --color "hsl(240,100%,50%)" --valign top
```

旧版 `Style` 接口继续可用，行为不变：

```python
from subburn import burn_subtitles, load_font, Style
burn_subtitles("in.mp4", "out.mp4", cues,
               font=load_font(size=32), style=Style(box=True))
```

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest          # 175 个用例：解析/样式系统/字体回退/混合数学/ffmpeg 端到端
python -m doctest subburn/srt.py
```

无 ffmpeg 的环境下，管道集成测试自动 skip，纯逻辑用例不受影响。

## 目录结构

```
subburn/
  srt.py        编码探测、换行归一化、时间码换算、四态状态机
  timeline.py   帧时间轴与半开区间字幕调度
  style.py      样式数据结构：Color/FontSpec/FontWeight/Position/Margins
  fonts.py      位图字体 + Pillow TTF + 字体族注册表与回退链
  render.py     排版、瓦片缓存、RGBA 源覆盖混合
  pipeline.py   ffprobe 探测、解码/编码双进程、音频直通
  cli.py        命令行入口（样式 JSON + 单项参数）
tests/          单元测试 + ffmpeg 端到端集成测试
demo/           示例视频与 SRT
```

## 已知边界

- 位图字体路径（无 Pillow 时）不含 CJK 字形，中文显示为空心方框；
  安装 Pillow 并存在系统 TTF 即自动获得完整字符集；
- 字重为九档语义映射：缺少原生 bold 文件时用合成粗体描厚，不支持
  可变字体的连续字重插值；
- SRT 富文本标签（`<b>`、`{\\an8}` 等）按普通文本处理，不做样式解析；
- 输出编码固定为 H.264/yuv420p（容器可选 MP4/MKV）。
