# Video-Subtitle-Burning

从零实现 **SRT 与 ASS/SSA 字幕**的状态机解析、毫秒级时间对齐与逐像素烧录的
命令行工具。Python 侧仅依赖标准库（Pillow 为可选增强），视频编解码经系统
`ffmpeg`/`ffprobe` 完成，而**解析、时间轴调度、排版、alpha 混合全部自行实现**，
不调用 ffmpeg 的 `subtitles`/`ass` 滤镜。

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

### 3.6 ASS/SSA 样式与特效（`subburn/ass.py` + `subburn/ass_render.py`）

ASS 与 SRT 的根本区别是「样式集中声明 + 行内覆盖标签」：`[V4+ Styles]`
段定义样式表，`[Events]` 的每条对话引用样式名，并用 `{\...}` 标签在文本
中途切换样式或声明特效。两个模块延续「解析与像素分离」的分层：

**解析层（`ass.py`，零像素）**

- 段驱动解析 `[Script Info]` / `[V4+ Styles]` / `[V4 Styles]` / `[Events]`，
  编码探测与换行归一化直接复用 SRT 通道（UTF-8/UTF-16 BOM/GB18030/latin-1）；
- ASS 私有颜色 `&HAABBGGRR` 在入口完成 BGR→RGB 通道交换与 alpha 反转
  （ASS alpha 是*透明度*，0 不透明），短色值按整数右对齐；时间码
  `H:MM:SS.cc`（百分秒，前导零有位值意义，`.05`≠`.5`）；
- 覆盖标签用「反斜杠分词 + 状态快照」状态机处理，文本被切成共享样式的
  不可变 `ASSRun`；支持 `\b \i \u \s`（粗/斜/下划线/删除线）、
  `\fn \fs`（字体/字号）、`\c \alpha`（颜色/透明度）、`\fscx \fscy \fsp`
  （缩放/字间距）、`\an \a`（numpad 对齐）、`\pos \move`（定位/移动）、
  `\fad \fade`（淡入淡出）、`\k \kf \K \ko`（卡拉OK）、`\r[样式名]`（重置）、
  `\N \h`（硬换行/硬空格）；未识别标签安静跳过，`\p` 矢量绘图内容丢弃；
- SSA v4 旧对齐编号自动换算为 ASS numpad，`Comment` 事件忽略，坏时间码/
  逆序区间/未定义样式按 SRT 同一套 `strict`/告警机制降级恢复。

**渲染层（`ass_render.py`）**

- PlayRes→真实帧的统一缩放系数在构造时确定，样式字号映射进既有
  `FontSpec` 回退链（含新增的**斜体**维度：原生斜体文件优先，缺失时
  位图字体做 2:7 切变、Pillow 字体做 AFFINE oblique 合成）；
- 一条对话光栅化为一张 RGBA 瓦片，仍走 `render.blend_rgba_tile`
  源覆盖混合（裁剪、alpha 快路径、整数公式全部继承，不重复实现混合）；
- 瓦片缓存键是**视觉状态**（淡入 alpha + 卡拉OK 音节到点态）而非帧号：
  无特效对话整条生命周期只有一张瓦片，`\move` 只改锚点、瓦片全程复用，
  淡入结束后回到共享状态——逐帧特效也只在状态真正变化时重建；
- 几何特效：numpad 九档对齐 + 三边距、`\pos` 锚点定位、`\move` 线性插值、
  BorderStyle=3 不透明底盒、`\fad` 全局 alpha、卡拉OK 主/次色到点切换。

### 4. ffmpeg 管道（`subburn/pipeline.py`）

解码进程输出 `rgb24` 裸流 → Python 按固定帧长读取、调度、就地混合 →
编码进程吃裸流回写 H.264（yuv420p）；音轨预先解为 m4a 再并入，无音轨自动跳过。
支持 CRF/preset、进度回调、MP4/MKV 输出。SRT 与 ASS 共用同一个 `_pump_frames`
逐帧泵，两种格式的差异只封装在各自的帧回调（`OverlayRenderer` vs
`ASSRenderer`）里，进程管理与读写循环只有一份实现。

## 环境要求

- Python 3.10+
- `ffmpeg` 与 `ffprobe` 在 PATH 中（也可用环境变量 `FFMPEG`/`FFPROBE` 指定）
- Pillow 为**可选**依赖：缺失时自动回退内置位图字体（仅支持 ASCII 字形）

## 快速开始

```bash
# 基本烧录：按扩展名/内容自动识别 SRT 或 ASS，输出 <视频名>_subbed.mp4
python burn.py input.mp4 subs.srt
python burn.py input.mp4 subs.ass

# 模块方式调用，指定输出/字体/质量（SRT 样式参数）
python -m subburn burn input.mp4 subs.srt -o out.mp4 --font-size 32 --crf 20

# 严格模式：字幕格式异常直接报错；JSON 摘要；覆盖已有输出
python -m subburn burn input.mp4 subs.ass --strict --json -y
```

ASS 的样式与特效完全由字幕文件内的 `[V4+ Styles]` 与行内标签决定，
SRT 的样式覆盖参数（`--color`/`--font-family`/`--box` 等）在 ASS 模式下
会被明确拒绝（返回码 2），避免「参数看似生效却被忽略」的歧义；`--crf`、
`--preset`、`--strict`、`--json`、`-y` 等通用参数两种格式都适用。

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

ASS 路径用同一套调度/管道，只换解析器与渲染器：

```python
from subburn import parse_ass_file, burn_ass

document, warnings = parse_ass_file("subs.ass")   # 样式表 + 带标签的对话
result = burn_ass("input.mp4", "out.mp4", document,
                  progress_callback=lambda r: print(f"{r*100:.0f}%"))
print(result.cues, len(document.styles))
```

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest          # 250+ 用例：SRT/ASS 解析、样式系统、字体回退、
                          # 混合数学、卡拉OK/淡入/定位特效、ffmpeg 端到端
python -m doctest subburn/srt.py
```

无 ffmpeg 的环境下，管道集成测试自动 skip，纯逻辑用例不受影响。

## 目录结构

```
subburn/
  srt.py        SRT：编码探测、换行归一化、时间码、四态状态机
  ass.py        ASS/SSA：段解析、&HAABBGGRR 颜色、覆盖标签状态机、样式表
  timeline.py   帧时间轴与半开区间字幕调度（SRT/ASS 共用）
  style.py      样式数据结构：Color/FontSpec(含斜体)/FontWeight/Position
  fonts.py      位图字体(含斜体切变) + Pillow TTF + 字体族注册表与回退链
  render.py     SRT 排版、瓦片缓存、RGBA 源覆盖混合
  ass_render.py ASS 排版：多样式 run、numpad 定位、卡拉OK/淡入/\move、底盒
  pipeline.py   ffprobe 探测、解码/编码双进程逐帧泵、音频直通
  cli.py        命令行入口（格式自动识别，SRT 样式参数 ASS 模式拒绝）
tests/          单元测试 + ffmpeg 端到端集成测试
demo/           示例视频、SRT 与 ASS 样例
```

## 已知边界

- 位图字体路径（无 Pillow 时）不含 CJK 字形，中文显示为空心方框；
  安装 Pillow 并存在系统 TTF 即自动获得完整字符集；
- 字重为九档语义映射：缺少原生 bold 文件时用合成粗体描厚，不支持
  可变字体的连续字重插值；斜体同理，缺失原生斜体时用切变合成 oblique；
- SRT 富文本标签（`<b>`、`{\\an8}` 等）按普通文本处理，不做样式解析；
- ASS 特效覆盖常用子集：粗斜体/下划线/删除线、颜色/透明度、字号/字间距、
  缩放、numpad 对齐、`\\pos`/`\\move`、`\\fad`/`\\fade`、卡拉OK、
  BorderStyle=3 底盒；**未覆盖**的标签安静跳过——不做智能自动换行
  （仅 `\\N` 硬换行生效，`\\q` 忽略）、不实现 `\\p` 矢量绘图（绘图文本
  丢弃）、不支持旋转 `\\fr`、模糊 `\\blur`、`\\t` 动画、逐字 3D 变换；
  卡拉OK 的 `\\kf`/`\\K` 扫光按到点整体变色近似，不做逐像素扫过动画；
- 输出编码固定为 H.264/yuv420p（容器可选 MP4/MKV）。
