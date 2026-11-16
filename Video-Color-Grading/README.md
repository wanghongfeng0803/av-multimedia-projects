# Video-Color-Grading

视频亮度 / 对比度 / 色相 / 饱和度调整工具，支持把 **3D LUT（.cube）作为独立调色节点**接入链路。纯 Python 标准库实现，ffmpeg / ffprobe 通过子进程调用。

## 功能

- 按帧调整四个通道：
  - 亮度（[-100, 100]）、对比度（[-100, 100]）；
  - **色相**旋转（[-180, 180] 度，逆时针为正）、**饱和度**（[-100, 100]，-100 完全去色、+100 翻倍）。
- **3D LUT 独立节点**：解析 Adobe ``.cube``（三线性插值），可按节点位置与其它步骤串联，强度 0–1 连续混合。
- **保留原音轨**（`-c:a copy` 直通，不重编码）。
- 参数超范围自动**钳制到边界并提示**（`UserWarning` + 结果消息），不中断批处理。
- 亮度/对比度用 256 字节**映射表，只构建一次**，全部帧 / 分块 / worker 进程复用；
  色相/饱和度为跨通道操作，按像素在 RGB↔HSV 间换算；3D LUT 只解析一次、随节点链分发。
- 帧内行带**分块 + 多进程并行**映射；在途块数有界，**内存占用不随分辨率 / 时长增长**。
- 无参数调用（四项均为 0、不挂 3D LUT）为逐字节恒等映射，是固定的兼容基线。
- 解析 / 域 / 尺寸失败时**保留原图、不产出半成品**，错误带文件名与行号可定位；同一输入两次输出逐字节一致。

## 通道顺序（全局统一，不得一处一套）

整条像素管道统一使用 **RGB** 字节序：ffmpeg 解码侧固定 `-pix_fmt rgb24`，
裸流中每个像素按 `R, G, B` 排列；`colorspace` / `chunks` / 编码侧
`-pix_fmt rgb24` 全部遵循同一约定。代码中**不出现 BGR 命名，也不提供
BGR 入口**。若将来接入 OpenCV 等默认 BGR 的库，必须在边界处显式转回
RGB 后再进入本管道。该约定由 `tests/test_chunks.py::test_rgb_channel_order_is_locked`
（红 `+120°` 必须得到纯绿）锁定。

## 处理顺序（固定，写死不可调换）

一个像素依次经过：

1. **亮度 / 对比度** 1D LUT（逐通道）；
2. **色相**旋转（在 HSV 空间）；
3. **饱和度**缩放（同一 HSV 空间）。

即色相与饱和度同时生效时固定 **先色相、后饱和度**。两者在纯 HSV 标量
运算上数学可交换，但实现上固定一个顺序，保证任何调用方、任何 worker
进程都得到逐字节一致的结果。该顺序由
`tests/test_colorspace.py::test_combined_order_matches_hue_then_saturation_reference`
和 `tests/test_chunks.py::test_lut_then_hsv_fixed_order_is_byte_exact` 锁定。

## 3D LUT 节点（.cube，三线性插值）

3D LUT 被建模为调色链路里的一个**独立节点**（`color_grading.lut3d` +
`color_grading.nodes.LUT3DNode`），与亮度/对比度、色相/饱和度节点地位
相同，可以按显式顺序串联。

### .cube 解析：宽容噪声，严格拒绝语义错误

- **容错**：`#` 注释（含行内）、空行、CRLF/LF、多余空白、关键字大小写、
  带不带引号的 `TITLE`、整数或浮点数据。
- **直接报错（不硬套）**：
  - 尺寸问题 → `LUTParseError`：缺/重复 `LUT_3D_SIZE`、N 超出 [2,256]、
    数据点数与 N³ 不符、数据行出现在尺寸声明之前、列数不为 3、
    token 不是数值、出现 NaN/Inf、`LUT_1D_SIZE`（1D LUT 不伪装成 3D）。
  - 域问题 → `LUTDomainError`：只接受未声明域或显式单位域
    （`DOMAIN_MIN 0 0 0`、`DOMAIN_MAX 1 1 1`）。任何带偏移/缩放/负数的
    非单位域无法在 rgb24 字节域无损套用，一律拒绝而非静默重映射。
  - 所有错误携带 `file_path` 与 `line_number`（整文件级问题行号为
    `None`），CLI 直接打印「文件名: 第 N 行: …」，可定位。
- 数据展平顺序按 Adobe 约定 **R 增长最快、G 次之、B 最慢**。

### 强度 0–1：两端边界精确

强度 `strength` ∈ [0,1] 在**输出字节空间**对原色与全量 LUT 结果做线性混合：
`out = round(orig + (lut_out - orig) * strength)`。

- `strength == 0`：节点恒等，输出与输入**逐字节相同**（跳过浮点往返）；
- `strength == 1`：输出与全量 LUT**逐字节相同**；
- 超范围自动钳制到 [0,1] 并提示（与其它参数同一策略）。

### 节点顺序：显式、可预期

- CLI / `adjust()` 用 `--lut3d-position` / `lut3d_position` 选择
  `first`（在亮度/对比度/色相/饱和度之前）或 `last`（默认，在其后）。
- 需要更自由的组合时用节点化入口 `adjust_nodes(input, output, nodes)`，
  `nodes` 是有序列表，**列表顺序就是像素处理顺序**；同一节点可重复出现，
  调换位置即改变函数复合顺序，结果确定且可重复（由
  `tests/test_nodes.py` 锁定「换位结果不同、各自可重复」）。

### 失败原子性与确定性

- 解析、域校验、强度钳制全部在启动 ffmpeg **之前**完成；这些环节失败时
  根本不会创建输出文件，原图不动。
- 像素管道 / ffmpeg 阶段失败时，本次运行新建的输出文件会被删除，预先
  存在的输出绝不改动（`tests/test_lut3d_pipeline.py` 覆盖两条路径）。
- 纯标准库、固定运算次序，同一输入 + 同一 LUT + 同一强度，任意进程 /
  任意次数输出逐字节一致。

## 用法

```bash
python grade.py input.mp4 -o output.mp4 -b 20 -c 10 -H 45 -s -20
# 或挂一张 3D LUT（强度 0.6，置于链路最前）
python -m color_grading input.mp4 -o output.mp4 \
    --lut3d film.cube --lut3d-strength 0.6 --lut3d-position first
```

| 参数 | 短选项 | 范围 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `--brightness` | `-b` | [-100, 100] | 0 | 亮度平移 |
| `--contrast` | `-c` | [-100, 100] | 0 | 以中灰为支点的对比度 |
| `--hue` | `-H` | [-180, 180] | 0 | 色相旋转角（度），逆时针为正 |
| `--saturation` | `-s` | [-100, 100] | 0 | 饱和度缩放百分比 |
| `--lut3d` | — | .cube 路径 | 无 | 3D LUT 文件，不给则不启用该节点 |
| `--lut3d-strength` | — | [0, 1] | 1 | 0 精确原图、1 全量 LUT，超范围钳制 |
| `--lut3d-position` | — | first / last | last | 3D LUT 节点在链路中的位置 |

其它参数：`--workers N`（并行进程数）、`--chunk-rows N`（分块行数）、
`--crf` / `--preset`（x264 编码参数）、`--overwrite`（允许覆盖输出）。

作为库使用：

```python
from color_grading import adjust, adjust_nodes, load_cube
from color_grading import BrightnessContrastNode, HueSaturationNode, LUT3DNode

# 简单用法：四参数 + 一张 3D LUT
result = adjust("in.mp4", "out.mp4",
                brightness=20, contrast=10, hue=45, saturation=-20,
                lut3d_path="film.cube", lut3d_strength=0.7,
                lut3d_position="last")
print(result.frames, result.lut3d_path, result.lut3d_strength)

# 节点化用法：顺序完全由列表控制（这里 LUT 在最前，再接亮度/色相）
cube = load_cube("film.cube")
chain = [
    LUT3DNode(cube, 0.7),
    BrightnessContrastNode(),
    HueSaturationNode(hue=30, saturation=-10),
]
adjust_nodes("in.mp4", "out.mp4", chain)
```

## 色相方向如何核对（不靠肉眼）

色相旋转的正确性用**色相环测试图 + 数值断言**核对，而非主观目测：

- 单元层：主色 / 间色色块（红 0°、黄 60°、绿 120°、青 180°、蓝 240°、
  品红 300°）旋转 ±60° 后的结果由代码断言；2 万随机像素与标准库
  `colorsys` 的独立 HSV 实现交叉比对，零误差。
- 端到端：生成一张水平铺满 0–360° 的连续色相环视频，经本工具旋转后，
  与 **ffmpeg 原生 `hue=h=60` 滤镜**的输出逐列比较色相角，整环最大偏差
  小于 12°（yuv420p 有损压缩容差），方向一致、逆时针为正。

## 设计说明

```
输入 ──ffmpeg 解码──> rgb24 裸流(RGB) ──分块并行节点链──> rgb24 裸流(RGB) ──ffmpeg 编码──> 输出
                              │  每块内按节点顺序，例如：
                              │    ① 亮度/对比度 1D LUT ② 色相 ③ 饱和度 ④ 3D LUT(strength)
                              │  （3D LUT 也可经 position=first 置于最前）
  └────────────────── 音轨 -c:a copy 原样直通 ───────────────────────────────┘
```

- **亮度 / 对比度映射公式**：`out = clamp(round((v - 128) * factor + 128 + offset))`，
  对比度以中灰 128 为支点，亮度为整体平移。两参数为 0 时
  `(v-128)*1.0+128.0` 在双精度下无舍入误差，逐字节恒等。
- **色相 / 饱和度**：RGB→HSV（H 以度表示），先把色相角加 `hue` 对 360°
  取模，再把饱和度乘 `(100+saturation)/100` 并钳到 [0,1]，最后 HSV→RGB；
  灰度像素（R=G=B）色相无定义，跳过旋转，饱和度对纯灰无效。
- **分块**：每块是一帧内连续的若干行（默认 128 行），块边界不跨帧，
  写出顺序与读入一致；内存上界 ≈ `在途块数 × 块行数 × 宽 × 3`。
- **并行**：`ProcessPoolExecutor`，LUT 与色相/饱和度参数经 initializer
  一次性分发给各 worker。
- **健壮性**：两条 ffmpeg 子进程的 stderr 由独立线程排空，避免管道
  写满死锁；编码器提前退出时主动终止解码器并报告 stderr 尾部。
- 奇数宽/高会自动向下缩放为偶数（yuv420p 限制），并在结果消息中说明。

## 兼容性

- 旧的 `adjust(...)` / `build_lut(...)` / `map_stream(...)` 调用签名保持
  可用：新参数 `hue`、`saturation` 以及 3D LUT 的 `lut3d_*` 全部带默认值
  （`lut3d_path=None` 即不启用）；不传时走与旧版完全相同的恒等 / LUT 路径，
  无参数调用逐字节结果不变（由字节级回归测试锁定，原 79 个测试无一修改）。
- `map_stream` 旧的 `lut` / `hue` / `saturation` 三参数保留，内部转成等价
  节点链；新的 `nodes` 节点链参数与旧三参数互斥，混用直接 `ValueError`，
  避免「哪套顺序生效」的歧义。
- 新增异常 `LUTParseError`（退出码 8）、`LUTDomainError`（退出码 9），
  均派生自统一基类 `ColorGradingError`，旧的 `except ColorGradingError`
  捕获逻辑不受影响。

## 测试

```bash
pip install -r requirements-dev.txt
pytest
```

纯逻辑测试（映射表、HSV、分块）不依赖 ffmpeg；集成测试在无 ffmpeg 的
环境自动跳过。
