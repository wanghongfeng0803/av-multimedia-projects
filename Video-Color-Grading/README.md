# Video-Color-Grading

视频亮度 / 对比度 / 色相 / 饱和度调整工具。纯 Python 标准库实现，ffmpeg / ffprobe 通过子进程调用。

## 功能

- 按帧调整四个通道：
  - 亮度（[-100, 100]）、对比度（[-100, 100]）；
  - **色相**旋转（[-180, 180] 度，逆时针为正）、**饱和度**（[-100, 100]，-100 完全去色、+100 翻倍）。
- **保留原音轨**（`-c:a copy` 直通，不重编码）。
- 参数超范围自动**钳制到边界并提示**（`UserWarning` + 结果消息），不中断批处理。
- 亮度/对比度用 256 字节**映射表，只构建一次**，全部帧 / 分块 / worker 进程复用；
  色相/饱和度为跨通道操作，按像素在 RGB↔HSV 间换算。
- 帧内行带**分块 + 多进程并行**映射；在途块数有界，**内存占用不随分辨率 / 时长增长**。
- 无参数调用（四项均为 0）为逐字节恒等映射，是固定的兼容基线。

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

## 用法

```bash
python grade.py input.mp4 -o output.mp4 -b 20 -c 10 -H 45 -s -20
# 或
python -m color_grading input.mp4 -o output.mp4 \
    --brightness 20 --contrast 10 --hue 45 --saturation -20
```

| 参数 | 短选项 | 范围 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `--brightness` | `-b` | [-100, 100] | 0 | 亮度平移 |
| `--contrast` | `-c` | [-100, 100] | 0 | 以中灰为支点的对比度 |
| `--hue` | `-H` | [-180, 180] | 0 | 色相旋转角（度），逆时针为正 |
| `--saturation` | `-s` | [-100, 100] | 0 | 饱和度缩放百分比 |

其它参数：`--workers N`（并行进程数）、`--chunk-rows N`（分块行数）、
`--crf` / `--preset`（x264 编码参数）、`--overwrite`（允许覆盖输出）。

作为库使用：

```python
from color_grading import adjust

result = adjust("in.mp4", "out.mp4",
                brightness=20, contrast=10, hue=45, saturation=-20)
print(result.frames, result.messages)  # messages 含钳制等提示
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
输入 ──ffmpeg 解码──> rgb24 裸流(RGB) ──分块并行──> rgb24 裸流(RGB) ──ffmpeg 编码──> 输出
                              │  每块内：① 亮度/对比度 LUT ② 色相 ③ 饱和度
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
  可用：新参数 `hue`、`saturation` 全部带默认值 0（仅在 `adjust` /
  `map_stream` 以关键字传入）；不传时走与旧版完全相同的恒等 / LUT 路径，
  无参数调用逐字节结果不变（由字节级回归测试锁定）。

## 测试

```bash
pip install -r requirements-dev.txt
pytest
```

纯逻辑测试（映射表、HSV、分块）不依赖 ffmpeg；集成测试在无 ffmpeg 的
环境自动跳过。
