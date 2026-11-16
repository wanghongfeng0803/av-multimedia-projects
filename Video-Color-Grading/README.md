# Video-Color-Grading

视频亮度 / 对比度调整工具。纯 Python 标准库实现，ffmpeg / ffprobe 通过子进程调用。

## 功能

- 按帧调整亮度（[-100, 100]）与对比度（[-100, 100]），**保留原音轨**（`-c:a copy` 直通，不重编码）。
- 参数超范围自动**钳制到边界并提示**（`UserWarning` + 结果消息），不中断批处理。
- 256 字节**映射表只构建一次**，全部帧 / 分块 / worker 进程复用。
- 帧内行带**分块 + 多进程并行**映射；在途块数有界，**内存占用不随分辨率 / 时长增长**。
- 无参数调用（亮度 0、对比度 0）为逐字节恒等映射，是固定的兼容基线。

## 用法

```bash
python grade.py input.mp4 -o output.mp4 -b 20 -c 10
# 或
python -m color_grading input.mp4 -o output.mp4 --brightness 20 --contrast 10
```

常用参数：`--workers N`（并行进程数）、`--chunk-rows N`（分块行数）、
`--crf` / `--preset`（x264 编码参数）、`--overwrite`（允许覆盖输出）。

作为库使用：

```python
from color_grading import adjust

result = adjust("in.mp4", "out.mp4", brightness=20, contrast=10)
print(result.frames, result.messages)  # messages 含钳制等提示
```

## 设计说明

```
输入 ──ffmpeg 解码──> rgb24 裸流 ──分块并行 LUT 映射──> rgb24 裸流 ──ffmpeg 编码──> 输出
  │                                                                          ▲
  └────────────────── 音轨 -c:a copy 原样直通 ───────────────────────────────┘
```

- **映射公式**：`out = clamp(round((v - 128) * factor + 128 + offset))`，
  对比度以中灰 128 为支点，亮度为整体平移。两参数为 0 时
  `(v-128)*1.0+128.0` 在双精度下无舍入误差，逐字节恒等。
- **分块**：每块是一帧内连续的若干行（默认 128 行），块边界不跨帧，
  写出顺序与读入一致；内存上界 ≈ `在途块数 × 块行数 × 宽 × 3`。
- **并行**：`ProcessPoolExecutor`，LUT 经 initializer 分发给各 worker，
  worker 只做一次 `bytes.translate`。
- **健壮性**：两条 ffmpeg 子进程的 stderr 由独立线程排空，避免管道
  写满死锁；编码器提前退出时主动终止解码器并报告 stderr 尾部。
- 奇数宽/高会自动向下缩放为偶数（yuv420p 限制），并在结果消息中说明。

## 测试

```bash
pip install -r requirements-dev.txt
pytest
```

纯逻辑测试（映射表、分块）不依赖 ffmpeg；集成测试在无 ffmpeg 的
环境自动跳过。
