# Video-Transition-Effects

两段素材之间的**交叉淡化（cross-fade）转场**工具，从零实现，仅依赖
Python 标准库与系统 ffmpeg/ffprobe。

## 核心承诺

- **成片总时长可推导**：成片总帧数恒为 `Nout = Na + Nb - No`
  （Na/Nb 为两段素材帧数，No 为重叠帧数），比硬接短恰好一个重叠区；
- **重叠区之外一帧不丢**：head 段逐帧对应 A、tail 段逐帧对应 B，
  帧映射在 `plan.py` 显式建账，解码帧数与写出帧数三方核对，不符即报错；
- **转场帧统一走帧读取器**：所有视频帧只经 `FrameReader` 的
  `ffmpeg → rawvideo rgb24` 管道进入 Python，帧序号即时间轴坐标；
- **音视频同帧淡化**：视频像素与音频 PCM 共用同一张帧映射表和同一套
  有理数权重（`w_b = k/(No-1)`，单帧重叠取 1/2），不会出现
  「画面淡了声音没淡」；
- **确定性整数运算**：混合全部用整数凸组合，结果可逐字节单测锁定。

## 用法

```bash
# 重叠 0.5 秒（按帧率换算为整数帧）
python -m video_transition a.mp4 b.mp4 -o out.mp4 --overlap 0.5 -y

# 直接指定重叠帧数（语义最精确），输出机器可读账目
python -m video_transition a.mp4 b.mp4 -o out.mp4 --frames 12 --json
```

输出示例（逐段账目）：

```
head=63 帧(纯A) + overlap=12 帧(交叉淡化) + tail=38 帧(纯B) = 113 帧；时长 4.520s
```

## 模块结构

| 模块 | 职责 |
| --- | --- |
| `probe.py` | ffprobe 探针，帧率保留 num/den 有理数，音轨有无单独探测 |
| `plan.py` | 纯逻辑时间轴：三段区间、`FrameJob` 取帧指令、淡化权重 |
| `blend.py` | RGB24 像素与 s16le PCM 的整数加权混合原语 |
| `audio.py` | 采样边界累积取整、整轨 PCM 淡化拼装（与视频同源） |
| `reader.py` | `FrameReader` 帧读取器、s16le PCM 解码（无音轨静音占位） |
| `pipeline.py` | 双路解码线程 + 有界队列 + 编码写入器 + 帧数三方核对 |
| `cli.py` | 命令行入口 |

## 设计要点

### 帧域账本

```
head     A[0, Na-No)        纯 A，Na-No 帧
overlap  A[Na-No, Na)       交叉淡化，No 帧
         B[0, No)
tail     B[No, Nb)          纯 B，Nb-No 帧
```

淡化首尾锚点精确：重叠区首帧 B 权重为 0（与 head 末帧的纯 A 连续）、
末帧 B 权重为 1（与 tail 首帧的纯 B 连续），边界无跳变帧。

### 音频采样对齐

第 f 帧的采样区间用累积取整边界 `round(f * sr / fps)` 定义，
A、B、输出三条时间轴用同一个函数求值，48kHz 配 NTSC 帧率时
每帧 1601/1602 采样交替也不会漂移。无音轨素材用等长静音 PCM 占位。

### 约束

- 两段素材必须**同帧率、同分辨率**（不做隐式缩放/重采样，避免账本失真）；
- 重叠帧数必须短于任一素材；
- 输出 H.264 + AAC，容器支持 `.mp4`/`.mkv`。

## 测试

```bash
python -m pytest
```

纯逻辑测试（规划账本、混合原语、采样数学）不依赖 ffmpeg；
端到端集成测试在无 ffmpeg 时自动 skip，覆盖帧数恒等式、逐帧映射、
音视频同帧淡化、静音占位、NTSC 有理帧率等场景。
