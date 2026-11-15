# Multi-stream-Video-Composition

双路视频左右并排合成工具：把两个视频按确定性策略对齐后拼接为单个规范的
MP4（H.264 + AAC）。对分辨率、时长、帧率、音频不一致的输入**不报错**，
而是按文档化的固定策略对齐，保证同样输入永远得到同样输出。

仅依赖 Python 标准库；ffmpeg / ffprobe 通过子进程调用。

## 安装与环境

- Python 3.10+
- 系统中安装 [FFmpeg](https://ffmpeg.org/) 并加入 `PATH`
  （或用环境变量 `FFMPEG` / `FFPROBE` 指定完整路径）

## 用法

```bash
# 基本合成（输出默认为 左路同目录/<左名>_x_<右名>.mp4）
python -m compose_tool 左视频.mp4 右视频.mp4 -o out.mp4

# 时长不一致：截断到较短者（默认 pad：短路冻结末帧补齐到较长者）
python -m compose_tool a.mp4 b.mp4 --duration-mode trim

# 等宽左右半屏（较窄一路居中加黑边）
python -m compose_tool a.mp4 b.mp4 --layout fit

# 覆盖自动推断：固定画布高度 720、帧率 30000/1001
python -m compose_tool a.mp4 b.mp4 --height 720 --fps 30000/1001

# 音频策略：mix 混音 / left 仅左 / right 仅右 / none 无音轨（默认 auto）
python -m compose_tool a.mp4 b.mp4 --audio-mode mix

# 只打印对齐规划不执行（调试用）
python -m compose_tool a.mp4 b.mp4 --dry-run

# 进度与机器可读输出
python -m compose_tool a.mp4 b.mp4 --progress --json
```

作为库调用：

```python
from compose_tool import compose, probe_media, plan_composition

result = compose("left.mp4", "right.mp4", "out.mp4",
                 duration_mode="pad", layout="auto")
print(result.to_dict())

# 也可以只做规划（纯逻辑，不调用 ffmpeg）
plan = plan_composition(probe_media("left.mp4"), probe_media("right.mp4"))
print(plan.to_dict())
```

## 确定性对齐策略

| 不一致项 | 默认策略 | 可选项 |
| --- | --- | --- |
| 分辨率 | 画布高 = 两路较大高度（向上取偶），每路按显示宽高比（含 SAR 修正）等比缩放，宽度取偶；输出宽 = 两路之和 | `--layout fit`：较窄一路居中黑边填充到等宽单元格；`--height` 覆盖画布高 |
| 时长 | `pad`：输出 = 较长者，短路冻结最后一帧（tpad clone），其音频补静音 | `--duration-mode trim`：输出 = 较短者，两路截断 |
| 帧率 | 输出 = 两路较大帧率，两路经 fps 滤镜统一为 CFR；帧率按有理数精确处理（30000/1001 不取整） | `--fps` 覆盖；两路都探测不到时回退 25fps 并告警 |
| 音频 | 两路都有 → 重采样为 48kHz 立体声后等权混音（amix normalize，防削波）；只有一路 → 用该路；都没有 → 无音轨 | `--audio-mode left/right/none/mix`；显式指定侧无音频时输出无声并告警（严格守约定） |
| 像素格式 | 统一转换为 yuv420p（hstack 要求两路一致，也是通用播放规范） | — |
| 时长未知 | 视频流时长 → 容器时长逐级回退；都无法确定时抛出 `InvalidInputError`（此时任何猜测都会破坏可复现性） | — |

## 输出格式规范

- 容器：MP4，`+faststart`（moov 前置，可流式播放）
- 视频：H.264（libx264）/ yuv420p / 偶数尺寸 / 恒定帧率
  （默认 CRF 20、preset veryfast，可用 `--crf` / `--preset` 调整）
- 音频：AAC 128k / 48kHz / 立体声（无音频输入时输出无音轨）
- 元数据：剥离输入元数据（`-map_metadata -1`）

## 可复现性

同版本 ffmpeg + 同输入 + 同参数 ⇒ 输出**逐字节一致**（有集成测试锁定）。
为此固定了所有不确定性来源：

- `-bitexact`：不写 muxer / 编码器版本串；
- `-map_metadata -1`：剥离输入元数据；
- `-video_track_timescale 90000`：固定输出时间基；
- 规划层全部取确定值：画布尺寸、帧率（有理数）、时长、补帧秒数
  都由纯函数算出，同输入必然同规划。

注意：x264 码流与线程数相关，跨机器（CPU 核数不同）复现需自行
固定 `-threads`；同机同版本下无需任何额外处理。

## 项目结构

```
compose_tool/
  plan.py         # 对齐规划（纯逻辑，确定性策略的核心）
  filtergraph.py  # 由规划生成 filter_complex
  probe.py        # ffprobe 探测
  composer.py     # 编排：探测→规划→命令→执行；输出规范与可复现开关
  runner.py       # ffmpeg 子进程运行器（流式进度、stderr 落盘）
  models.py       # 数据模型
  cli.py          # 命令行入口
tests/            # pytest 套件（纯逻辑测试不依赖 ffmpeg；
                  # 集成测试用 ffmpeg 生成样本，无 ffmpeg 自动 skip）
```

## 运行测试

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## 已知限制

- 仅支持两路左右拼接（项目名中的 multi-stream 指「两路输入流」）；
- 不处理旋转元数据（rotate/displaymatrix），竖屏素材请先自行转正；
- 不做色彩空间转换，源若为 bt2020/10bit 会先降为 yuv420p 8bit；
- 视频编码固定 libx264，需要本机 ffmpeg 带该编码器。
