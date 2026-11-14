# Video-Transcoding-Tool

把输入视频转码为 **H.264 / H.265(HEVC) / VP9 / AV1 / VP8** 等多种编码的命令行工具，
支持 **MP4 / WebM / MKV** 容器、大文件分段转码与实时进度。
基于系统 `ffmpeg` / `ffprobe`，Python 侧**仅使用标准库**，所有底层处理（编码器能力
探测、进度流解析、分段规划、容器兼容决策、时间戳处理）均自行实现，无需第三方运行时依赖。

## 环境要求

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/)，`ffmpeg` 与 `ffprobe` 在 PATH 中可用
  - 也可通过环境变量 `FFMPEG` / `FFPROBE` 显式指定可执行文件完整路径
  - 各编码需要 ffmpeg 构建包含对应库：`libx264`、`libx265`、`libvpx`/`libvpx-vp9`、
    `libsvtav1`（或 `libaom`/`librav1e`）。工具会自动探测本机可用编码器并按优先级降级。

## 快速开始

```bash
# 默认：H.264(yuv420p) + MP4，输出 <输入名>_h264.mp4
python transcode.py input.mov

# 指定编码（H.265 / VP9 / AV1 / VP8），容器按编码自动选择
python transcode.py input.mp4 --codec h265
python transcode.py input.mp4 --codec vp9          # 自动输出 .webm
python transcode.py input.mp4 --codec av1          # 自动输出 .mp4

# 显式指定容器与输出路径
python transcode.py input.mkv -o out.webm --codec vp9 --container webm

# 调节质量与速度（CRF 范围随编码不同；preset 支持命名档或数字档）
python transcode.py input.mp4 --codec h265 --crf 28 --preset slow

# 大文件：按每段时长或段数分段转码、无损拼接，带实时进度
python transcode.py huge.mkv --codec h265 --segment-duration 300 --progress
python transcode.py huge.mkv --segment-count 8 --progress

# 覆盖已存在的输出、机器可读 JSON、查看本机编码支持情况
python transcode.py input.mp4 -o out.mp4 --overwrite --json
python transcode.py --list-codecs
```

也可以用模块方式调用：`python -m transcode_tool input.mp4 -o out.mp4 --codec av1`。

作为库使用：

```python
from transcode_tool import transcode, probe_media, list_codecs

# 结果包含 video_codec / container / encoder 等字段
result = transcode("input.mov", "out.mkv", codec="h265", crf=28, preset="medium",
                   segment_count=4,
                   progress_callback=lambda r: print(f"{r*100:.1f}%"))
print(result.video_codec, result.container, result.encoder, result.width, result.height)

# 查看注册表（纯数据，不启动进程）
for spec in list_codecs():
    print(spec.canonical, spec.crf_min, spec.crf_max, spec.default_crf)
```

## 支持的编码与容器

| 编码（别名） | ffmpeg 编码器（按优先级） | CRF 范围 | 默认 CRF | 首选容器 |
| --- | --- | --- | --- | --- |
| `h264`（avc/x264） | libx264 | 0–51 | 23 | mp4, mkv |
| `h265`（hevc/x265） | libx265（自动加 hvc1 标签） | 0–51 | 28 | mp4, mkv |
| `vp9` | libvpx-vp9（常量质量 + row-mt） | 0–63 | 31 | webm, mkv |
| `vp8` | libvpx | 4–63 | 31 | webm, mkv |
| `av1`（av01） | libsvtav1 → libaom-av1 → librav1e | 0–63 | 30 | mp4, webm, mkv |

**容器 × 视频编码兼容矩阵**（不兼容的组合在 ffmpeg 启动前就明确报错）：

| 容器 | H.264 | H.265 | VP9 | VP8 | AV1 | 音频策略 |
| --- | --- | --- | --- | --- | --- | --- |
| mp4 | ✅ | ✅ | ❌ | ❌ | ✅ | aac/mp3/ac3/eac3/alac 直通，否则转 AAC |
| webm | ❌ | ❌ | ✅ | ✅ | ✅ | opus/vorbis 直通，否则转 Opus（缺编码器降级 Vorbis） |
| mkv | ✅ | ✅ | ✅ | ✅ | ✅ | 常见编码几乎都可直通，否则转 Opus |

`preset` 统一使用 x264 风格十档命名（`ultrafast` … `placebo`），由各编码模块翻译成
编码器原生参数；libvpx/libaom/SVT-AV1/rav1e 也接受原生数字档位（如 `"4"`）。

## 大文件分段转码

一次性转码超大视频会导致单次 ffmpeg 运行极长、失败即从头再来。启用分段后：

1. 按 ffprobe 时长把任务切成均匀的小段（`--segment-duration` 或 `--segment-count`）；
2. 每段独立编码到临时目录，采用 ffmpeg **双 `-ss`** 模式（输入侧快速关键帧寻址 +
   输出侧时间戳归零），既保证大文件性能又保证帧精确边界，规避 MKV 等容器的时间戳错位；
3. 用 `concat` 分离器以**流拷贝**方式无损拼接（不二次编码、不损画质）；
4. 无论成功失败都清理临时目录；进度按各段时长加权聚合成全局 0–100%。

段数有硬上限（1000）与最小段长（1s）保护，异常时长输入会自动回退为单段整体转码并告警。

## 对不规范输入的处理策略

工具先用 `ffprobe` 按**文件实际内容**（而非扩展名）探测，再决定转码参数：

| 输入情况 | 处理方式 |
| --- | --- |
| 非目标编码（H.265、MPEG-4、Xvid 等） | 按 `--codec` 重编码 |
| 0 字节文件 | 明确报 `InvalidInputError`，不启动 ffmpeg |
| 奇数宽/高 | 向下取整为偶数尺寸（yuv420p 硬性要求），并告警 |
| 非 yuv420p 像素格式（10bit、RGB、yuv444 等） | 转换为 yuv420p，保证通用兼容 |
| 音频与容器不兼容 | 按容器策略重编码（MP4→AAC，WebM/MKV→Opus），兼容则无损直通 |
| 无音频流 | 输出纯视频（`-an`），给出提示 |
| 多音轨/字幕/数据流 | 只保留主视频流与第一条音频流，其余丢弃并提示 |
| 扩展名缺失/与容器不符 | 输出扩展名自动规范化为容器对应扩展名 |
| 输出路径是已存在目录 | 在该目录下生成 `<输入名>_<编码>.<扩展名>` |
| 文件损坏/截断但可部分读取 | 容错参数（`ignore_err`、重建索引/PTS）尽量挽救，问题记入警告 |
| 纯文本伪装、严重损坏、纯音频 | 明确报错，不产生半成品文件 |
| 本机缺少目标编码的全部编码器 | 报 `EncoderUnavailableError`（退出码 7），附候选编码器信息 |
| 编码与容器不兼容 | 启动前报 `ContainerError`（退出码 1） |
| 输出文件已存在 | 默认拒绝覆盖；`--overwrite` 显式授权后才覆盖 |

所有产物都会再经一次 `ffprobe` 复核，确认视频编码与目标一致；转码失败时自动清理半成品，
并把 ffmpeg 完整日志写到 `<输出>.ffmpeg.log`（成功时不留日志）。

## 命令行参数

```
input                 输入视频（按内容探测，扩展名不限）
-o, --output PATH     输出路径；省略时在输入旁生成 <名>_<编码>.<容器扩展名>
--codec CODEC         h264 / h265 / vp9 / vp8 / av1（含别名，默认 h264）
--container NAME      mp4 / webm / mkv；省略时按输出扩展名推断，再取编码首选容器
--crf N               质量参数，范围随编码；省略用编码默认值
--preset NAME         十档命名或编码器原生数字档位
--audio-bitrate RATE  音频重编码码率（默认 128k）
--segment-duration S  分段单段时长（秒），与 --segment-count 二选一
--segment-count N     分段数量
--progress            在 stderr 实时显示进度百分比
--overwrite           覆盖已存在的输出
--json                JSON 格式输出结果摘要
--list-codecs         列出本机 ffmpeg 的编码/编码器/容器支持情况后退出
```

## 退出码

| 码 | 含义 |
| --- | --- |
| 0 | 成功 |
| 1 | 通用错误 / 参数不合法（CRF 超范围、preset 非法、编码与容器不兼容等） |
| 2 | 找不到 ffmpeg / ffprobe |
| 3 | 输入文件不存在 |
| 4 | 输入不是有效视频媒体（无法解析、无视频流或 0 字节） |
| 5 | 输出文件已存在且未允许覆盖 |
| 6 | ffmpeg 执行失败 |
| 7 | 本机 ffmpeg 不支持目标编码所需的任何编码器 |

## 项目结构

```
transcode_tool/
├── __init__.py       # 对外 API
├── __main__.py       # python -m transcode_tool 入口
├── binaries.py       # ffmpeg/ffprobe 查找（PATH、环境变量、同目录伴随）
├── errors.py         # 异常类型与退出码映射（含 EncoderUnavailableError）
├── models.py         # MediaInfo / StreamInfo / TranscodeResult 数据模型
├── codecs.py         # 编码注册表：CRF/preset 语义、编码器候选、ffmpeg 参数构建
├── containers.py     # 容器注册表：兼容矩阵、扩展名、音频直通/重编码策略
├── capabilities.py   # 解析 ffmpeg -encoders 探测本机能力（缓存 + 乐观降级）
├── probe.py          # ffprobe 探测，容忍损坏文件
├── runner.py         # Popen 流式进度解析、stderr 落盘的低内存 ffmpeg 运行器
├── segments.py       # 分段规划、concat 无损拼接、临时目录管理
├── transcoder.py     # 核心编排：探测→规划→编码(可分段)→拼接→复核
└── cli.py            # argparse 命令行
tests/                # pytest 单元与端到端测试（自动生成各类不规范/分段样本）
transcode.py          # 便捷脚本入口
```

## 设计要点

- **模块化、数据驱动**：新增一种编码只需在 `codecs.py` 注册一个 `CodecSpec`，
  新增一种容器只需在 `containers.py` 注册一个 `ContainerSpec`，编排层无需改动。
- **纯标准库**：能力探测解析原始文本、进度解析 ffmpeg 机器可读 `-progress` 键值流、
  分段全为纯数值逻辑，均不依赖第三方包。
- **面向大文件**：分段切片 + 快速寻址 + 流拷贝拼接；stderr 边产生边落盘、
  仅保留尾部于内存，长任务不会因日志膨胀占用大量 RAM。
- **防御式输入处理**：所有外部文本解析为不抛异常的纯函数，异常/截断/伪装输入
  都有受控的错误类型与退出码。

## 运行测试

测试用 ffmpeg 的 `lavfi` 源现场生成 H.265、奇数尺寸、RGB、10bit、无音频、多音轨、
截断文件、文本伪装、稀疏 GOP 分段等样本，覆盖探测、各编码转码、分段拼接与 CLI 全链路：

```bash
pip install -r requirements-dev.txt
python -m pytest
```
