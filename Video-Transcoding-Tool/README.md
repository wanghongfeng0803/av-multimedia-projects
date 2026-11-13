# Video-Transcoding-Tool

把输入视频转码为 **H.264（libx264 / yuv420p）编码的 MP4** 文件的命令行工具。
基于系统 `ffmpeg` / `ffprobe`，Python 侧仅使用标准库，无需安装第三方运行时依赖。

## 环境要求

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/)（需包含 `libx264`），`ffmpeg` 与 `ffprobe` 在 PATH 中可用
  - 也可通过环境变量 `FFMPEG` / `FFPROBE` 显式指定可执行文件完整路径

## 快速开始

```bash
# 输出文件默认生成在输入旁：<输入名>_h264.mp4
python transcode.py input.mov

# 指定输出路径
python transcode.py input.mkv -o output.mp4

# 调节质量与速度（CRF 越小质量越高；preset 越慢压缩率越好）
python transcode.py input.mp4 --crf 20 --preset slow

# 覆盖已存在的输出
python transcode.py input.mp4 -o output.mp4 --overwrite

# 机器可读的 JSON 结果
python transcode.py input.mp4 --json
```

也可以用模块方式调用：`python -m transcode_tool input.mp4 -o output.mp4`。

作为库使用：

```python
from transcode_tool import transcode

result = transcode("input.mov", "output.mp4", crf=23, preset="medium")
print(result.video_codec, result.width, result.height)
```

## 对不规范输入的处理策略

工具会先用 `ffprobe` 按**文件实际内容**（而非扩展名）探测，再决定转码参数：

| 输入情况 | 处理方式 |
| --- | --- |
| 非 H.264 编码（H.265、MPEG-4、Xvid 等） | 统一重编码为 H.264 |
| 奇数宽/高（如 641×363） | 向下取整为偶数尺寸（yuv420p 硬性要求），并给出警告 |
| 非 yuv420p 像素格式（10bit、RGB、yuv444 等） | 转换为 yuv420p，保证通用播放器兼容 |
| 非 AAC 音频且 MP4 不支持 | 重编码为 AAC（默认 128k）；AAC/MP3/AC3 等可直接保留 |
| 无音频流 | 输出纯视频（`-an`），给出提示 |
| 多条音轨/字幕/数据流 | 只保留主视频流与第一条音频流，其余丢弃并提示 |
| 扩展名不是 .mp4 / 缺少扩展名 | 输出路径自动规范化为 `.mp4` |
| 输出路径是已存在目录 | 在该目录下生成 `<输入名>_h264.mp4` |
| 文件损坏/截断但可部分读取 | 以容错参数（`ignore_err`、重建索引/PTS）尽量挽救，问题记入警告 |
| 纯文本伪装成 .mp4、严重损坏、纯音频 | 明确报错，不产生半成品文件 |
| 输出文件已存在 | 默认拒绝覆盖；`--overwrite` 显式授权后才覆盖 |

所有转码产物都会再经一次 `ffprobe` 复核，确认视频编码确实为 `h264`；
转码失败时自动清理半成品，并把 ffmpeg 完整日志写到 `<输出>.ffmpeg.log`。

## 退出码

| 码 | 含义 |
| --- | --- |
| 0 | 成功 |
| 1 | 通用错误 / 参数不合法（如 CRF 超范围、preset 非法） |
| 2 | 找不到 ffmpeg / ffprobe |
| 3 | 输入文件不存在 |
| 4 | 输入不是有效视频媒体（无法解析或无视频流） |
| 5 | 输出文件已存在且未允许覆盖 |
| 6 | ffmpeg 执行失败 |

## 项目结构

```
transcode_tool/
├── __init__.py      # 对外 API
├── __main__.py      # python -m transcode_tool 入口
├── binaries.py      # ffmpeg/ffprobe 查找（PATH、环境变量、同目录伴随）
├── errors.py        # 异常类型与退出码映射
├── models.py        # MediaInfo / StreamInfo / TranscodeResult 数据模型
├── probe.py         # ffprobe 探测，容忍损坏文件
├── transcoder.py    # 核心：参数构建、滤镜决策、转码与产物复核
└── cli.py           # argparse 命令行
tests/               # pytest 端到端测试（自动生成各类不规范样本）
transcode.py         # 便捷脚本入口
```

## 运行测试

测试会用 ffmpeg 的 `lavfi` 源现场生成 H.265、奇数尺寸、RGB、无音频、
多音轨、截断文件、文本伪装等样本，覆盖探测、转码与 CLI 全链路：

```bash
pip install -r requirements-dev.txt
python -m pytest
```
