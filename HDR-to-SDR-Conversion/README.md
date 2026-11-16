# HDR-to-SDR-Conversion

HDR（PQ / HLG）→ SDR（BT.709）转换工具。纯 Python 标准库实现核心算法，
ffmpeg / ffprobe 仅负责解码与编码的原始帧搬运。

## 功能特性

- **不靠猜的 HDR 类型识别**：只认 ffprobe 确证的 `color_transfer`
  （`smpte2084` → PQ，`arib-std-b67` → HLG）；字段缺失或为 SDR 时直接报错，
  必须显式 `--assume pq|hlg` 才放行。
- **完整转换链**（解码后、编码前逐像素处理）：
  EOTF（PQ ST.2084 / HLG BT.2100，含 HLG OOTF）→ 亮度色调映射 →
  BT.2020→BT.709 色域映射（保亮度压缩，不偏色）→ BT.709 OETF → 重编码。
- **不偏色**：色调映射按亮度加权（同系数作用 RGB），色域映射保持亮度与色相；
  引擎带中性轴保护，灰场不发彩。
- **不过曝**：色调曲线严格单调、输出上界为 1；色域压缩把越界色缩回 [0,1]，
  不做硬截断（`clip` 方法除外，仅供对照）。
- **两套 Reinhard**：默认 `reinhard` 为膝盖式亮度压缩（0.75·peak 以下恒等，
  行为保持不变）；`reinhard_classic` 为经典 L/(1+L) 全局压缩，白点
  `--white` 可调（越大压缩越温和），参数越界自动钳制并提示。
- **软编/硬编同引擎**：Python 引擎在编码前产出确定的原始帧，软编
  （libx264/libx265）与硬编（nvenc/qsv/amf 共 6 个编码器）画面数学完全
  一致，仅末端编码器不同；同输入两次运行逐字节确定。
- **位深/色彩空间自适应**：自动探测码深（8/10/12/16）、limited/full range、
  YUV/RGB 像素族；YUV 走 `yuv444p16le`、RGB 走 `gbrpf32le` 两条解码路径。
- **清 HDR 元数据**：编码侧通过编码器内联 VUI 参数显式写入 BT.709 三件套
  （primaries/transfer/matrix），并 `-map_metadata -1` 丢弃容器级元数据，
  输出不含任何 PQ/HLG 语义。

## 安装与依赖

- Python 3.10+，**仅用标准库**，无需 pip 安装第三方包；
- ffmpeg / ffprobe 在 PATH（或用 `FFMPEG` / `FFPROBE` 环境变量指定）。

## 命令行用法

```bash
# 基本转换（自动识别 PQ/HLG，默认 Reinhard、libx264、8bit limited）
python -m hdr_to_sdr input_hdr.mkv output_sdr.mp4

# 只探测不转换
python -m hdr_to_sdr input.mkv - --probe-only

# HLG、10bit、x265、BT.2390 曲线、4 进程分带并行
python -m hdr_to_sdr input.mkv out.mkv --assume hlg --10bit \
    --codec libx265 --tone bt2390 --workers 4

# 经典 Reinhard，白点 2.0（高光保留更多），NVIDIA 硬编
python -m hdr_to_sdr input.mkv out.mp4 --tone reinhard_classic --white 2.0 \
    --codec h264_nvenc
```

主要参数：

| 参数 | 说明 |
| --- | --- |
| `--tone` | `reinhard`（默认，膝盖 0.75·peak 单调有理滚降）/ `reinhard_classic`（经典 L/(1+L) 全局压缩，白点可调）/ `bt2390`（膝盖 0.7）/ `clip`（硬截断，对照用） |
| `--white W` | 仅经典 Reinhard 生效的白点（相对 peak 的倍数，默认 1.0）；越大压缩越温和，越界钳制到 [1e-3, 100] 并提示 |
| `--codec` | 软编 `libx264`（默认）/`libx265`；硬编 `h264_nvenc`/`hevc_nvenc`/`h264_qsv`/`hevc_qsv`/`h264_amf`/`hevc_amf` |
| `--10bit` | 输出 10bit，默认 8bit |
| `--full-range` | 输出 full range，默认 limited/TV range |
| `--assume` | 元数据不可信时显式指定 `pq`/`hlg`，正常情况无需使用 |
| `--workers N` | 帧内水平分带并行进程数，默认 1（串行，结果逐像素确定） |
| `--probe-only` | 只打印探测结果 |
| `--progress N` | 每 N 帧打印进度 |

## Python API

```python
from hdr_to_sdr import probe_hdr, convert_hdr_to_sdr, FrameProcessor

info = probe_hdr("input.mkv")          # 确证 PQ/HLG，缺失则抛 UnsupportedHDRTypeError
result = convert_hdr_to_sdr("input.mkv", "output.mp4", tone_method="bt2390")
```

## 模块结构

```
hdr_to_sdr/
├── errors.py        # 异常体系
├── binaries.py      # ffmpeg/ffprobe 查找
├── transfer.py      # PQ/HLG EOTF、BT.709 OETF、1D LUT
├── colorscience.py  # 色域矩阵、YUV↔RGB 量化、保亮度色域压缩
├── tonemap.py       # 膝盖式/经典 Reinhard、BT.2390、clip 色调映射
├── encoders.py      # 软编/硬编编码器规格、质量参数与 VUI 翻译、可用性探测
├── probe.py         # ffprobe 探测与 HDR 类型确证
├── frames.py        # FrameProcessor 逐像素引擎 + 分带并行
├── pipeline.py      # 双 ffmpeg 进程管线、音频直通/AAC 回退、元数据清除
└── cli.py           # 命令行入口
tests/               # pytest（纯数学不依赖 ffmpeg，集成/硬编测试无对应能力自动 skip）
```

## 关键设计决策

1. **峰值归一化**：线性光统一到归一化域（PQ 的 1.0 = 10000nit）。PQ 峰值取
   mastering display 的 MaxMDL（缺失默认 1000nit）；HLG 经 OOTF 后显示域
   1.0 即标称峰值，参考白落在约 0.8。
2. **色调映射的数学约束**：「膝盖下恒等 + f(peak)=1 + 上界 1 + 严格单调」
   不可同时满足，故膝盖设在峰值之下（Reinhard 0.75、BT.2390 0.7），
   保证输出严格有界不过曝且高光不反转。
3. **经典 Reinhard 的有界形式**：教科书扩展 Reinhard（Reinhard 2002 白点式
   x(1+x/w²)/(1+x)）在输入超过白点后输出会突破 1，违背不过曝约束；故经典
   模式采用恒有界单极点有理曲线 f(x)=wx/(1+wx)，white=1 即 L/(1+L)、
   渐近线恒为 1，white 越大肩部越靠后、压缩越温和。
4. **软编/硬编一致性**：画面处理全部在编码前的 Python 引擎完成，软编与硬编
   拿到的是同一批确定原始帧，差异只来自末端编码器的量化取整。VUI 写码流
   方式不同（软编实测必须 `-x264-params` 内联，硬编外层 `-color_*` 即生效），
   由 `encoders.py` 按编码器族分别翻译；硬编运行时可用性以真实一帧编码冒烟
   确证，不能只看 ffmpeg 是否编入。
5. **色域映射两步走**：先标准 3×3 矩阵转 BT.709 原色，再对越界像素沿
   「通道−亮度」方向等比缩回，亮度严格不变、色相保持，只降饱和度。
6. **VUI 标签内联**：实测 ffmpeg 9.0 的 libx264/libx265 外层
   `-color_trc/-color_primaries` 不写进码流，必须用 `-x264-params/-x265-params`
   内联 `colorprim/transfer/colormatrix` 才能被 ffprobe 确证为 BT.709。

## 测试

```bash
pytest          # 纯数学测试始终运行；需要 ffmpeg 的集成测试在无 ffmpeg 时 skip
```
