# MP4 → MKV 无损重封装工具

把 ISO-BMFF/MP4 容器中的媒体**无损重封装（remux）**为 Matroska/MKV 容器。
全程只在容器层操作，**不做任何解码或重编码**——每一帧、每一个音频包的
编码字节都原样拷贝，因此画质/音质零损失。

纯 Python 标准库实现，无任何第三方运行时依赖。

## 特性

- **真正无损**：H.264/HEVC 视频帧、AAC 音频包逐字节原样写入，附
  `ffmpeg` 端到端测试用 `framemd5` 证明解码结果逐比特一致。
- **时间戳正确**：
  - 处理 `stts` / `ctts`（含 version 1 有符号合成偏移），正确表达 B 帧；
  - 应用 edit list（`edts/elst`）的时间轴偏移；
  - 用 Matroska 的 `CodecDelay` 表达 AAC 编码器前导延迟（priming），
    用 `BlockGroup + DiscardPadding` 表达尾部填充裁剪。
- **多轨道交织**：视频在前、音频在后，跨轨道按归一化 DTS（精确分数）排序。
- **容错与清晰报错**：对损坏盒结构、越界偏移、计数不一致等给出中文提示；
  明确拒绝碎片化 MP4（`moof`）与加密媒体（`sinf`/`encv`/`enca`）。
- 支持 64 位 `largesize` 盒、`co64` 偏移、`mdhd/ctts` version 0/1。

## 支持范围（基础版本）

| 项目 | 支持 |
| --- | --- |
| 视频编码 | H.264/AVC（`avc1`/`avc3`）、HEVC（`hvc1`/`hev1`） |
| 音频编码 | AAC（`mp4a`，私有数据为 AudioSpecificConfig） |
| 容器布局 | 非分片 MP4；moov 在头部（faststart）或尾部均可 |
| 每轨 sample entry | 仅 1 个 |
| edit list | 单段、rate=1.0 |
| 输入规模 | 面向小规模数据，整体读入内存，上限 512 MiB |

不支持（会明确报错而非产出坏文件）：碎片化 MP4、加密/DRM 媒体、
多段或变速 edit list、多 sample entry 的轨道、其它编码格式。

## 用法

### 命令行

```bash
python -m mp4_to_mkv input.mp4 output.mkv [--overwrite]
```

示例：

```
$ python -m mp4_to_mkv movie.mp4 movie.mkv
完成：movie.mp4 -> movie.mkv
  轨道数 2，样本数 70，输出 14230 字节（全程未重编码）
```

无效输入会打印中文错误并以退出码 `2` 结束：

```
$ python -m mp4_to_mkv missing.mp4 out.mkv
错误：输入文件不存在: missing.mp4
```

### Python API

```python
from mp4_to_mkv.mp4_parser import parse_mp4
from mp4_to_mkv.mkv_writer import write_mkv

with open("input.mp4", "rb") as f:
    mp4 = parse_mp4(f.read())
mkv_bytes = write_mkv(mp4)
with open("output.mkv", "wb") as f:
    f.write(mkv_bytes)
```

## 工作原理

```
MP4 字节流
   │  mp4_parser：递归解析 box 树，展开样本表
   │  stts/stsc/stsz/stco/co64/ctts/stss → 每样本 offset/size/DTS/PTS/关键帧
   │  avcC/hvcC/esds → CodecPrivate；edts/elst → 时间轴偏移
   ▼
Mp4File（tracks + samples，帧数据仍指向原缓冲，零拷贝）
   │  mkv_writer：
   │  · EBML 头 / Info / Tracks（含 CodecDelay、CodecPrivate）
   │  · 跨轨按 DTS 排序、毫秒时间戳换算、AAC priming 计算
   │  · Cluster 切分，SimpleBlock（常规）/ BlockGroup（尾部裁剪）
   ▼
MKV 字节流
```

### 几个关键约定

- **时间基**：MKV `TimestampScale` 取默认值 1,000,000，即所有时间戳单位为 1 毫秒。
- **SimpleBlock 承载 PTS**：帧仍按 DTS 解码顺序写出，块的 16 位时间码写
  显示时间戳，B 帧重排由解码器依据码流自身的参考关系完成（与 libavformat 一致）。
- **Cluster 切分**：达到 5 MiB 后在下一个视频关键帧切分，硬上限 100 MiB；
  并保证块的相对时间码不超出 int16 范围。
- **AAC priming**：MP4 用 edit list 的 `media_time` 表示编码器前导延迟，
  转成 MKV 的 `CodecDelay`（纳秒）；末包被截短的时长差转成 `DiscardPadding`，
  从而与源文件解码出的 PCM 采样严格对齐。

## 项目结构

```
mp4_to_mkv/
├── __init__.py
├── __main__.py        # CLI 入口（python -m mp4_to_mkv）
├── errors.py          # RemuxError 统一异常
├── ebml.py            # EBML/VINT 编码基础
├── matroska_ids.py    # Matroska 元素 ID 与 CodecID 常量
├── mp4_parser.py      # ISO-BMFF/MP4 解析与样本表展开
├── mkv_writer.py      # Matroska 结构构建与写出
└── remuxer.py         # 读取→解析→写出 的编排与输入校验
tests/
├── mp4_builder.py          # 手工构造自洽 MP4 的测试夹具
├── ebml_reader.py          # 最小 EBML 读取器（仅测试用）
├── test_ebml.py            # EBML 已知答案向量
├── test_mp4_parser.py      # 解析器与容错
├── test_remuxer.py         # 端到端结构/无损/CLI
└── test_integration_ffmpeg.py  # 真实文件 + framemd5 无损验证
```

## 测试

```bash
python -m unittest discover -s tests
```

- 前三个测试文件不依赖任何外部工具，用手工构造的 MP4/已知字节向量验证；
- `test_integration_ffmpeg.py` 在检测到本机装有 `ffmpeg/ffprobe` 时自动运行，
  覆盖真实 H.264(+B 帧)/AAC、HEVC、faststart、分片拒绝等场景，并用
  `framemd5` 逐帧哈希比对证明无损；无 ffmpeg 时自动跳过。

## 退出码

| 码 | 含义 |
| --- | --- |
| 0 | 成功 |
| 2 | 输入无效或包含不支持的结构（中文提示） |
| 3 | 内存不足等非预期错误 |
