# MP4 → MKV 无损重封装工具

把 ISO-BMFF/MP4 容器中的媒体**无损重封装（remux）**为 Matroska/MKV 容器。
全程只在容器层操作，**不做任何解码或重编码**——每一帧、每一个音频包的
编码字节都原样拷贝，因此画质/音质零损失。

纯 Python 标准库实现，无任何第三方运行时依赖。

## 特性

- **真正无损、全轨道保留**：视频帧、音频包、字幕文档逐字节原样写入。
  视频支持 H.264/HEVC，音频支持 AAC/AC-3/E-AC-3/MP3，字幕支持 TTML/WebVTT；
  附 `ffmpeg` 端到端测试用 `framemd5` 证明解码结果逐比特一致。
- **时间戳正确**：
  - 处理 `stts` / `ctts`（含 version 1 有符号合成偏移），正确表达 B 帧；
  - 应用 edit list（`edts/elst`）的时间轴偏移；
  - 用 Matroska 的 `CodecDelay` 表达各音频编码的前导延迟（priming），
    用 `BlockGroup + DiscardPadding` 表达尾部填充裁剪。
- **字幕完整保留**：TTML 样本（完整 XML 文档）直拷；WebVTT 的 `vttc` 盒
  重组为 Matroska `S_TEXT/WEBVTT`，cue 正文入 Block，cue id/settings 经
  `BlockAdditions`（BlockAddID 1/2）保留，空 cue（`vtte`）也不丢。
- **章节（chapter）元数据字节级保留**：解析 `moov/udta/meta/chpl` 章节清单，
  转写为 Matroska `Chapters`（EditionEntry/ChapterAtom/ChapterDisplay）。
  - 多层嵌套（`gnre`/`ilst` 等容器）与 FullBox/普通盒两种 `meta` 布局均能定位；
  - 兼容真实 ffmpeg/GPAC 的 **100ns 事实时间单位**（自动识别）与 movie
    timescale 规范单位，时间戳精确换算为纳秒；标题长度按 uint8 Pascal 风格解析；
  - 容错损坏/不规则输入：负载与标题截断按可用字节恢复、计数不符以实际为准、
    非法 UTF-8 替换解码、空标题回退确定性占位名，且章节解析失败绝不影响媒体；
  - 每章带 `ChapterTimeEnd`，零时间戳显式编码（避免 libavformat 丢弃首章），
    ChapterUID 取归一化序号，同一输入反复处理**逐字节可复现**；
  - 同时存在的 QuickTime 章节文本轨（`handler=text` + `gmhd`）被识别并跳过，
    不再因不支持文本编码而拒绝整个文件。
- **多轨道交织**：轨道排列固定为 视频 → 音频 → 字幕，同类保持源顺序，
  跨轨道按归一化 DTS（精确分数）排序；每类首轨为默认轨，语言码透传。
- **容错与清晰报错**：对损坏盒结构、越界偏移、计数不一致等给出中文提示；
  明确拒绝碎片化 MP4（`moof`）与加密媒体（`sinf`/`encv`/`enca`）。
- 支持 64 位 `largesize` 盒、`co64` 偏移、`mdhd/ctts` version 0/1。

## 支持范围

| 项目 | 支持 |
| --- | --- |
| 视频编码 | H.264/AVC（`avc1`/`avc3`）、HEVC（`hvc1`/`hev1`） |
| 音频编码 | AAC（`mp4a`+esds）、MP3（`mp4a` 无 ASC 或 objectType=0x6B）、AC-3（`ac-3`）、E-AC-3（`ec-3`） |
| 字幕编码 | TTML（`stpp`，文档直拷）、WebVTT（`wvtt`，vttc 重组） |
| 轨道数量 | 不限：视频/音频/字幕各若干条，全部保留 |
| 容器布局 | 非分片 MP4；moov 在头部（faststart）或尾部均可 |
| 每轨 sample entry | 仅 1 个 |
| edit list | 单段、rate=1.0 |
| 输入规模 | 面向小规模数据，整体读入内存，上限 512 MiB |

不支持（会明确报错而非产出坏文件）：碎片化 MP4、加密/DRM 媒体、
多段或变速 edit list、多 sample entry 的轨道、表中以外的编码格式
（如 `tx3g`/mov_text 位图字幕、`hint`/`meta` 等非媒体轨）。

## 用法

### 命令行

基础入口（单文件转换，无重试）：

```bash
python -m mp4_to_mkv input.mp4 output.mkv [--overwrite]
```

带失败重试的全轨道入口（推荐，保留视频/音频/字幕全部轨道）：

```bash
python remux_all_tracks.py input.mp4 output.mkv [--overwrite]
                           [--retries N] [--retry-delay SECONDS] [--temp]
```

`python -m mp4_to_mkv` 基础入口同样接受 `--temp`。默认流式直写；
`--temp` 切换为“`.part` 临时文件 + 原子替换”，覆盖过程中目标始终完整。

重试策略：仅对**瞬时性 I/O 错误**（文件被占用、网络盘抖动等 `OSError`）
按线性退避重试，默认额外重试 3 次；输入不支持/损坏等**确定性错误**
（`RemuxError`）立即失败、绝不重试。默认**流式直写**目标文件——
Segment 用未知大小 VINT、Cluster 逐簇落盘，全程不产生 `.part` 临时文件、
内存里同一时刻只保留一个簇；写失败自动清理半成品目标。需要覆盖过程也
保持原子性时加 `--temp`，退回“临时文件 + `os.replace`”。退出码：
`0` 成功，`2` 输入无效，`3` 重试用尽。

示例：

```
$ python remux_all_tracks.py movie.mp4 movie.mkv
完成：movie.mp4 -> movie.mkv
  轨道数 3（视频/音频/字幕全保留），样本数 70，输出 14230 字节，尝试 1 次（全程未重编码）
```

无效输入会打印中文错误并以退出码 `2` 结束：

```
$ python -m mp4_to_mkv missing.mp4 out.mkv
错误：输入文件不存在: missing.mp4
```

#### 批量重封装

批量入口在单文件重试管线之上做「发现 → 规划 → 逐项隔离执行」，
可一次处理文件与目录混排的一批输入：

```bash
python batch_remux.py OUT_DIR INPUT [INPUT ...] [--recursive/--no-recursive]
                       [--on-existing {rename,overwrite,skip}]
                       [--dry-run] [--stop-on-error]
                       [--retries N] [--retry-delay SECONDS]
# --overwrite / --skip-existing 是 --on-existing 的快捷写法（三者互斥）
```

- **输入发现**：目录默认递归扫描，只收 `.mp4/.m4v/.mov/.m4a` 普通文件，
  显式给出的文件不做后缀限制；按解析后的绝对路径去重，处理顺序确定可复现。
- **输出碰撞**：所有输出平铺到 `OUT_DIR`，默认 `rename` 策略把同名文件
  自动改名为 `stem_2.mkv`、`stem_3.mkv`……（非破坏，绝不覆盖已有文件）；
  `overwrite` 走单文件管线的原子替换覆盖；`skip` 跳过已存在的输出。
- **故障隔离**：单个文件失败（不支持的结构、重试用尽等）只记录该项的
  中文原因，其余文件照常处理；`--stop-on-error` 可切换为遇错即停。
- `--dry-run` 只打印发现结果与输出规划，不创建目录、不执行转换。
- 退出码：`0` 全部成功（含全跳过/空批），`2` 整批前置错误（如输入不存在），
  `5` 批中至少一项失败。

示例（两个不同目录下的同名 `song.mp4` 自动分流为两个输出）：

```
$ python batch_remux.py mkv_out album
[成功] album\live\song.mp4 -> mkv_out\song.mkv（尝试 1 次）
[成功] album\song.mp4 -> mkv_out\song_2.mkv（尝试 1 次）
汇总：共 2 项，成功 2，失败 0，跳过 0
```

### Python API

直接使用底层包。内存路径（一次拼出完整字节，输出带已知大小 Segment）：

```python
from mp4_to_mkv.mp4_parser import parse_mp4
from mp4_to_mkv.mkv_writer import write_mkv

with open("input.mp4", "rb") as f:
    mp4 = parse_mp4(f.read())
mkv_bytes = write_mkv(mp4)  # 全部 track 一并写出
with open("output.mkv", "wb") as f:
    f.write(mkv_bytes)
```

流式路径（无临时文件、内存有界，可写管道等不可寻址流；Segment 用
未知大小 VINT，块负载与内存路径逐字节等价）：

```python
from mp4_to_mkv.mp4_parser import parse_mp4
from mp4_to_mkv.mkv_writer import write_mkv_stream

with open("input.mp4", "rb") as f:
    mp4 = parse_mp4(f.read())
with open("output.mkv", "wb") as out:
    n = write_mkv_stream(mp4, out)   # 逐簇 write，返回写出字节数
```

带重试的编排层：

```python
from remux_all_tracks import remux_all_tracks, RemuxRetryError
from mp4_to_mkv.errors import RemuxError

try:
    result = remux_all_tracks("in.mp4", "out.mkv", retries=3)
    print(result.track_count, result.sample_count, result.attempts)
except RemuxError:
    ...      # 确定性输入错误，重试无意义
except RemuxRetryError:
    ...      # 瞬时性错误，重试用尽
```

批量编排层（逐项结果可程序化检查，单文件失败不会中断整批）：

```python
from batch_remux import remux_batch

report = remux_batch(["movie.mp4", "dir_in"], "out_dir",
                     on_existing="rename", dry_run=False)
for item in report.results:
    print(item.status, item.source, item.output, item.reason)
# 汇总属性：report.total / report.succeeded / report.failed / report.skipped
```

## 工作原理

```
MP4 字节流
   │  mp4_parser：递归解析 box 树，展开样本表
   │  stts/stsc/stsz/stco/co64/ctts/stss → 每样本 offset/size/DTS/PTS/关键帧
   │  avcC/hvcC/esds → CodecPrivate；edts/elst → 时间轴偏移
   ▼
Mp4File（tracks + samples，帧数据仍指向原缓冲，零拷贝）
   │  mkv_writer（两条等价路径，共用样本放置/时间戳/簇切分逻辑）：
   │  · EBML 头 / Info / Tracks（含 CodecDelay、CodecPrivate）
   │  · 跨轨按 DTS 排序、毫秒时间戳换算、AAC priming 计算
   │  · Cluster 切分，SimpleBlock（常规）/ BlockGroup（尾部裁剪）
   │  · write_mkv：内存拼整份（已知大小 Segment）
   │  · write_mkv_stream：未知大小 Segment + 逐簇落盘（流式、无临时文件）
   ▼
MKV 字节流
```

### 几个关键约定

- **时间基**：MKV `TimestampScale` 取默认值 1,000,000，即所有时间戳单位为 1 毫秒。
- **SimpleBlock 承载 PTS**：帧仍按 DTS 解码顺序写出，块的 16 位时间码写
  显示时间戳，B 帧重排由解码器依据码流自身的参考关系完成（与 libavformat 一致）。
- **Cluster 切分**：达到 5 MiB 后在下一个视频关键帧切分，硬上限 100 MiB；
  并保证块的相对时间码不超出 int16 范围。
- **流式 vs 内存两条写出路径**：块负载、时间戳、轨道与章节完全相同，
  唯一区别是 Segment 的 size VINT——内存路径写已知大小（最短 VINT），
  流式路径写 8 字节未知大小 VINT（`0x01 0xFF×7`，与 ffmpeg 直播流一致），
  Segment 由流末尾隐式闭合，因此无需预知总长度、不 seek、不暂存整份输出。
- **音频 priming（通用）**：MP4 用 edit list 的 `media_time` 表示编码器
  前导延迟（AAC/AC-3/E-AC-3/MP3 均可能携带），统一转成 MKV 的
  `CodecDelay`（纳秒）；末包被截短的时长差转成 `DiscardPadding`，
  从而与源文件解码出的 PCM 采样严格对齐。
- **字幕块**：字幕统一用 `BlockGroup`（含 `BlockDuration` 表达 cue 时长）；
  TTML 文档原样入 Block，WebVTT 的 cue id/settings 入 `BlockAdditions`。

## 项目结构

```
batch_remux.py          # 批量重封装编排：发现/规划/逐项隔离（CLI + Python API）
remux_all_tracks.py     # 带失败重试的全轨道重封装编排（CLI + Python API）
mp4_to_mkv/
├── __init__.py
├── __main__.py        # 基础 CLI 入口（python -m mp4_to_mkv）
├── errors.py          # RemuxError 统一异常
├── ebml.py            # EBML/VINT 编码基础
├── matroska_ids.py    # Matroska 元素 ID 与 CodecID 常量
├── mp4_parser.py      # ISO-BMFF/MP4 解析与样本表展开（含字幕/新音频编码）
├── webvtt.py          # wvtt(vttc) → Matroska WebVTT 块转换
├── mkv_writer.py      # Matroska 结构构建与写出（全轨道交织）
└── remuxer.py         # 读取→解析→写出 的编排与输入校验
tests/
├── mp4_builder.py          # 手工构造自洽 MP4（含多轨道）的测试夹具
├── ebml_reader.py          # 最小 EBML 读取器（仅测试用）
├── test_ebml.py            # EBML 已知答案向量
├── test_mp4_parser.py      # 解析器与容错
├── test_remuxer.py         # 端到端结构/无损/CLI
├── test_all_tracks.py      # 全轨道（视频/音频/字幕）结构与无损
├── test_retry.py           # 失败重试的边界与清理
├── test_batch.py           # 批量发现/规划/故障隔离/CLI（39 个用例）
├── test_streaming.py       # 流式写出：未知大小 Segment、不可寻址流、
│                           #   无临时文件、失败清理/重试、两路径等价（18 个用例）
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
| 0 | 成功（批量入口中，全部成功、全跳过或空批均为 0） |
| 2 | 输入无效或包含不支持的结构（中文提示）；批量入口的整批前置错误同码 |
| 3 | 单文件入口：重试用尽；`python -m mp4_to_mkv`：内存不足等非预期错误 |
| 5 | 批量入口：批中至少一项失败（其余项可能已成功） |
