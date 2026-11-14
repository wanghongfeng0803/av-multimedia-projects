# HLS MPEG-TS 切片打包工具（HLS Segment Packaging）

把任意本地媒体文件按 **GOP（关键帧）边界**封装为 MPEG-TS（`.ts`）分片，
并生成符合 RFC 8216 的 M3U8 播放列表。**全程流拷贝、不重编码**，
分片策略与播放列表完全由本项目自己控制，不使用 `ffmpeg -f hls` 的成品输出。

## 功能特性

- **GOP 对齐切片**：切点只落在 IDR 关键帧上，每个分片首帧可独立解码，
  播放器任意 seek / 起播都不会花屏。切点策略可选 `nearest`（默认，
  距理想切点最近）/ `before`（片长不超过目标值）/ `after`（片长不短于
  目标值）三种对齐模式。
- **自定义分片时长**：`-d/--segment-duration` 接受任意正浮点秒数，
  规划器在关键帧序列中按选定模式就近切点；末段不足时长时自然收尾，
  纯音频输入的过短尾片自动并入前一片。
- **可配置命名模板**：`--name-template` 支持 `{name}` 前缀、
  `{sequence}` 绝对序号（MEDIA-SEQUENCE）、`{index}` 运行内序号、
  `{ext}` 容器扩展名四个占位符，序号支持任意位宽零填充
  （如 `{sequence:05d}`）；默认模板 `{name}_{sequence:04d}.{ext}`
  与历史文件名逐字节一致。模板含路径分隔符 / Windows 保留字符 /
  渲染重名时在切片前明确报错。
- **多种分片封装**：`--segment-format` 支持 `ts`（MPEG-TS，默认）、
  `fmp4`（自包含碎片化 MP4 / CMAF 风格，每片带 ftyp 可独立解码）、
  `mp4`（普通 ISO-BMFF）、`mkv`（Matroska）；播放列表版本随容器
  自动选择（TS=3，ISO-BMFF=7），各容器有独立的流拷贝编码白名单。
- **时间戳连续**：各分片保留全局连续的 PTS/DTS，跨分片首尾相接、
  无回退、无跳变；拼接后 DTS 严格单调。
- **规范的 M3U8**：自带 `EXT-X-VERSION`、`EXT-X-TARGETDURATION`、
  `EXT-X-MEDIA-SEQUENCE`、`EXT-X-PLAYLIST-TYPE`、`EXTINF`、
  `EXT-X-INDEPENDENT-SEGMENTS`、`EXT-X-ENDLIST` 等标签，
  `TARGETDURATION` 按实际最大片长向上取整，保证每片 `EXTINF` 都不超限。
- **点播 / 直播两种列表**：默认 VOD 全量列表；可指定滑动窗口，
  通过 `MEDIA-SEQUENCE` 表达窗口滚动。
- **处理高效**：一次 ffmpeg 调用 + `-c copy` 流拷贝，无解码/重编码，
  开销接近纯 IO；关键帧探测优先走只解复用的包级查询。
- **元数据落盘**：`segments.json` 记录每片的源内起点、输出时间轴起点、
  实测时长、序号，便于下游核对。
- **文字水印（可选）**：完全自研文字栅格化——手写 UTF-8 解码器、
  TrueType（.ttf/.ttc）解析器、扫描线超采样抗锯齿与 PNG 编码，零第三方
  依赖；自带 5x7 点阵字体作为字体全部缺失时的最终回退。叠加走
  `overlay` 过滤器，参数全部以独立 argv 传递、路径不进 filtergraph，
  杜绝注入；不指定水印时行为与纯流拷贝版本完全一致。
- **AES-128 分片加密（可选，RFC 8216）**：纯标准库从零实现 AES-128 块密码
  （FIPS-197）、CBC 模式（SP 800-38A）与 PKCS#7 填充，通过 NIST 已知答案
  向量与 openssl 字节级交叉验证；按规范输出 `EXT-X-KEY`（METHOD/URI/IV），
  支持随机 IV 与 RFC §5.2 序号派生 IV；密钥/IV/URI 异常均有明确中文报错，
  不启用时与旧版本逐字节等价。
- **零第三方 Python 依赖**：仅用标准库，外部只依赖系统 `ffmpeg`/`ffprobe`。

## 目录结构

```
hls_packager/
├── __init__.py      # 公开 API
├── __main__.py      # python -m hls_packager 入口
├── cli.py           # argparse 命令行
├── errors.py        # 异常体系（环境/探测/切片/播放列表/水印）
├── binaries.py      # ffmpeg / ffprobe 可执行文件发现
├── process.py       # 子进程执行封装
├── models.py        # 不可变数据模型（StreamInfo/MediaInfo/SegmentInfo...）
├── probe.py         # ffprobe 探测 + 关键帧时间序列提取
├── planner.py       # GOP 对齐分片规划（纯函数，无 IO，含三种对齐模式）
├── naming.py        # 命名模板：占位符渲染、零填充、安全校验、ffmpeg 模式翻译
├── containers.py    # 分片容器规格（ts/fmp4/mp4/mkv 的 muxer/扩展名/版本/编码白名单）
├── ts_writer.py     # segment muxer 一次成型切片 + 逐片复测（含水印分支）
├── m3u8.py          # M3U8 渲染与合规校验（纯函数，含 EXT-X-KEY）
├── packager.py      # 编排层：probe → plan → 切片 → 复测 → 加密 → 播放列表
├── watermark/       # 文字水印：自研栅格化 + 安全叠加（仅依赖标准库）
│   ├── utf8.py          # 手写 UTF-8 解码器（拒绝过长/代理/截断/越界）
│   ├── bitmap_font.py   # 内置 5x7 点阵字体（零依赖最终回退）
│   ├── ttf_font.py      # 自研 TrueType/TTC 解析（cmap/glyf/hmtx/loca...）
│   ├── rasterizer.py    # 扫描线 + 4x 超采样抗锯齿字形栅格化
│   ├── font_stack.py    # 字体发现与按码点回退链（显式→系统→点阵→豆腐块）
│   ├── rasterize.py     # 排版绘制：码点序列 → RGBA 位图 → PNG
│   ├── png_buf.py       # RGBA 位图与自研 PNG 编码
│   ├── escaping.py      # 过滤器路径转义 + shell/argv 引号包裹（防注入）
│   ├── overlay.py       # 九宫格定位 + ffmpeg overlay segment 命令构造
│   └── config.py        # WatermarkConfig 不可变配置与校验
└── encryption/      # AES-128 分片加密（RFC 8216，纯标准库从零实现）
    ├── aes_cbc.py       # AES-128（FIPS-197）+ CBC（SP 800-38A）+ PKCS#7
    ├── config.py        # 密钥/IV/URI 解析校验、EncryptionConfig、序号 IV 派生
    ├── key_info.py      # EXT-X-KEY 描述模型与标签行渲染
    └── segment_io.py    # 分片流式加密 + 原子替换、密钥文件落盘
tests/               # pytest：纯逻辑单测 + ffmpeg/openssl 集成测试
```

数据单向流动，模块之间只通过 `models.py` 里的不可变 dataclass 通信：

```
输入文件
  │  probe_media()        ffprobe：流信息 + 全部关键帧时间
  ▼
MediaInfo
  │  plan_segments()      纯函数：在关键帧中选最近切点
  ▼
[SegmentInfo 规划]
  │  write_all_segments() 一次 ffmpeg segment muxer，流拷贝
  ▼
[WrittenSegment 实测时长]
  │  _rebuild_timeline()  用实测时长重建连续时间轴
  ▼
[SegmentInfo 终态] ──► (可选) encrypt_segment_file() 逐片 AES-128-CBC
  │                                              + write_key_file() 裸密钥
  ├──────────────► render_playlist() ──► playlist.m3u8（含/不含 EXT-X-KEY）
  └──────────────► _write_metadata() ──► segments.json
```

## 核心设计与正确性保证

### 1. GOP 对齐算法（`planner.py`）

每个分片从当前边界 `current` 出发，理想切点为 `current + 目标时长`，
再按 `--align-mode` 在关键帧中选点：

- `nearest`（默认）：距理想切点**最近**的关键帧，距离相同取**靠前**
  的一个（保证分片不会无谓变长）；
- `before`：只取理想切点之前（含重合）最近的关键帧，非末片时长不会
  超过目标值；首个 GOP 就长于目标值、前侧无候选时退取之后最近的
  关键帧，避免切出零长度分片；
- `after`：只取理想切点之后（含重合）最近的关键帧，非末片时长不会
  短于目标值。

无论哪种模式：

- 所有切点必然是关键帧时刻，分片首帧即 IDR；
- **大 GOP 场景**（关键帧间隔 > 目标时长）下分片会比目标长——这是
  GOP 对齐的必然代价，`TARGETDURATION` 会按实际最大片长自动放大；
- 纯音频 / 无关键帧信息时退化为等长规划，过短的尾片自动并入前一片，
  对齐模式对其无意义（切点不受关键帧约束）。

**末段与精度处理**：理想切点越过源末尾时不再切，剩余内容自然并入
末片；内部时间比较统一走 1ms 容差（`_EPS`）吸收容器时间戳抖动，
落盘值 round 到 6 位小数（微秒级，高于 MPEG-TS 90kHz 时钟的单 tick
精度）；计划片长之和与源时长严格相等，实际片长以切后 ffprobe
复测为准。

### 2. 为什么用 segment muxer 而不是逐片 `-ss/-t`

最初的逐进程方案（每片一个 `ffmpeg -ss X -t Y`）在 **B 帧重排**与
**AAC priming（编码器延迟采样）**下会过切：`-t` 按输出包起点计时，
音频 priming 会让容器多吞下一个 GOP 的关键帧，导致分片变长、混入额外 IDR。

ffmpeg 的 `segment` muxer 在解复用器内部按关键帧边界切分，正确处理
重排与 priming。本工具仍然**完全掌握切片策略**——切点通过
`-segment_times` 显式传入（来自自己的规划器），播放列表也由自己生成，
segment muxer 只负责"在给定时刻做流拷贝切割"，并未使用 ffmpeg 的
hls muxer 或其自带播放列表。

### 3. 时间戳连续性（`ts_writer.py` + `packager.py`）

- 切片**不启用** `-reset_timestamps`，各分片保留全局时间戳，相邻分片
  PTS/DTS 首尾相接；每个 TS 分片又都由 muxer 写入独立的 PAT/PMT，
  因此既能独立解析，又在时间轴上连续。
- 编排层用 ffprobe **逐片复测实际时长**，并以"前序实测时长之和"重建
  输出时间轴，使 `EXTINF` 之和与分片真实时长严格一致，避免帧级漂移累积。

### 4. M3U8 合规（`m3u8.py`）

渲染前做硬性自检，违规直接抛错而不是产出非法播放列表：

- `TARGETDURATION = ceil(max(分片时长))`，且断言每片不超过它；
- 分片序号必须连续；直播窗口的 `MEDIA-SEQUENCE` 与窗口首片严格对齐；
- `EXTINF` 统一 3 位小数浮点秒；VOD 列表带 `ENDLIST`，直播列表不带。

### 5. AES-128 分片加密（`encryption/`，RFC 8216 §4.3.2.4 / §5.2）

加密是切片之后的可选旁路，与切片、水印完全解耦：

```
明文 TS 分片（复测通过）
  │  encrypt_segment_file()   64 KiB 流式 CBC 加密 → 临时文件 → os.replace 原子替换
  ▼
密文 TS 分片（长度向上取整到 16 字节倍数）+ encryption.key（16 字节裸密钥）
  │  render_key_tag()         EXT-X-KEY 写入首个 EXTINF 之前
  ▼
带 EXT-X-KEY 的 playlist.m3u8
```

- **算法正确性**：AES-128 块密码严格按 FIPS-197 实现（S 盒由 GF(2^8)
  乘法逆元 + 仿射变换程序化生成，避免手抄常量表出错）；CBC 链式加密按
  NIST SP 800-38A；非块对齐数据按 PKCS#7（RFC 5652）补齐，去填充时
  严格校验填充值与填充区一致性，密文损坏/密钥不匹配会明确报错。
  测试内置 FIPS-197 附录 B、SP 800-38A 官方向量，并在装有 openssl 时
  做 9 组随机长度的双向字节级交叉验证。
- **IV 处理**：默认随机生成 16 字节显式 IV 并写入 `EXT-X-KEY` 的
  `IV=0x...`；也支持 RFC §5.2 的隐式序号 IV——IV 为分片
  MEDIA-SEQUENCE 的 128 位大端表示，此时播放列表省略 `IV` 属性，
  与非零 `--media-sequence` 组合时按绝对序号派生。
- **容错边界**：密钥/IV 接受原始字节、32 位 hex、Base64 与密钥文件
  四种来源，长度或格式非法、URI 含引号/控制字符/非 ASCII、密钥文件
  缺失或为空，全部以中文 `EncryptionError`（`PackagerError` 子类）
  抛出，CLI 统一走退出码 2。
- **失败安全**：加密先写同目录 `.enc.tmp` 临时文件，全部密文 fsync 后
  才 `os.replace` 原子覆盖；中途 IO 失败会清理临时文件，原明文分片不动。
- **不启用即等价**：`encryption=None` 时该旁路完全不执行，播放列表
  无 `EXT-X-KEY`、分片首字节仍是 TS 同步字节 `0x47`，由回归测试锁定。

### 6. 命名模板（`naming.py`）

模板是一段普通字符串，四个占位符：

| 占位符 | 含义 | 示例 |
| --- | --- | --- |
| `{name}` | 文件名前缀（`-p/--filename-prefix` 的值） | `seg` |
| `{sequence}` | 绝对序号，即 EXT-X-MEDIA-SEQUENCE | `0, 1, 2 ...` |
| `{index}` | 运行内序号，与 MEDIA-SEQUENCE 起始值无关，恒从 0 起 | `0, 1, 2 ...` |
| `{ext}` | 容器扩展名（由 `--segment-format` 决定，不含点） | `ts` / `m4s` |

序号占位符支持零填充宽度说明：`{sequence:05d}` 产出 `00042`；
不写宽度（`{index}`）则不补零。模板最终被翻译成 ffmpeg segment
muxer 的计数器模式串（`%05d` 风格），因此非零填充格式说明
（如 `{index:>5}`）会被拒绝；一个模板有且只能有一个序号占位符。

安全边界：模板字面部分禁止路径分隔符与 Windows 保留字符
（`<>:"/\\|?*`）、控制字符与字面 `%`；全部文件名在切片前预渲染
并检测重名，非法输入在任何 ffmpeg 调用之前报 `NamingError`。

### 7. 多容器封装（`containers.py`）

容器只影响**封装层**，GOP 规划、时间戳连续性、逐片复测与加密
逻辑完全复用：

| 短名 | ffmpeg muxer | 扩展名 | HLS 版本 | 说明 |
| --- | --- | --- | --- | --- |
| `ts`（默认） | `mpegts` | `.ts` | 3 | RFC 8216 原生格式，兼容性最好 |
| `fmp4` | `mp4` + `empty_moov` 等 movflags | `.m4s` | 7 | 自包含碎片化 MP4，每片带 ftyp、无需独立 init 段即可独立起播 |
| `mp4` | `mp4` | `.mp4` | 7 | 普通 ISO-BMFF 分片，通用播放器可直接打开 |
| `mkv` | `matroska` | `.mkv` | 3 | Matroska，编码承载面最宽（含 FLAC/Vorbis/AV1） |

每种容器维护独立的流拷贝编码白名单：主视频编码不被承载时直接
报错，个别不支持的音频轨跳过（与既有策略一致）。

## 安装与环境

需要系统提供 `ffmpeg` 与 `ffprobe`（建议 4.4+）。可执行文件查找顺序：

1. 环境变量 `FFMPEG_BIN` / `FFPROBE_BIN`；
2. `PATH`；
3. Python 解释器同目录（便携部署）。

无需安装 Python 第三方库；运行测试才需要 pytest：

```bash
pip install -r requirements-dev.txt
```

## 命令行用法

```bash
# 基本：VOD 点播，目标分片 6 秒
python -m hls_packager input.mp4 output_dir/

# 指定目标分片时长（实际切点对齐最近关键帧）
python -m hls_packager input.mp4 output_dir/ --segment-duration 4

# 切点不超过目标时长（before 模式），适合要求片长有上限的归档场景
python -m hls_packager input.mp4 output_dir/ -d 4 --align-mode before

# 自定义命名模板：5 位零填充绝对序号 + fMP4 容器
python -m hls_packager input.mp4 output_dir/ -d 4 \
  --segment-format fmp4 \
  --name-template "{name}-{sequence:05d}.{ext}"

# 文件名按运行内序号编号（MEDIA-SEQUENCE 仍可从任意值起算）
python -m hls_packager input.mp4 output_dir/ -s 100 \
  --name-template "{name}_{index:03d}.{ext}"

# 输出 Matroska 分片（编码承载面更宽）
python -m hls_packager input.mp4 output_dir/ --segment-format mkv

# 直播滑动窗口（保留最近 3 片）+ 自定义起始序号与文件名前缀
python -m hls_packager input.mp4 output_dir/ -d 6 -w 3 -s 100 -p chunk

# 允许写入非空输出目录（覆盖同名产物）
python -m hls_packager input.mp4 output_dir/ -y

# 叠加文字水印（视频轨重编码，音频仍流拷贝；不指定水印则保持流拷贝）
python -m hls_packager input.mp4 output_dir/ -d 6 -y \
  --watermark-text "直播水印 HLS 2026!" \
  --watermark-size 30 --watermark-color white \
  --watermark-anchor bottom-right

# 指定显式字体（可重复传入形成回退链）与九宫格位置
python -m hls_packager input.mp4 output_dir/ -y \
  --watermark-text "机密" --watermark-font C:/Windows/Fonts/msyh.ttc \
  --watermark-anchor top-left --watermark-margin-x 16 --watermark-margin-y 16

# 启用 AES-128 加密：自动生成随机密钥与随机 IV，密钥写入输出目录
python -m hls_packager input.mp4 output_dir/ -d 6 -y --encrypt

# 使用已有密钥（32 位 hex 或 Base64）与显式 IV
python -m hls_packager input.mp4 output_dir/ -y \
  --key 2b7e151628aed2a6abf7158809cf4f3c --iv 000102030405060708090a0b0c0d0e0f

# 从密钥文件读取（16 字节原始二进制，或 hex/Base64 文本），
# IV 改为 RFC §5.2 序号派生（播放列表不写 IV），URI 指向 CDN
python -m hls_packager input.mp4 output_dir/ -y \
  --key-file secret.key --iv-mode sequence \
  --key-uri https://cdn.example.com/keys/encryption.key
```

参数一览：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `input` | 输入媒体文件（mp4/mkv/mov/ts/m4a 等） | 必填 |
| `output_dir` | HLS 产物输出目录（自动创建） | 必填 |
| `-d/--segment-duration` | 目标分片时长（秒，支持小数），切点对齐 GOP | 6 |
| `--align-mode` | 切点对齐模式：`nearest`/`before`/`after`（纯音频无效） | nearest |
| `--segment-format` | 分片容器：`ts`/`fmp4`/`mp4`/`mkv` | ts |
| `--name-template` | 命名模板，占位符 `{name}`/`{sequence}`/`{index}`/`{ext}`，序号可零填充 | `{name}_{sequence:04d}.{ext}` |
| `-w/--window-size` | 直播滑动窗口分片数；不指定即 VOD | 无 |
| `-s/--media-sequence` | 首片 `EXT-X-MEDIA-SEQUENCE` 序号 | 0 |
| `-p/--filename-prefix` | 分片文件名前缀（模板中 `{name}` 的值） | seg |
| `-y/--overwrite` | 允许写入非空目录并覆盖同名产物 | 关 |
| `--watermark-text` | 水印文字（UTF-8，支持多字节/特殊字符）；不指定即不叠加 | 无 |
| `--watermark-font` | 显式字体文件，可重复传入按顺序回退 | 系统字体 |
| `--watermark-size` | 水印字号（像素） | 28 |
| `--watermark-color` | 文字颜色（`#RGB`/`#RRGGBB`/命名色） | white |
| `--watermark-opacity` | 不透明度 0.0~1.0 | 1.0 |
| `--watermark-anchor` | 九宫格锚点（如 `top-left`/`center`/`bottom-right`） | bottom-right |
| `--watermark-margin-x/y` | 水印距画面边缘的像素留白 | 24 / 24 |
| `--encrypt` | 启用 AES-128 加密；未给密钥时自动生成随机密钥+IV | 关 |
| `--key-file` | 密钥文件（16 字节原始二进制，或 hex/Base64 文本），隐含启用加密 | 无 |
| `--key` | 直接传密钥：32 位 hex 或 Base64（解码后 16 字节） | 无 |
| `--key-uri` | EXT-X-KEY 的 URI（相对文件名或 https 绝对地址） | encryption.key |
| `--iv` | 显式 IV：32 位 hex（可带 `0x`）或 Base64 | 随机生成 |
| `--iv-mode` | 无显式 IV 时：`random`（随机显式 IV）/ `sequence`（序号派生） | random |

退出码：`0` 成功；`2` 可预期的输入/环境/切片错误；`3` 未知内部错误。

### 文字水印的安全与回退设计

- **防注入两道防线**：① 子进程始终以 argv **列表**启动、绝不经过 shell，
  含空格/引号/分号/`$()`/反引号/中文的路径与文本作为独立元素原样传递；
  ② 水印 PNG 经第二个 `-i` + 流标签进入过滤器图，**路径永不拼进
  filtergraph 字符串**，从根上消除过滤器注入面。`escaping.py` 另提供
  filtergraph 两级转义与 POSIX/Windows 引号包裹（仅用于日志展示）。
- **字体回退链（逐码点决策）**：显式字体 → 系统 TrueType（CJK 字体优先，
  支持 .ttf/.ttc）→ 内置 5x7 点阵（仅 ASCII）→ 豆腐块占位。解析失败的
  字体被缓存跳过，绝不静默丢字。
- **GOP 承诺不破坏**：重编码时把规划切点同时喂给 `-force_key_frames`，
  并对图片输入加 `-loop 1`（否则单帧图片会经 `shortest` 截断整段视频），
  保证每片首帧仍是 IDR、时间戳全局连续、`segment_times` 精确生效。

## 作为库调用

```python
from hls_packager import package, plan_segments, render_playlist, PlaylistConfig
from hls_packager import WatermarkConfig, render_text_png

# 一键打包
result = package("input.mp4", "out/", target_duration=4.0)
for seg in result.segments:
    print(seg.sequence_number, seg.filename, seg.start_time, seg.duration)

# 自定义分片容器 + 命名模板 + 对齐模式
result = package(
    "input.mp4", "out/",
    target_duration=4.0,
    container="fmp4",                              # ts / fmp4 / mp4 / mkv
    align_mode="before",                           # 片长不超过目标值
    name_template="{name}-{sequence:05d}.{ext}",
    filename_prefix="ep",
    overwrite=True,
)

# 也可直接使用命名模板与容器规格这两个纯逻辑组件
from hls_packager import NamingConfig, get_container

spec = get_container("fmp4")
naming = NamingConfig(template="{name}_{index:03d}.{ext}",
                      name_prefix="seg", extension=spec.extension)
print(naming.render_many(count=3, sequence_start=100))
# -> ['seg_000.m4s', 'seg_001.m4s', 'seg_002.m4s']
print(naming.ffmpeg_pattern("out", sequence_start=100))
# -> ('out/seg_%03d.m4s', 0)  # index 模式 ffmpeg 计数器恒从 0 起

# 带文字水印的打包（watermark=None 时与纯流拷贝行为完全一致）
wm = WatermarkConfig(
    text="内部资料 2026",
    font_size=30,
    color="#FF8800",
    anchor="bottom-right",
)
result = package("input.mp4", "out/", target_duration=6.0, watermark=wm, overwrite=True)

# 带 AES-128 加密的打包（encryption=None 时与旧版本行为完全一致）
from hls_packager import EncryptionConfig

enc = EncryptionConfig.generate()                 # 随机密钥 + 随机显式 IV
# enc = EncryptionConfig.from_key_file("secret.key", iv=None)  # 序号派生 IV
result = package("input.mp4", "out/", target_duration=6.0,
                 encryption=enc, overwrite=True)
print(result.extra["encryption"])  # key_file / key_uri / key_hex / iv_mode / iv_hex

# 也可单独使用纯函数模块
planned = plan_segments(duration=30.0, target_duration=4.0,
                        keyframe_times=(0.0, 2.0, 4.0, ...))
text = render_playlist(planned, PlaylistConfig(target_duration=4))

# 只做文字栅格化（自行产出 RGBA PNG，不跑 ffmpeg）
rendered = render_text_png(WatermarkConfig(text="水印 ABC"), "watermark.png")
print(rendered.width, rendered.height, rendered.used_fallbacks)
```

## 产物说明

```
output_dir/
├── playlist.m3u8     # 播放列表（加密时含 EXT-X-KEY:METHOD=AES-128,URI=...,IV=...）
├── encryption.key    # AES-128 裸密钥（16 字节；仅启用加密时生成）
├── seg_0000.ts       # MPEG-TS 分片（188 字节定长包；加密后为 16 字节对齐密文）
├── seg_0001.ts
├── ...
└── segments.json     # 机器可读元数据（源内起点 / 时间轴起点 / 实测时长 / 加密信息）
```

## 支持的编码

流拷贝不重编码，各容器的可承载编码白名单（定义在 `containers.py`）：

- **MPEG-TS（ts）**：视频 H.264/AVC、HEVC、MPEG-1/2；音频 AAC、MP3、
  AC-3、E-AC-3、MP2、Opus；
- **ISO-BMFF（fmp4 / mp4）**：视频 H.264、HEVC、MPEG-4 Part 2；
  音频 AAC、MP3、AC-3、E-AC-3、Opus；
- **Matroska（mkv）**：在上述基础上另支持 AV1 视频与 FLAC、Vorbis、
  PCM 音频，承载面最宽。

主视频编码不被目标容器支持时直接报错（避免静默产出只有声音的
"视频"）；个别不支持的音频轨道会跳过。字幕 / 数据流不打入分片。

## 运行测试

```bash
pytest
```

- `tests/test_planner.py`、`test_m3u8.py`、`test_probe.py`、
  `test_ts_writer.py`、`test_naming.py`、`test_containers.py`：
  纯逻辑单测，不依赖 ffmpeg——覆盖 GOP 三种对齐模式、命名模板渲染
  与安全校验、容器规格/编码白名单/命令构造；
- `tests/test_encryption_aes.py`、`test_encryption_config.py`、
  `test_encryption_segment_io.py`：加密纯逻辑单测——FIPS-197 / SP 800-38A
  官方向量、PKCS#7 畸形填充、密钥/IV/URI 容错、流式分块等价、原子替换
  失败回滚；装有 openssl 时自动追加字节级交叉验证；
- `tests/test_encryption_integration.py`：真实 ffmpeg 切片后加密，
  openssl 解密回明文 TS 并由 ffprobe 复测时长、EXT-X-KEY 合规、
  序号 IV、滑动窗口/非零 MEDIA-SEQUENCE 组合、CLI 退出码与旧路径回归；
- `tests/test_watermark_*.py`：水印单测（UTF-8 异常输入、字体回退、
  转义注入、PNG 渲染）+ ffmpeg 集成测试（首帧 IDR、水印可见、
  特殊字符路径、时间戳连续、CLI 端到端）；
- `tests/test_integration.py`、`test_cli.py`、
  `test_containers_integration.py`：用 ffmpeg 合成
  H.264/HEVC、纯音频、大 GOP 等素材做端到端校验，覆盖首帧 IDR、
  独立解码、GOP 对齐、全局时间戳连续、concat 单调、M3U8 合规、
  滑动窗口，以及四种容器落盘签名、模板命名与对齐模式。无 ffmpeg
  时自动 skip。

## 已知限制

- 不带水印时只做**重封装**，不转码；源编码无法承载于目标容器时会
  明确报错，不会偷偷转码。
- **带水印必然重编码视频轨**（像素级叠加无法流拷贝），仅支持 H.264/HEVC/
  MPEG-1/2 视频；音频仍流拷贝。纯音频输入不能叠加文字水印，会明确报错。
- 自研 TrueType 解析器只支持 TrueType 轮廓与 cmap format 0/4/12；
  OpenType/CFF（OTTO）、WOFF 会被跳过并走字体回退链，不影响主流程。
- 单码率输出，不生成多码率 master playlist。
- 直播模式产出的是"当前窗口快照"式 VOD 风格文件，不包含分片滚动
  清理与周期刷新守护进程；`MEDIA-SEQUENCE` 语义已就绪，可作为
  实时打包循环的构建块。
- 切点数量极多导致命令行超长时会明确报错并提示增大分片时长
  （Windows 命令行长度上限保护）。
- RFC 8216 的 `METHOD=AES-128` 只提供机密性、不提供完整性校验（CBC 无
  MAC）；本工具的去填充校验能发现意外损坏，但不能防御主动篡改，这是
  协议方法本身的属性。需要完整性应升级到 `SAMPLE-AES` 或配合 HTTPS
  分发与外部签名机制。
- 加密密钥以 16 字节裸文件写入输出目录，生产环境应通过 `--key-uri`
  指向独立的鉴权密钥服务，不要把密钥与播放列表放在同一公开路径下。
