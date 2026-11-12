# VFR 转码器 (VFR → CFR Transcoder)

专门处理老素材的转码器：源是 **VFR（可变帧率）、时间基 `1/1000`、带 B 帧重排**，
输出 **CFR（恒定帧率）**，保证音画偏移 **≤ 40 ms（首、中、尾都测）**，保留原始
时间戳语义（copyts），并提供 HTTP API + worker 进程池、任务状态机和 **kill 后断点
续跑**。

---

## 1. 快速开始

```bash
pip install -r requirements.txt          # fastapi / uvicorn / pydantic / numpy / httpx
# 需要系统安装 ffmpeg / ffprobe（开发与验证基于 ffmpeg 9.x）

# 命令行直接转码
python -m vfr_transcoder --work-root ./work transcode input.mkv out.mp4 --target-fps 25

# HTTP API 服务（只负责任务状态，不跑转码）
python -m vfr_transcoder serve --host 0.0.0.0 --port 8080

# Worker 进程（可起多个，横向扩展，共享同一个 work 目录）
python -m vfr_transcoder worker
python -m vfr_transcoder worker --once   # 处理一个任务后退出
```

### 生成测试素材 & 校验

```bash
# 自造 VFR 长片（1/1000 时间基、毫秒级抖动、B 帧、非零起始 PTS、音画标记）
python tools/make_vfr_sample.py --duration 1800 samples/vfr_30min.mkv   # 30 分钟
python tools/make_vfr_sample.py --duration 10800 samples/vfr_3h.mp4     # 3 小时

# 音画同步 / CFR 网格 / copyts / DTS 全量校验（CI 门禁，失败返回非 0）
python tools/verify_sync.py out.mp4 --source src.mkv

# 一键长片门禁（拒绝 < 30 分钟的样本）
python tools/run_long_gate.py --work-dir ./work/long --duration 1800
python tools/run_long_gate.py --work-dir ./work/long3h --duration 10800
```

---

## 2. 目录结构

```
vfr_transcoder/
  config.py        全部可调参数（环境变量可覆盖）
  ffmpeg_utils.py  ffmpeg/ffprobe 子进程封装、-progress 解析
  media.py         源分析：时间基、主流帧率检测、VFR/B 帧判定
  pipeline.py      分段、可续跑的 VFR→CFR 转码流水线
  state.py         SQLite 任务状态机、租约、分段进度
  worker.py        worker 轮询/认领/心跳/崩溃恢复
  api.py           FastAPI HTTP 接口
  cli.py           transcode / serve / worker 三个命令
tools/
  make_vfr_sample.py  自造 VFR 长片（含音画同步标记）
  verify_sync.py      逐帧结构校验 + 内容级音画漂移校验
  run_long_gate.py    30 分钟 / 3 小时长片门禁
tests/                unittest 套件（含 kill -9 续跑、API+worker 集成）
```

---

## 3. VFR → CFR 算法

### 3.1 为什么不能简单丢帧/补帧

VFR 源的相邻帧间隔是变化的（本项目测试素材在 25 fps 名义值上以 30 ms / 50 ms
交替抖动）。朴素做法按固定间隔"抽帧"或"复制帧"会在时间轴上引入与帧内容无关的
位移，并随片长累积。

### 3.2 采用 ffmpeg `fps` 滤镜做时间戳重映射

每个视频分段用：

```
ffmpeg -ss <seg_start> -t <seg_len> -i src -vf fps=<target> -fps_mode cfr ... out.ts
```

`fps` 滤镜在理想 CFR 时间网格上选最近邻的源帧来填充输出帧。关键性质：

* **输出帧的 PTS 由理想网格直接给出**，即 `pts[i] = origin + i / fps`，不依赖前面
  任何一帧的舍入结果；
* 因此第 i 帧的时间戳误差是一个**有界常量**（本实现实测最大网格误差 0.0 ms，
  30 fps 在 1/1000 时间基下最坏 ±1 ms），**不存在逐帧累积**——这正是 3 小时素材
  片尾仍和片头处于同一误差预算内的根本原因；
* 滤镜只改呈现时间、**从不重排帧序**，B 帧的解码/呈现顺序关系保持不变。

### 3.3 主流帧率检测

VFR 文件的 `r_frame_rate` 只报告名义帧率，不可直接信赖。`media.py` 读取**全部视频
包**，按 PTS（呈现顺序，而非受 B 帧重排影响的文件顺序）排序后计算间隔，用**均值**
（对称抖动零均值，中位数会偏向交替间隔中较多的一侧）估计帧率，再吸附到 12/15/
23.976/24/25/29.97/30/… 等常见帧率；同时参考容器 `r_frame_rate` 兜底。

### 3.4 两个实测得到的 ffmpeg 9.x 关键坑

* **保留 VFR 必须 `-enc_time_base 1/1000 -fps_mode vfr`**：编码器时间基默认取帧率
  （如 1/25），毫秒级 PTS 会在进编码器前被量化回粗网格，VFR 被静默抹平。生成测试
  素材时这两个参数缺一不可。
* **`setpts` 的单位是输入时间基 tick 而非秒**，且 `settb` 必须放在 `setpts` 之前；
  filtergraph 里函数参数的逗号要转义（`mod(n\,2)`），drawbox 的 `enable` 用小写 `n`
  （大写 `N` 只在 setpts 中有效）。

---

## 4. 音画同步保障（40 ms 误差预算）

误差被拆成三个互不累积的来源，全部远小于 40 ms：

| 来源 | 量级 | 是否累积 |
|---|---|---|
| 视频帧到 CFR 网格的量化 | 25 fps 为 0；最坏帧率/时间基下 ≤ 1 ms | 否（每帧独立） |
| 音频包粒度（AAC 1024 样本/包，48 kHz ≈ 21.3 ms） | 最近邻 ≤ 11 ms | 否 |
| 分段边界 | 0（见下） | 否 |

设计要点：

1. **音频整体抽取、绝不切段**（`asetpts=N/SR/TB` 归一到 0 后一次编码）。切段会在每个
   边界引入编码器 priming 间隙，是长片片尾偏移的典型来源；单段音频只有一个 priming 点。
2. **分段为 MPEG-TS**：每段首帧是关键帧，concat demuxer 跨文件**延续而非重置**时间戳，
   段间 DTS 严格递增，边界处帧间隔仍是精确的 `1/fps`。
3. **最终只做一次 mux**：原型验证发现多一次中间 remux 会让开头 B 帧的 DTS 出现重复；
   直接从 concat 后的 TS + 完整音频一次复用可避免。
4. 校验在**首、中、尾**三点分别测量包级音画偏移，并对内容标记测量"片尾相对片头的
   gap 漂移"，直接针对题目描述的"片尾偏移 200 ms"故障模式。

---

## 5. copyts（保留原始时间戳）

语义：输出时间轴的**原点**与源一致，方便监控系统按原始时间戳关联。

实现：流水线内部所有分段和音频都先归一化到从 0 开始（便于独立分段与续跑），在分析
阶段记录源的最早呈现时间 `start_origin`（视频与音频起点取最小），最终 mux 时通过

```
ffmpeg -i video.ts -i audio.m4a -c copy -output_ts_offset <origin> out.<ext>
```

一次性把两条流的时间轴平移回源原点。校验项 `copyts origin preserved` 逐次比对源与
输出的 `start_time`（实测偏差 0.0 ms）。

---

## 6. 断点续跑设计

### 6.1 持久化单元

* **视频分段**：固定时长（默认 60 s）的 MPEG-TS。每段先写 `*.tmp<pid>` 再原子
  `os.replace` 成正式文件，随后在 SQLite `segments` 表落库并提交。kill 发生在落库前
  → 下次重编码该段（临时文件不会被误认为成品）；落库后 → 永久跳过。
* **音频**：整文件抽取，同样临时名 + 原子替换 + `audio_done` 标志。
* **最终产物**：写 `output.tmp<pid>`，成功后原子改名，kill 在 mux 期间不会留下半成品。
* 所有状态在同一个 SQLite（WAL 模式、`busy_timeout`）中，多 worker 共享。

### 6.2 租约与崩溃恢复

worker 认领养活一个带到期时间的租约，并在 ffmpeg 每帧 `-progress` 回调中刷新心跳。
进程被 `kill -9`（Windows 下为 `taskkill /F`）后心跳停止、租约过期；新 worker 启动时
`requeue_expired()` 把过期的 running 任务退回 queued 并重新认领，**已完成的分段全部
保留**。集成测试 `test_resume_kill.py` 真实地起独立进程、硬杀、再换新进程续跑验证。

---

## 7. 任务状态机

```
pending ──► queued ──► running ──► completed
               ▲          │
               │          ├──► failed ──► queued   （手动/接口重试）
               └──────────┘   (过期租约)
completed ──► queued                          （手动重跑）
```

非法迁移由 `IllegalTransition` 拒绝。HTTP 层：

| 方法 | 路径 | 作用 |
|---|---|---|
| POST | `/jobs` | 提交任务（body: source, output, target_fps?） |
| GET  | `/jobs?limit=` | 列表 |
| GET  | `/jobs/{id}` | 详情（含分段进度、origin_pts、错误信息） |
| POST | `/jobs/{id}/retry` | 重试，`{"reset_progress": true}` 可清分段从头来 |
| GET  | `/health` | 健康检查 |

---

## 8. 测试与长片验证

```bash
python -m unittest discover -s tests -v
```

| 测试文件 | 覆盖 |
|---|---|
| `test_state_machine.py` | 状态迁移、非法迁移拒绝、租约回收、分段落库、跨进程重开 DB |
| `test_media_analysis.py` | 真实 VFR 源的 VFR/B 帧/1-1000 时间基/25&30 fps 检测 |
| `test_pipeline_e2e.py` | CLI 端到端、CFR 网格误差、首尾无漂移、copyts、DTS 严格递增 |
| `test_resume_kill.py` | **独立进程硬 kill 后续跑**、分段保留、二次续跑幂等 |
| `test_api_worker.py` | HTTP 全流程、失败重试、双 worker 不重复认领 |

**长片门禁**（`run_long_gate.py`，硬性拒绝 < 1800 s 的输入）：30 分钟样本逐帧校验
CFR 网格、DTS、copyts、首中尾包级偏移，并对周期性音画标记测量内容级漂移；3 小时
资格验证把 `--duration` 换成 10800 即可。报告 JSON 落在工作目录的 `gate_report.json`。

### 实测结果（本机 ffmpeg 9.0.2，25 fps，1/1000 时间基）

| 指标 | 30 分钟（45 001 帧） | 3 小时（270 001 帧） |
|---|---|---|
| CFR 网格最大误差 | 0.000 ms（首=尾） | 0.000 ms（首=尾） |
| 累积漂移（尾−首网格误差） | 0.000 ms | 0.000 ms |
| 视频 DTS | 44 999 个显式值严格递增 | 269 999 个显式值严格递增 |
| copyts 原点偏差 | 0.0 ms | 0.0 ms |
| 包级音画偏移 首/中/尾 | 0 / 11 / 3 ms | 0 / 0 / 3 ms |
| 内容标记（红框/蜂鸣） | 6/6 保留，漂移 0.0 ms | 18/18 保留，漂移 0.0 ms |

全部远优于 40 ms 预算，且 3 小时与 30 分钟误差相同——印证了"误差为每帧独立的有界
常量、不随片长累积"的设计。

### 校验器测什么

* **结构层（逐帧 ffprobe）**：CFR 网格最大误差、首尾误差差（累积漂移）、DTS 严格递增、
  copyts 原点、首中尾包级音画偏移（尾点回退一个音频包间隔，避开容器末端填充）。
* **内容层（真实解码）**：素材在帧号门控的红框与挂钟门控的 1 kHz 蜂鸣上打了同步标记；
  校验器逐帧解码（不做会丢帧的 fps 重采样）定位标记，测量输出内部"视频-音频标记
  gap"相对片头的漂移。直接比较 VFR 与 CFR 的 PTS 没有意义（VFR 帧 PTS 本就偏离理想
  内容时间，规整它正是转码的职责），因此度量的是**输出内部随时间的漂移**，恒定原点
  偏移在差值中自然抵消。

---

## 9. 配置（环境变量）

`VFR_WORK_ROOT`、`VFR_SEGMENT_SECONDS`、`VFR_VCODEC`、`VFR_ACODEC`、`VFR_PRESET`、
`VFR_CRF`、`VFR_LEASE_SECONDS`、`VFR_HEARTBEAT_SECONDS`、`VFR_POLL_SECONDS`、
`VFR_FFMPEG`、`VFR_FFPROBE` 等，见 `config.py`。
