# ABR 阶梯生成器 (ABR Ladder Generator)

输入一个源片，输出 **1080p / 720p / 480p / 360p** 四档 H.264 编码文件与质量报告：

- 各档**实测视频码率**在标称值 **±3%** 内（闭环校正，通常 1–2 轮 pass-2 即收敛）；
- **帧号级关键帧对齐**：四档共享同一套关键帧帧号，切档点全部是闭合 GOP 的 IDR；
- **切档不花屏**：独立校验脚本从切档关键帧开始解码并与源片对比，确认首帧完整、可解码；
- **VMAF + SSIM**：每档给出原生分辨率与「放大到最高档同屏对比」两个视角的评分、最差 2 秒窗口、最差场景；
- **选档依据 + 相邻档质量差**：`report.json` / `report.md` 中直接给出，含「肉眼能否分辨」的经验判断；
- **10 分钟素材、8 线程预算，整条阶梯 < 25 分钟**（8 逻辑核 affinity 实测 15.2 分钟，见[性能基准](#性能基准)）。

---

## 1. 快速开始

### 依赖

| 依赖 | 要求 |
|---|---|
| ffmpeg / ffprobe | 需启用 `libx264`；VMAF 模式需启用 `libvmaf`（本仓库在 ffmpeg 9.0.2 full build 上验证） |
| Python | 3.8+，**无第三方依赖**（仅标准库） |
| 平台 | Windows / Linux / macOS |

```bash
ffmpeg -hide_banner -encoders | grep libx264
ffmpeg -hide_banner -filters  | grep libvmaf
python --version
```

如果 ffmpeg 不在 `PATH`，可设置 `FFMPEG_BIN` / `FFPROBE_BIN` 指向可执行文件。

### 直接运行（无需安装）

Windows CMD / PowerShell：

```bat
abr-ladder.cmd input.mp4 output_dir/ --preset fast
abr-ladder-verify.cmd output_dir
```

Linux / macOS / Git Bash：

```bash
./abr-ladder input.mp4 output_dir/ --preset fast
./abr-ladder-verify output_dir
```

或直接用 Python 模块：

```bash
python abr-ladder.py input.mp4 output_dir/
python abr-ladder-verify.py output_dir
```

### 通过 pip 安装（可选）

```bash
pip install .
abr-ladder input.mp4 output_dir/
abr-ladder-verify output_dir
```

### 最小示例

```bash
abr-ladder source.mp4 out/source_ladder \
  --preset fast \
  --threads 8 \
  --vmaf-model vmaf_v0.6.1.json   # 可省略，默认用 libvmaf 内置模型
```

默认参数即满足题目规格：四档 5M / 2.5M / 1M / 600k、2 秒 GOP、VMAF 评估、±3% 码率容差。

---

## 2. 输出目录

```
output_dir/
├── 1080p.mp4              # 四档编码文件（统一音频、帧号级对齐的关键帧）
├── 720p.mp4
├── 480p.mp4
├── 360p.mp4
├── report.json            # 机器可读：码率、VMAF/SSIM、关键帧、切档校验
├── report.md              # 人类可读：选档依据、相邻档质量差、性能
├── logs/                  # 每档编码/质量/校验的 ffmpeg 命令与日志、run.log
├── metrics/               # VMAF JSON 与 SSIM 统计文件（审计用）
└── .work/                 # 两遍编码统计、临时音频；默认成功后删除
```

`report.json` 的关键字段：

```jsonc
{
  "rungs": [{
    "name": "1080p", "width": 1920, "height": 1080, "nominal_kbps": 5000,
    "measured_kbps": 4925.47, "deviation_pct": -1.491, "mode": "2pass-vbr",
    "attempts": [...],
    "quality": {
      "metric": "vmaf", "mean": 96.301, "min": 82.937, "p1": ...,
      "worst_windows": [{"start": 128, "end": 130, "mean": 87.312}],
      "worst_scene": {...}, "scenes": [...],
      "display": { "target_width": 1920, "mean": 96.301, "min": 82.937, ... },
      "ssim_mean": 0.98698
    },
    "rationale": {"summary": "...", "bpp": 0.079, "bpp_band": "均衡", ...}
  }],
  "keyframes": {"frame_indices": [0, 60, 120, ...], "max_gap_frames": 60, ...},
  "adjacent_gaps": [{"from": "1080p", "to": "720p",
                     "native_delta": -0.6, "display_delta": 6.8,
                     "bandwidth_saving_pct": 50.0,
                     "visible_judgement": "可察觉", ...}],
  "verification": {"passed": true, "keyframe_alignment": {...},
                   "switch_tests": [...], "audio": {...}},
  "performance": {"timings": {...}, "within_time_budget": true}
}
```

---

## 3. 设计说明

### 3.1 码率控制：为什么能做到 ±3%

**策略：2-pass capped VBR + 闭环校正 + CBR filler 兜底。**

1. **一遍分析、两遍编码**。pass 1 只写统计文件（`-pass 1`），pass 2 按目标码率编码。
   pass 1 的统计与目标码率无关，所以后续调整目标时**只需重跑 pass 2**，代价很低。
2. **实测而不是相信编码器**。码率从 MP4 中 `v:0` 的 packet 字节总数 / 视频时长算出
   （不含音频/容器开销），与标称值比较。
3. **闭环校正**。若实测偏离目标，按 `new_target = target × nominal / measured`
   调整 pass 2 码率，最多 3 次。正常内容通常 1 次就落进 ±3%；验收阈值留了 2.8%
   的安全边界。
4. **CBR + filler 兜底**。静态/极简画面即使把目标码率调高，x264 也会因质量上限
   吃不满码率（表现为「目标涨了、实测不动」）。工具会检测这种平台期（或首轮低于
   目标 8% 以上），切换到
   `nal-hrd=cbr + filler=1`，用填充 NAL 精确补足到标称值。代价是简单画面有少量
   无效填充，但这是满足「实测 ±3%」硬约束的必要手段；正常内容不会走到这条路径。
5. **流媒体友好**。VBR 默认带 `maxrate=1.5×`、`bufsize=3×` 的 VBV 约束，避免单档
   出现无限峰值；码率误差仍由闭环负责。

选择 capped VBR 而不是全程 CBR 的原因：CBR 会在简单场景浪费大量码率、复杂场景
画质骤降；VBR + 闭环在大多数内容上既准确又画质更好，只有极简内容才退回 filler。

### 3.2 关键帧对齐：帧号级，而不是时间戳级

- **统一帧率**：所有档位按源的 CFR 帧率编码（VFR 源会转为 CFR），使用同一时间基
  （MP4 timescale 90000）。
- **一次性生成关键帧计划**：在 320×180 分析流上检测场景切点（`select=gt(scene,0.35)`），
  与固定 2 秒网格合并。距网格 0.15 秒内的切点会被网格覆盖；相邻切点最小间隔 0.5 秒。
- **用帧号强制关键帧**：`-force_key_frames "expr:not(mod(n,60))+eq(n,150)+..."`。
  `n` 是帧号，因此计划在每一档完全一致，不会因浮点时间戳舍入而漂移。
- **关闭额外关键帧**：`keyint=9999:min-keyint=1:scenecut=0:open-gop=0`。
  x264 不会自行插入关键帧，关键帧集合严格等于计划；`open-gop=0` 保证强制关键帧
  是闭合 GOP 的 IDR。
- **其余结构参数四档一致**：preset、profile、B 帧数、参考帧数、像素格式相同，
  只有分辨率与码率不同，播放器切档时只需在 IDR 处重配 SPS/PPS。
- **编码后校验**：校验脚本用 `-skip_frame nokey` 快速扫描每档关键帧帧号，
  要求四档集合完全相等，且等于计划集合。

### 3.3 切档不花屏：怎么证明

关键在两点：切档点必须是 IDR、切过去第一帧必须完整可解码。工具在生产和校验阶段都验证：

1. **对齐校验**：四档关键帧帧号完全一致（上面 3.2）。
2. **切档解码测试**：对每对相邻档，在 1/3、2/3 处的关键帧开始，从目标档解码 8 帧，
   使用 `-xerror`（任何解码错误直接失败，杜绝「绿屏/花屏但进程不报错」）。
3. **边界画质测试**：同一时间点，把目标档与源片缩放参考做 SSIM。如果 IDR 损坏或
   帧错位，SSIM 会显著偏低；正常切档应 > 0.99（测试内容）。
4. **音频一致性**：四档复用同一份音频轨道，校验脚本对每档音频 packet 做 MD5，
   要求完全一致，切档时不会出现音频跳变。

> 说明：本工具输出的是四个对齐 MP4，不负责 HLS 切片；打包器按关键帧切分后，
> 上述 IDR 对齐与音频一致性即可保证播放器切档连续。

### 3.4 VMAF / SSIM 质量评估方法论

每个档位计算两个视角：

| 视角 | 做法 | 用途 |
|---|---|---|
| **原生** | 编码档 vs 源片缩放到**该档分辨率**（bicubic） | 该档在自己分辨率下的编码质量 |
| **同屏** | 编码档**放大到最高档分辨率** vs 源片（最高档分辨率） | 包含分辨率损失的实际观感；大屏/同屏对比更接近肉眼 |

只用原生 VMAF 会出现「360p 分数反而比 1080p 高」的现象——降分辨率把细节藏起来了。
因此：

- **相邻档质量差与「肉眼能否分辨」使用同屏指标**；
- 原生指标用于排查编码问题（例如某档在同分辨率下明显偏低）。

VMAF 用 libvmaf 内置 `vmaf_v0.6.1`（可通过 `--vmaf-model` 替换模型 JSON）；SSIM 在
同一次解码中计算，几乎零额外开销。10 分钟素材默认每 5 帧取 1 帧算 VMAF
（`--vmaf-subsample 0` 为自动；短于 60 秒的素材默认全帧），最差窗口按 2 秒聚合。
容量与时间允许时可设 `--vmaf-subsample 1` 做全帧评估。

相邻档「肉眼」判断阈值（SDR 经验规则，报告中标明为经验判断）：

| 同屏 VMAF 差 | 判断 |
|---|---|
| < 3 | 几乎无法分辨 |
| 3–6 | 仔细对比可分辨 |
| 6–12 | 可察觉 |
| > 12 | 明显可察觉（阶梯断层） |

### 3.5 选档依据

每档的 `rationale` 由以下信息组成：

- **像素码率密度** `bpp = 码率 / (宽 × 高 × 帧率)`：题目四档的标称值都约
  0.08 bit/pixel/frame，压缩压力一致；
- **原生/同屏 VMAF 与 SSIM** 及等级（极佳 / 优秀 / 良好 / 一般 / 较差）；
- **最差 2 秒窗口与最差场景**；
- **相对相邻档的带宽节省与画质差**；
- 码控模式与闭环尝试次数，便于解释该档是否用了 filler 兜底。

报告还会标记潜在问题：

- 相邻档同屏质量差 < 1.5 分但码率差很大 → **冗余档嫌疑**；
- 同屏质量差 > 12 分 → **阶梯断层**；
- 最差窗口明显低于均值 → 指出具体时间段，便于回看。

### 3.6 性能设计（8 线程 / 10 分钟素材 / 25 分钟）

- **共享线程预算调度器**：默认预算 8 线程。四档编码按像素数分配
  `1080p=4, 720p=2, 480p=1, 360p=1`，同时启动；某档编码完成后，它的质量评估
  立即在空闲槽位启动，与尚未完成的 1080p 编码重叠，而不是串行两阶段。
- **pass-1 统计复用**：码率校正只重跑 pass 2。
- **场景检测降采样**：320×180 分析流，约 50× 实时；10 分钟素材约 25 秒。
- **VMAF 子采样**：10 分钟素材默认 1/5 帧。
- **音频只准备一次**：AAC 直接 copy，非 AAC 转一次 AAC，四档复用同一文件。
- **关键帧扫描用 `-skip_frame nokey`**，避免解码整条视频。
- **平台期早停**：编码器无法花掉更多码率时立即切换 CBR 兜底，不做无效的重编码。

---

## 4. 性能基准

> 说明：本机为 i5-13600K（14 核 20 逻辑线程），**运行时把编码/质量线程预算固定为 8**，
> 模拟题目的 8 核机器。ffmpeg 9.0.2 full build，preset `fast`。

测试源由 `tools/make_test_source.py` 生成：1080p30、600 秒、含 7 段不同画面拼接
（真实场景切点）、轻微颗粒噪声、AAC 128k 音频；可用如下命令复现：

```bash
python tools/make_test_source.py bench10m.mp4 --minutes 10
```

运行命令：

```bash
abr-ladder bench10m.mp4 out10m --preset fast --threads 8
```

<!-- BENCH_START -->
**测试结果（进程树通过 CPU affinity 限制到 8 个逻辑 CPU，严格模拟 8 核机器）**

| 阶段 | 结果 |
|---|---|
| 源片探测 + 场景检测/关键帧计划 | 25.6s（352 个关键帧，51 个场景切点） |
| 四档编码（并行，1080p=4 / 720p=2 / 480p=1 / 360p=1 线程） | 累计 1829s；墙钟约 530s（1080p 最长） |
| 质量评估（原生 + 同屏 VMAF/SSIM，四档并行） | 累计 1256s；与编码重叠执行 |
| 码率 / 关键帧对齐 / 切档校验 | 9.1s |
| **总墙钟** | **914.6s ≈ 15.2 分钟**（预算 25 分钟，余量约 39%） |

| 档位 | 实测码率 | 偏差 | 原生 VMAF | 同屏 VMAF | 编码耗时 |
|---|---|---|---|---|---|
| 1080p | 4925.4 kb/s | -1.49% | 96.30 | 96.30 | 529.8s |
| 720p | 2461.9 kb/s | -1.52% | 96.91 | 89.44 | 394.2s |
| 480p | 985.6 kb/s | -1.44% | 96.99 | 81.99 | 516.8s |
| 360p | 591.9 kb/s | -1.35% | 97.26 | 75.39 | 388.5s |

- 相邻档同屏 VMAF 差：1080p→720p **6.86**、720p→480p **7.46**、
  480p→360p **6.60**，均落在「可察觉但可接受」区间；
- 352 个关键帧四档帧号完全一致，切档边界解码 SSIM 最低 **0.9986**；
- 音频 packet MD5 四档一致；校验脚本整体 **PASS**。

作为对照，同一台机器不限制 affinity（20 逻辑线程可用、内部仍按 8 线程预算调度）
时总墙钟约 **9.1 分钟**。8 逻辑核下的数据为本次 25 分钟指标的验收口径。
<!-- BENCH_END -->

**结论：** 整条阶梯（4 档编码 + 原生/同屏 VMAF + SSIM + 码率/对齐/切档校验）在
8 逻辑核限制下 **15.2 分钟**完成，低于 25 分钟预算；编码阶段由 1080p 决定（4 线程），
质量阶段与编码重叠，1080p 的 VMAF 为长尾但仍在预算内。

---

## 5. 校验脚本

```bash
abr-ladder-verify output_dir
# 或
python abr-ladder-verify.py output_dir
```

检查项：

1. 每档视频流实测码率与 `report.json` 标称值的偏差 ≤ 3%；
2. 分辨率、时长、帧数一致；
3. 四档关键帧帧号集合完全一致，且等于 `report.json` 中的计划；
4. 每对相邻档在采样切点的切档解码（`-xerror`）与边界 SSIM（阈值 0.50）；
5. 音频 packet MD5 四档一致；
6. 退出码：`0` 全部通过，`1` 校验失败，`2` 输入/环境错误。

```bash
abr-ladder-verify output_dir --json       # 输出完整 JSON
abr-ladder-verify output_dir --tolerance 3 --sample-points 3
```

生产流程结束时也会自动跑同一套校验，并把结果写入 `report.json` 的
`verification` 字段；校验失败时生成器退出码为 `2`，不会静默成功。

---

## 6. 测试

```bash
python -m unittest tests.test_unit tests.test_e2e -v
```

- `test_unit.py`：分辨率适配、关键帧计划（场景切点/去重/最大 GOP）、线程分配、
  质量差判断阈值、bpp 计算。
- `test_e2e.py`：生成 8 秒双场景 + PCM 音频源片，跑完整管线，校验 4 档文件、
  ±3% 码率、关键帧计划、音频一致、报告字段和 Markdown 结构。

---

## 7. 设计取舍与局限

- **编码器**：默认 libx264（H.264 High profile，8-bit yuv420p）。题目允许 x264/x265；
  选 x264 是为了兼容性与 8 核下的速度。x265 需要额外评估档位对应的 CRF/码率曲线。
- **码率口径**：按视频 packet 计，不含音频与容器。若业务按文件总大小计费，需要把
  音频码率（128–192k）加到对应档；`report.json` 中保留了文件大小。
- **HDR / 10-bit**：源为 10-bit 时会转成 8-bit，不保留 HDR 元数据；不适用于 HDR 交付。
- **HLS**：不生成切片/播放列表，只输出对齐的 MP4；后续打包器可直接使用。
- **音频**：只取第一条音频；AAC 原样 copy，其他编码转 AAC。不做多语言音频组。
- **VMAF 分辨率**：同屏视角固定使用最高档作为显示分辨率；若源为 4K 而最高档是
  1080p，4K 原生细节的损失不在评分中体现。
- **filler**：极简内容为了满足 ±3% 会使用 filler NAL，牺牲少量带宽；报告会明确标出
  `cbr-filler` 模式。
- **VMAF 采样**：默认 1/5 帧用于长素材，均值/最差窗口是采样估计；需要严格评估时用
  `--vmaf-subsample 1`（更慢）。
- **非 16:9 源**：按目标尺寸做等比缩放 + 黑边填充（`pad`），不裁切。
- **场景切点**：阈值 0.35 针对硬切；渐变转场可能不加关键帧，但 2 秒网格仍保证最大 GOP。

---

## 8. 目录结构

```
abr-ladder-generator/
├── abr-ladder.py / abr-ladder.cmd / abr-ladder       # 生成器入口
├── abr-ladder-verify.py / .cmd / abr-ladder-verify   # 校验脚本入口
├── abr_ladder/
│   ├── cli.py          # 参数与退出码
│   ├── pipeline.py     # 调度器、并行编码/质量、报告汇总
│   ├── config.py       # 档位规格、设置、宽高比适配
│   ├── ff.py           # ffmpeg/ffprobe 封装、码率/关键帧测量
│   ├── keyframes.py    # 场景检测 + 帧号级关键帧计划
│   ├── encode.py       # 两遍编码、闭环码率、CBR filler、音频/封装
│   ├── quality.py      # VMAF/SSIM 原生 + 同屏评估
│   ├── verify.py       # 独立校验
│   └── report.py       # report.json / report.md
├── tests/              # 单元 + 端到端测试
├── tools/make_test_source.py
├── pyproject.toml
└── README.md
```
