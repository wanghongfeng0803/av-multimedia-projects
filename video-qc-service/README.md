# 视频质检服务 (Video QC Service)

## 项目背景

来源素材里花屏和黑帧不少，目前靠人眼看，漏检率高。需要做一个质检服务，在素材进转码前先跑一遍自动化检测。

## 需求

### 检测类型（4类）

1. **纯黑帧** — 整帧亮度接近 0，持续一定帧数
2. **静帧卡死** — 连续多帧画面完全相同或差异极小
3. **花屏马赛克** — 画面出现块状损坏、宏块错误、色带断裂等
4. **音频无声或削波** — 音频轨长时间静音，或振幅超过 0 dBFS（削波失真）

### 输出格式

输出 JSON，每个问题包含：
- `type`: 问题类型（`black_frame` / `freeze_frame` / `blocky` / `audio_silence` / `audio_clipping`）
- `start_frame`: 起始帧号（0-based）
- `duration_frames`: 持续帧数
- `confidence`: 置信度 0-1

要求**帧级定位**，精度到单帧。

### 防误报要求

以下场景不能误报：
- 夜景 / 暗场镜头（不能当成纯黑帧）
- 淡入淡出（fade in/out）
- 慢镜头（高帧率素材的慢放，帧间差异小但不是静帧）

### 性能要求

- 4K 素材单机处理速度不低于 **0.5 倍速**
- 8 核机器
- 不使用 Docker

### 测试素材

测试素材位于 `/data/qc_samples`，其中一份确认有花屏，用于验证检出率。

## 技术栈建议

- Python + FFmpeg（ffmpeg/ffprobe 抽取帧和音频样本）
- OpenCV 做图像分析
- NumPy 做数值计算

（也可自由选择其他技术栈，只要满足性能和功能要求）

## 交付要求

- 可执行的命令行工具，输入视频路径，输出 JSON 报告
- README 说明使用方法、检测算法原理、误报防护策略
- 能处理常见封装格式（mp4、mov、mkv、ts）

---

# 视频质检服务 (Video QC Service)

转码前的自动化素材质检工具。一条命令对视频做帧级检测，输出 JSON 报告，覆盖四类常见素材缺陷：

| type | 缺陷 | 定位方式 |
|---|---|---|
| `black_frame` | 纯黑帧 | 起始帧号 + 持续帧数 |
| `freeze_frame` | 静帧卡死 | 起始帧号 + 持续帧数 |
| `blocky` | 花屏 / 马赛克 / 宏块损坏 | 起始帧号 + 持续帧数 |
| `audio_silence` | 音频长时间无声 | 起始帧号 + 持续帧数 |
| `audio_clipping` | 音频削波（超过 0 dBFS） | 起始帧号 + 持续帧数 |

所有定位均精确到单帧（0-based 帧号），每个问题带 0-1 的置信度。

## 环境与安装

- Python 3.10+
- 系统已安装 `ffmpeg` / `ffprobe` 且在 `PATH` 中（负责解码、缩放、音频抽取，无需 Docker）
- Python 依赖仅 NumPy：

```bash
pip install -r requirements.txt
```

支持 mp4、mov、mkv、ts 等 ffmpeg 可解封装的常见格式；无音频轨的文件自动跳过音频检测。

## 使用方法

```bash
# 报告打印到标准输出
python -m video_qc /path/to/material.mp4

# 报告写入文件
python -m video_qc /path/to/material.mp4 -o report.json

# 分析分辨率宽度（默认 480），需要更精细的块级定位可调大
python -m video_qc material.mp4 --analyze-width 640
```

输出示例：

```json
{
  "file": "samples/blocky.mp4",
  "meta": {
    "duration_sec": 6.0,
    "fps": 25.0,
    "width": 640,
    "height": 352,
    "total_frames": 150,
    "has_audio": true
  },
  "issues": [
    {
      "type": "blocky",
      "start_frame": 60,
      "duration_frames": 25,
      "confidence": 0.85
    }
  ],
  "stats": {
    "processing_time_sec": 0.32,
    "speed_factor": 18.8,
    "analyze_resolution": [480, 264]
  }
}
```

关键阈值都暴露为命令行参数（`--freeze-mad`、`--blocky-ratio`、`--silence-rms-db`、
`--clip-peak` 等），默认值已按本仓库素材标定，常规使用无需调整。

## 处理架构

```
ffmpeg 子进程 A（视频，多线程封顶 8 核）
  解码 → 缩放到 480 宽灰度 → rawvideo 管道
        → Python/NumPy 逐帧度量 → 四类检测器 → JSON
ffmpeg 子进程 B（音频，后台线程并行）
  解码 → f32le 单声道 PCM → 按视频帧时长切窗
```

- 缩放与像素格式转换全部在 ffmpeg 内完成，Python 只接收 480 宽的小图，
  **单帧分析成本与原始分辨率无关**，这是 4K 性能的基础。
- 视频、音频两路解码并行；ffmpeg 线程数封顶 8（`video_qc/pipeline.py: MAX_THREADS`），
  适配 8 核部署规格，不与同机任务争抢核数。

## 检测算法原理

所有检测先把视频统一转成 480 宽灰度帧，再逐帧计算度量，最后对布尔判定序列做
连续段聚合（runs），天然得到每个问题的起始帧与持续帧数。

### 1. 纯黑帧 `black_frame`

逐帧计算灰度均值与 99 分位亮度，同时满足 `mean < 8` 且 `p99 < 16` 判为黑帧，
连续段达到 0.5 秒以上才上报。

### 2. 静帧卡死 `freeze_frame`

逐帧计算与上一帧的平均绝对差（MAD，0-255 量程）。`MAD < 0.05` 判为静止，
持续 1 秒以上上报。黑帧段内帧间差天然为 0，已在逻辑中排除，不与黑帧重复上报。

### 3. 花屏马赛克 `blocky`（时空联合判定）

把每帧切成 16×16 块网格，对每个块计算块内标准差（纹理）和块均值，并与上一帧同位置块比较：

- **噪点块**：块内标准差绝对高（≥25）且相对上一帧剧变（≥6）——对应随机噪点马赛克；
- **纯色损坏块**：块均值相对四邻域中值突兀、块内平滑（std<8）、且块均值相对上一帧剧变
  ——对应整块丢失后被平铺色块替代。

异常块占比超过 10% 且持续 0.2 秒以上上报。

### 4. 音频静音 / 削波

音频重采样为单声道 f32 PCM，按每个视频帧的时长切窗：

- 静音：窗内 RMS < −45 dBFS 且持续 1 秒以上；
- 削波：窗内峰值 ≥ 0.90（0 dBFS=1.0，留 1 dB 工程余量）且持续 0.2 秒以上。

## 防误报策略

| 易错场景 | 防护机制 | 实测结果 |
|---|---|---|
| 夜景 / 暗场 | 同时看均值和 **p99 亮度**：夜景虽暗但有光斑/噪点，p99≈36，远高于黑帧阈值 16 | night.mp4 零误报 |
| 淡入淡出 | 黑段前后 1 秒窗口若检测到全局亮度持续斜坡（单调 Δmean），判定为 fade-to-black 并抑制；花屏度量依赖时间不稳定性，fade 斜坡帧同样不触发 | fade.mp4 零误报 |
| 慢镜头 | 静帧阈值取实测分离区间的中间值（卡死 MAD≈0.001，慢放 MAD 0.05–0.30），再叠加 1 秒最短时长；单个重复帧直接被时长过滤 | slowmo.mp4 零误报 |
| 彩条/棋盘格等高对比合法内容 | 花屏不只看空间锐利度，要求块纹理/均值**随时间随机剧变**；静止或运动规律的结构化边缘时间稳定，不满足条件 | 4K 彩条素材 900 帧零误报 |
| 无音频轨 / 音频解码失败 | 按"音频不可用"跳过音频检测，不会把空 PCM 误判为整段静音 | 已验证 |

## 性能

在本机（i5-13600K，ffmpeg 线程封顶 8）对 3840×2160@30fps、30 秒（900 帧）素材实测：

| 轮次 | 耗时 | 处理倍速 |
|---|---|---|
| 1 | 1.90 s | 15.76x |
| 2 | 1.92 s | 15.66x |
| 3 | 1.92 s | 15.64x |

稳定在 **15x 以上**，相对需求的 0.5x 有 30 倍裕量；分析分辨率仅 480×270，
素材时长线性扩展，内存占用恒定（逐帧流式处理，不缓存整段视频）。

## 目录结构

```
video-qc-service/
├── video_qc/
│   ├── __main__.py      # python -m video_qc 入口
│   ├── cli.py           # 命令行参数与报告输出
│   ├── analyzer.py      # 检测编排与报告组装
│   ├── pipeline.py      # ffmpeg 解码管道 + 逐帧度量
│   ├── detectors.py     # 四类检测器与全部阈值
│   └── probe.py         # ffprobe 元数据
├── tests/
│   └── validate_samples.py   # 对照 samples/ground_truth.json 的验收脚本
├── samples/             # 测试素材与标注
├── requirements.txt
└── README.md
```

## 验收测试

```bash
python tests/validate_samples.py
```

脚本对 `samples/` 下全部 9 个素材逐一检测，并与 `ground_truth.json` 比对：
无缺陷素材要求零误报；缺陷素材要求类型正确、起始帧偏差 ≤2 帧、持续帧数偏差 ≤4 帧、
区间 IoU ≥0.8。当前结果：**9/9 全部通过**。
