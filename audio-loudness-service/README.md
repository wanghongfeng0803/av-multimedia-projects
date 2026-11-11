# 批量音频响度处理服务 (Audio Loudness Service)

台里验收口径 **-23 LUFS / 真峰值 -1 dBTP**。本服务对成片做批量响度归一化：
**一次测量 → 一次增益 → 一次真峰值限幅**，一次成片，不调用 `loudnorm` 两遍流程
（两遍 = 先测再压，相当于对音频做两次动态处理，会引入二次压缩失真）。

## 处理流程

```
输入 (wav/mp3/flac/aac/mp4/mov/mkv...)
  │
  ├─ 1. 解码 + 5.1 下混立体声 (ITU-R BS.775)
  │
  ├─ 2. 测量: 综合响度 I (BS.1770-4, K 加权 + 门控)
  │
  ├─ 3. 增益 = 目标响度 - I  (纯线性增益, 不改变动态)
  │
  ├─ 4. 若真峰值 > -1 dBTP: 真峰值限幅器 (4x 过采样域, 前视 + 指数释放)
  │
  ├─ 5. 复测输出 (报告用) → 写 24-bit WAV
  │
  └─ 6. 前后对比报告 (JSON + CSV)
```

### 为什么不用 loudnorm 两遍

`loudnorm` 动态模式内部含压缩器，第一遍测量、第二遍处理时会对音频做动态范围
压缩；本服务第 3 步只做**线性增益**（整体电平平移，动态完全保留），只有真峰值
超天花板时限幅器才介入，且只作用于超标的瞬时峰值，对响度的影响通常 < 0.2 LU。

### 测量算法 (BS.1770-4)

- **K 加权**：高架滤波 (4 dB @ 1.5 kHz 以上) + 高通 (38 Hz)，由 pyloudnorm 提供
  （与 ffmpeg `ebur128` 同一套滤波器定义）。
- **综合响度**：400 ms 块 / 75% 重叠，绝对门限 -70 LUFS，相对门限为未门控均值
  再降 10 LU，门控后块能量平均。
- **LRA**：3 s 块 / 1 s 步进，绝对门限 -70 LUFS、相对门限 -20 LU，取门控后短时
  响度的 P95 - P10。
- **真峰值**：4 倍过采样（多相 FIR）后取峰值，单位 dBTP。

### 5.1 下混公式 (ITU-R BS.775)

```
Lo = L  + 0.7071·C + 0.7071·Ls
Ro = R  + 0.7071·C + 0.7071·Rs
```

LFE 默认**丢弃**（BS.775 不参与响度下混；`--include-lfe` 可按 0.7071 混入）。
声道顺序按 WAV 标准 FL FR FC LFE SL/BL SR/BR。单声道复制为双声道。

### 真峰值限幅器

- 4 倍过采样域计算包络（立体声联动，避免声像偏移）；
- 前视 1.5 ms（增益提前下坡，无硬拐点）+ 100 ms 指数释放；
- 增益在过采样域施加后降采样回原采样率，输出真峰值保证 ≤ 天花板；
- 未超标时完全不介入（旁通）。

### 响度与真峰值冲突时

若素材峰值因数过高（如峰值主导响度的内容），压到 -1 dBTP 后响度可能无法
同时落在 -23 ± 0.5 内。此时**真峰值优先**（硬指标），报告中以 `warning`
字段标出偏差量，交人工定夺。

## 安装

```bash
pip install -r requirements.txt
```

- wav / flac / mp3 / ogg / aiff：纯 Python 解码（libsndfile），无需 ffmpeg；
- aac / m4a / mp4 / mov / mkv / ac3：需要 PATH 中有 `ffmpeg` + `ffprobe`。

## 使用

```bash
# 批量处理 (递归扫描, 输出保持相对目录结构)
python -m audio_loudness input_dir output_dir

# 完整参数
python -m audio_loudness input_dir output_dir \
    --target -23 --true-peak -1 \
    --concurrency 8 --retries 2 \
    --report out/report

# 只重跑上次报告中失败的条目
python -m audio_loudness input_dir output_dir --retry-failed out/report.json
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--target` | -23 | 目标综合响度 (LUFS) |
| `--true-peak` | -1 | 真峰值天花板 (dBTP) |
| `--concurrency` | 8 | 并发上限（进程池） |
| `--retries` | 2 | 失败重试次数，指数退避 2s→4s→…(封顶 30s) |
| `--include-lfe` | 关 | 5.1 下混时混入 LFE |
| `--report` | `<output_dir>/report` | 报告路径前缀（生成 .json + .csv） |
| `--retry-failed` | — | 指定上次报告 JSON，只重跑失败条目 |

## 报告格式

每条目包含（JSON / CSV 同字段）：

- **输入**：时长、采样率、声道数、LUFS、LRA、True Peak
- **输出**：LUFS、LRA、True Peak、应用增益 (dB)、限幅器最大衰减 (dB)
- **状态**：`success` / `failed`（含 error）/ `skipped`（静音或过短）
- `attempts`：实际尝试次数；`warning`：响度超差提示

## 复核 (ebur128)

```bash
ffmpeg -i out/xxx.wav -filter_complex ebur128=peak=true -f null -
```

对照报告中的 `out_lufs` / `out_true_peak_db` 即可（同为 BS.1770 测量，
差异应在 ±0.1 LU / ±0.1 dB 内）。

## 自检

```bash
python scripts/selftest.py
```

生成 5 类合成素材（安静正弦 / 高响度音乐 / 5.1 环绕 / 超幅短峰 / 极端峰值），
跑完整批量流程并断言：下混公式正确、输出 -23 ± 0.5 LUFS、真峰值 ≤ -1 dBTP、
限幅器按需介入、极端素材真峰值达标且有 warning。

## 目录结构

```
audio_loudness/
  meter.py      # BS.1770 响度 / LRA / 真峰值测量
  dsp.py        # 5.1 下混、线性增益、真峰值限幅器
  io.py         # 解码 (soundfile 优先, ffmpeg 回退) 与 24-bit WAV 写出
  processor.py  # 单条流水线: 测量→增益→限幅→复测→写盘
  batch.py      # 进程池并发、指数退避重试、进度、报告
  cli.py        # 命令行入口
scripts/selftest.py
```
