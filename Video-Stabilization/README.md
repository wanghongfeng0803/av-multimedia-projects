# Video-Stabilization 视频防抖

估计全局运动并输出补偿参数，**原视频只读不写**。平移与旋转分别
估计、分别输出；估计、平滑、补偿三步解耦，裁剪复用同一份参数文件。
纯 Python 标准库实现运动估计，像素变换全部交给 ffmpeg。

## 设计总览

全局运动只建模两类：**平移 (dx, dy)** 与 **绕画面中心的旋转角**。

| 阶段 | 模块 | 输入 → 输出 | 是否动像素 |
| --- | --- | --- | --- |
| 估计 | `stabilize/motion.py` `pipeline.py` | 视频 → `stabilize-motion` v1 JSON | 否，只读 |
| 平滑 | `stabilize/smoothing.py` | motion JSON → `stabilize-smooth` v1 JSON（含裁剪量） | 否 |
| 补偿 | `stabilize/compensate.py` | 视频 + smooth JSON → 稳定视频（含待裁黑边） | ffmpeg |
| 裁剪 | 同上 `crop_video` | 补偿视频 + 同一份 smooth JSON → 成品 | ffmpeg |

### 平移与旋转如何区分

- **平移**：缩小灰度帧（目标宽 96px，块平均下采样）上的粗-精两级
  块匹配，整像素穷举 SAD + 三点抛物线亚像素拟合。
- **旋转**：12 个同心环 × 128 角度 bin 的去均值环形签名，旋转在
  签名上表现为循环平移，用自研 FFT 做循环互相关求主峰，抛物线
  插值取亚 bin 精度。
- 两者交替精修一次（先平移后旋转，旋转补偿后复核平移），抑制耦合。
- 场景切换/低置信帧显式置 `valid=false`，平滑阶段线性插值，
  绝不把切点误当巨幅抖动。

### 稳定输出格式（版本化契约）

`stabilize-motion` v1：逐帧相邻增量（`dx/dy` 小图像素、`angle` 度、
`confidence`、`valid`、`scene_cut`），第 0 帧为恒等基准。

`stabilize-smooth` v1：每帧含 `trajectory`（原始累积轨迹）、
`smoothed`（平滑轨迹）、`compensation`（补偿量 = 平滑 − 原始），
另有全局 `crop`（统一四边 inset 与诊断字段）。后续阶段只依赖契约，
同一份 smooth JSON 可反复用于不同裁剪量，裁剪不重新估计。

### 补偿如何落地

`sendcmd` 命令文件按帧 PTS 同时驱动两个独立滤镜：
`rotate@st`（反向旋转角，弧度）与 `crop@cp`（移动取样原点实现
平移）。音轨 `-c:a copy` 直通，帧数/帧率不变。

### 黑边三种处理模式（--border-mode）

平移/旋转补偿会让部分画面移出画幅，露出黑边。三种处理模式是
**补偿期参数，不写进 smooth JSON**——同一份 JSON 可分别产出三种
结果，输出同尺寸、同帧率、同帧数，可直接逐帧对比：

| 模式 | 原理 | 选择依据 | 代价 |
| --- | --- | --- | --- |
| `fixed` 固定裁剪（默认，旧行为） | 全片统一 inset 裁边，Lanczos 缩回原尺寸 | 画面稳定、无伪影，适合交付成片 | **丢画面**：所有帧都按全片最大补偿量裁，抖动小的帧也白丢边缘；缩回有轻微放大模糊 |
| `dynamic` 动态裁剪 | 逐帧只裁该帧实际需要的量，帧内缩回原尺寸 | 想尽量保留画面内容，能接受轻微缩放变化 | **伪影**：逐帧缩放比不同会产生「呼吸」感（已用与轨迹同半径的时域平滑抑制）；不丢尺寸但低抖动帧被放大更多 |
| `mirror` 镜像填充 | 输入先按所需边距镜像扩边，补偿取样落在扩展区内 | 不能丢任何画面（如监控、证据素材） | **伪影**：边缘是镜像内容，边缘有文字/人脸等结构化纹理时可察觉；不丢画面、输出尺寸不变 |

`fixed` 需要再接 `crop` 子步骤；`dynamic`/`mirror` 的补偿输出即
成品（CLI 会提示无需 crop）。镜像扩边量由 `crop.raw_inset`（上限
钳制前的真实需求）按旋转角折算放大，补偿角超过约 30° 时扩边
发散，工具会拒绝并建议改用限幅。

### 补偿量限幅（--max-shift / --max-angle）

平滑窗口过大或轨迹突变时，补偿量可能过冲（把画面反向推过头）
或放大残余抖动。`smooth`/`run` 的 `--max-shift`（像素）与
`--max-angle`（度）对补偿量做**硬限幅**：

- **限幅与平滑不打架的关键是顺序**：限幅在平滑完成之后作用于
  最终补偿信号（平滑轨迹 − 原始轨迹），平滑器永远看不到被夹的
  值，二者无反馈回路；若反过来先夹轨迹再平滑，平滑器会试图
  「修复」夹子造成的折点，两者互相拉扯。
- 硬限幅只让补偿速度出现折点，位置保持连续，不引入新抖动。
- 裁剪量由限幅后的补偿推导，限幅直接降低裁剪量/扩边量。
- 默认不限幅（`None`），旧参数模式行为与 JSON 输出完全不变。

### 两种模式结果可对比

- 限幅开启时 smooth JSON 增加 `limits` 段：阈值、被夹帧数、
  限幅前后峰值（`raw_peak` vs `peak`），与不限幅的旧输出对照
  即可量化限幅影响；帧条目 schema 不变。
- 三种黑边模式输出同尺寸同帧率，可直接用 ffmpeg 逐帧对比：

```bash
ffmpeg -i 稳定_fixed.mp4 -i 稳定_mirror.mp4 -lavfi "[0:v][1:v]ssim" -f null -
```

## 用法

```bash
# 分步执行（每步产物可人工检查）
python stabilize_cli.py estimate  输入.mp4 motion.json
python stabilize_cli.py smooth    motion.json smooth.json --radius 15
python stabilize_cli.py compensate 输入.mp4 smooth.json padded.mp4
python stabilize_cli.py crop      padded.mp4 smooth.json 稳定.mp4

# 一把梭（中间产物默认放临时目录，自动清理）
python stabilize_cli.py run 输入.mp4 稳定.mp4 --keep-work

# 限幅 + 动态裁剪（输出即成品，无需 crop 步骤）
python stabilize_cli.py run 输入.mp4 稳定.mp4 --max-shift 12 --border-mode dynamic
```

常用参数：`--est-width`（估计小图宽）、`--radius`（平滑窗半径，
窗口 2r+1）、`--method uniform|gaussian`、`--max-shift`/`--max-angle`
（补偿限幅，默认不限）、`--border-mode fixed|dynamic|mirror`
（黑边处理，默认 fixed）、`--inset`（裁剪步骤覆盖推荐裁剪量）、
`--crf`。

退出码：0 成功；1 通用失败；2 参数/契约错误；8 ffmpeg 流水线失败。

## 工程约定

- 零三方依赖：FFT、块匹配、环形签名、轨迹平滑全部标准库实现，
  ffmpeg/ffprobe 走子进程。
- 中文 docstring，公开符号列入 `__all__` 并由包 `__init__` 重导出。
- pytest 套件：纯逻辑测试不依赖 ffmpeg；集成测试在无 ffmpeg 时
  自动 skip。运行：`python -m pytest`。

## 模块清单

- `stabilize/fft_mod.py`：迭代基-2 FFT / IFFT / 循环互相关
- `stabilize/motion.py`：下采样、块匹配、环形签名、帧对联合估计
- `stabilize/smoothing.py`：无效帧插值、刚体轨迹累积、平滑、补偿
  限幅、裁剪推导（固定/动态）
- `stabilize/pipeline.py`：ffprobe 探测、gray 解码驱动、motion JSON
- `stabilize/compensate.py`：sendcmd 生成（三种黑边模式）、ffmpeg
  补偿与裁剪
