# Video-Denoising（vdenoise）

视频降噪，含三种滤波模式。处理接在 ffmpeg 解码与编码之间：
解码为原始帧 → 滤波 → 重新编码，音轨 `-c:a copy` 直通。零三方
依赖（Python 标准库 + ffmpeg/ffprobe 子进程）。

- **均值模式 `mean`（默认）**：可分离盒式滤波，边界窗口截断；
- **中值模式 `median`**：Huang 直方图加速中值滤波，镜像边界，
  专门压制椒盐噪声；
- **时域模式 `temporal`**：全局位移估计对齐后再做时间窗混合，
  运动区逐像素掩码避免拖影；缓存帧数 / 片长 / 内存预算三者互推，
  超预算在启动 ffmpeg 前直接拒绝。

## 用法

```bash
# 均值（默认，参数与历史版本完全一致）
python denoise.py 输入.mp4 输出.mp4 [-r 半径] [--budget-mb 预算]

# 中值：窗口边长按素材配置（正奇数），镜像边界
python denoise.py 输入.mp4 输出.mp4 --mode median --window 5

# 中值并把每帧边界区域落盘，供单独抽帧复核
python denoise.py 输入.mp4 输出.mp4 --mode median --window 3 --dump-border 目录

# 时域：全局对齐 + 时间窗混合，缓存/片长/预算自动互推
python denoise.py 输入.mp4 输出.mp4 --mode temporal [--temporal-radius 2]
python denoise.py 输入.mp4 输出.mp4 --mode temporal --max-cache 7 --band-height 16
```

- `-r/--radius`：空间滤波半径，窗口 (2r+1)²，默认 2（均值模式）；
- `--mode`：`mean`（默认）/ `median` / `temporal`；
- `--window`：中值模式窗口边长（正奇数），给出时优先于 `--radius`；
  缺省沿用 `-r`，即默认窗口 5×5；window=1（radius=0）时中值退化为恒等；
- `--dump-border`：中值模式把每帧边界区域（距边缘 ≤ 半径的像素，
  掩码外置 0）写成工作格式原始字节 `frame_XXXXXX_border.raw`，
  并附 `manifest.json`；
- `--budget-mb`：内存预算（MB），缺省按可用内存的 1/4 推导。

时域模式专属参数：

- `--temporal-radius`：时间窗半径 r，窗为 [t-r, t+r]，默认 2（名义 5 帧）；
- `--max-shift`：全局整像素位移的搜索半径，默认 6；
- `--motion-threshold`：对齐残差超过该灰度差即判运动区，该参考帧在该
  像素不参与混合，默认 24；全部参考被拒时回退当前帧，杜绝拖影；
- `--max-cache`：缓存帧数硬上限，缺省取模块内置上限（15）；
- `--band-height`：显式片长（每带核心行数），缺省在预算内取最大片长。

边界 raw 可用 ffmpeg 读回（工作格式与尺寸见 manifest）：

```bash
ffmpeg -f rawvideo -pix_fmt yuv420p -s 64x48 -i 目录/frame_000003_border.raw 复核.png
```

也可作为库使用：

```python
from vdenoise import denoise_video
stats = denoise_video("in.mp4", "out.mp4", mode="median", window=5,
                      border_dir="border_frames")
# 时域：超预算抛 MemoryError（在任何子进程启动之前）
stats = denoise_video("in.mp4", "out.mp4", mode="temporal",
                      temporal_radius=3, max_shift=8, motion_threshold=24)
```

## 设计要点

- **可分离盒式滤波（均值）**（`vdenoise/filter.py`）：二维均值拆成
  水平前缀和 + 垂直列运行和两遍，复杂度与半径无关；边界窗口截断
  不越界，除以实际像素数；两遍各自舍入，与全局均值直算最多差 ±1。
- **Huang 直方图加速中值（中值）**（`vdenoise/median.py`）：中值不
  可分离，单遍二维处理。维护 256 桶直方图，横向滑动时移出/移入两个
  竖条，中值用累计计数增量行走定位，每像素代价 O(r + 256)。
- **镜像边界（中值）**：采用 REFLECT_101（不复制边缘像素），每个
  输出窗口恒为 (2r+1)² 个有效像素，与均值模式的截断语义刻意区分。
- **时域全局对齐混合（时域）**（`vdenoise/temporal.py`）：
  - **不按同坐标混合**：先在块平均下采样网格做粗级穷举、再回原分辨率
    精级邻域块匹配，最小化采样 SAD，求每个参考帧到当前帧的整像素
    全局位移 (dx, dy)；位移以亮度平面为准，色度按采样比缩放
    （4:2:0 位移减半）。
  - **运动区防拖影**：对齐后逐像素比较，残差超 `motion_threshold`
    即拒绝该参考帧在此像素的贡献；所有参考被拒时回退当前帧原值，
    绝不平均错位内容。混合为均匀权重整数均值，窗口在序列边界自然截断。
  - **有界帧缓存**：时间窗 [t-r, t+r]，实际缓存帧数
    K=min(2r+1, 硬上限, 预算可负担)，由有界 deque 淘汰旧帧，长视频
    内存与总帧数无关；第 t 帧在读到 t+half 帧后才发射，结尾冲刷尾部，
    输出帧数与逐帧 pts 与输入严格对齐。
  - **缓存 / 片长 / 预算三互推**（`plan_temporal`）：统一内存口径
    `峰值 = (单帧字节×K + 带工作区原始字节)×safety`。给定预算可反推
    缓存帧数与每带核心行数，也可显式指定片长反推可负担帧数；预算连
    「1 帧缓存 + 每平面 1 行带」都放不下时抛 `MemoryError`，在任何
    ffmpeg 进程启动之前拒绝，不产生半成品。
- **横向分带压内存**（`vdenoise/bands.py` / `median.py`）：长视频
  逐帧流式处理，单帧再按内存预算分带。均值分带读 2r 行上下文并对
  重叠区加权合并（分带与整帧逐像素相等，零容差测试锁定）；中值
  分带只需 r 行上下文；时域混合无空间邻域，带间行不重叠、无需上下文。
- **帧数与时间戳对齐**：三种模式共用同一套 rawvideo 帧循环——
  一帧入、一帧出，framerate 原样透传给编码器，输出帧数与逐帧
  pts_time 与输入严格对齐（集成测试锁定）。
- **内存预算推导**（`vdenoise/memory.py`）：ctypes/sysconf 探测可用
  物理内存，取 1/4，保底 4 MiB，探测失败兜底 512 MiB。
- **像素格式自适应**（`vdenoise/pixfmt.py`）：gray / yuv420p /
  yuv422p / yuv444p / rgb24 / gbrp 直接作为工作格式，其余回退
  yuv420p；packed RGB 拆成 3 个平面分别滤波。RGB 工作格式自动换
  libx264rgb（libx264 只收 YUV 系）；gray 输出转 yuv420p。
- **流水线**（`vdenoise/pipeline.py`）：ffprobe 探测 → ffmpeg 解码
  rawvideo → 滤波 → ffmpeg 编码，第二路 `-i` 供音轨 `-c:a copy`；
  子进程 stderr 一律 DEVNULL 防管道死锁。

## 测试

```bash
python -m pytest
```

189 项：纯逻辑（均值/中值/时域滤波正确性、中值对暴力直算零容差、
分带零容差等价、直方图加速性能护栏、镜像填充、边界掩码、预算推导、
位移估计找回已知平移、运动掩码回退、帧缓存有界、内存三互推与提前
拒绝、像素格式映射）不依赖 ffmpeg；集成测试（端到端、帧数/pts 对齐、
椒盐与静态噪声压制、平移镜头、音轨保留、RGB/nut、gray 转换、边界
转储、小预算多带、超预算开工前拒绝）在无 ffmpeg 时自动 skip。
