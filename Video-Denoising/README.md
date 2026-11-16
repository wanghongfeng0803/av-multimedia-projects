# Video-Denoising（vdenoise）

均值滤波视频降噪。处理接在 ffmpeg 解码与编码之间：解码为原始帧 →
可分离盒式滤波 → 重新编码，音轨 `-c:a copy` 直通。零三方依赖
（Python 标准库 + ffmpeg/ffprobe 子进程）。

## 用法

```bash
python denoise.py 输入.mp4 输出.mp4 [-r 半径] [--budget-mb 预算]
```

- `-r/--radius`：滤波半径，窗口 (2r+1)²，默认 2；
- `--budget-mb`：单帧分带内存预算（MB），缺省按可用内存的 1/4 推导。

也可作为库使用：

```python
from vdenoise import denoise_video
stats = denoise_video("in.mp4", "out.mp4", radius=2)
```

## 设计要点

- **可分离盒式滤波**（`vdenoise/filter.py`）：二维均值拆成水平前缀和 +
  垂直列运行和两遍，复杂度与半径无关；边界窗口截断不越界，除以实际
  像素数；两遍各自舍入，与全局均值直算最多差 ±1。
- **横向分带压内存**（`vdenoise/bands.py`）：长视频逐帧流式处理，
  单帧再按内存预算分带。每带读核心行上下各 **2r** 行上下文、输出
  核心行加各 r 行扩展区；相邻带重叠的 2r 行两带都是精确值，线性
  加权合并退化为恒等操作——分带结果与整帧滤波**逐像素相等**
  （零容差测试锁定）。预算按核心行 + 4r 行扣减。
- **内存预算推导**（`vdenoise/memory.py`）：ctypes/sysconf 探测可用
  物理内存，取 1/4，保底 4 MiB，探测失败兜底 512 MiB。
- **像素格式自适应**（`vdenoise/pixfmt.py`）：gray / yuv420p /
  yuv422p / yuv444p / rgb24 / gbrp 直接作为工作格式，其余回退
  yuv420p；packed RGB 拆成 3 个平面分别滤波。RGB 工作格式自动换
  libx264rgb（libx264 只收 YUV 系）；gray 输出转 yuv420p。
- **流水线**（`vdenoise/pipeline.py`）：ffprobe 探测 → ffmpeg 解码
  rawvideo → 逐帧分带滤波 → ffmpeg 编码，第二路 `-i` 供音轨
  `-c:a copy`；子进程 stderr 一律 DEVNULL 防管道死锁。

## 测试

```bash
python -m pytest
```

88 项：纯逻辑（滤波正确性、分带零容差等价、预算推导、像素格式
映射）不依赖 ffmpeg；集成测试（端到端、音轨保留、RGB/nut、gray
转换、小预算多带）在无 ffmpeg 时自动 skip。
