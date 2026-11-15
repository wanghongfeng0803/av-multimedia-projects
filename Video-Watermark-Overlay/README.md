# Video-Watermark-Overlay

从零实现的图片水印叠加工具：**不调用任何图像合成/处理库**，像素缓冲自行读写，
PNG 编解码仅用标准库 `zlib`/`struct` 手写，核心算法（src-over 混合、双线性缩放、
边缘检测、积分图）全部逐像素实现。运行时**零第三方依赖**，仅需 Python 3.10+。

## 核心原理

### 1. 预乘 alpha 的 src-over 合成

直接在直接 alpha 空间混合会在半透明边缘产生颜色误差，本工具先把像素预乘
（`rgb' = rgb * a / 255`），再按 Porter-Duff *source over destination* 合成：

```
out_a   = a_s + a_d · (1 - a_s)
out_rgb = rgb_s + rgb_d · (1 - a_s)        # 均为预乘值
```

输出时反预乘 `rgb = rgb' · 255 / out_a`。全程 0..255 定点整数，除法用
`(x + 127) // 255` 四舍五入；全透明源、全不透明源、全透明目标均有专门快速路径。

### 2. 预乘空间双线性插值缩放

采样坐标采用半像素中心映射 `src = (dst + 0.5) · scale - 0.5`，负坐标用 `floor`
（而非向 0 截断），边界 clamp。RGB 在**预乘后**再做四邻域加权，避免透明边缘
混入黑色（黑边），插值后反预乘回直接 alpha。

### 3. 九宫格锚点定位

`top/center/bottom` × `left/center/right` 共 9 个锚点，定位后可再叠加像素偏移。
定位不裁剪——原点允许为负或超出帧边界；混合层按帧矩形逐像素裁剪，
因此水印超帧、部分越界、完全在帧外都安全（无交集时零像素被修改）。

### 4. 基于帧内容的自适应蒙版

对帧的 Rec.601 亮度平面计算两路“关键区域”信号，取最大值：

- **边缘强度**：3x3 Sobel 梯度幅值（边界像素 clamp 采样，杜绝假边缘）；
- **局部亮度变化**：积分图 O(1) 求邻域均值，像素与均值的偏差。

细节强度经线性区间映射为逐像素透明度系数：

```
t      = clamp((detail - detail_low) / (detail_high - detail_low), 0, 1)
alpha' = base_alpha · (1 - mask_strength · t)
```

平坦区域保持原透明度，边缘/纹理（人脸轮廓、文字、物体边界）处自动变淡，
`mask_strength=0` 时退化为固定透明度。

### 5. 随时间变化的运动水印（关键帧轨迹）

位置偏移与旋转角都是时间 t 的连续函数，由多关键帧分段插值得到：

- **帧率映射**：`t = frame_index / fps`，支持 29.97 等任意正帧率，
  整数帧索引先映射为秒再对轨迹求值；
- **缓动插值**：`linear` / `ease-in` / `ease-out` / `ease-in-out`，
  作用于每个关键帧段的进度参数；t 超出首末关键帧时保持端点状态；
- **角度处理**：插值沿最短弧（350°→10° 走 +20°），最终结果归一化到
  [0, 360)，负角度与超 360° 输入自动折叠；
- **旋转 + 亚像素平移**：绕可配置锚点（相对坐标，默认中心）旋转，
  逆映射双线性重采样在**预乘 alpha 空间**进行，越界邻居按全透明参与，
  旋转空白角自然透明、半透明边缘无黑边；亚像素位移精确烘焙进采样坐标，
  画布原点取整（floor 包围盒角点）不引入定位误差；
- **逐帧蒙版**：自适应蒙版按当前帧的实际位置与旋转后尺寸实时重算；
- **退化保证**：角度为 0 且位移为 0 时走快速路径，结果与静态叠加逐字节一致。

### 6. 分散布局的抗裁剪水印

单点水印（如固定右下角）一旦被裁掉对应区域就彻底丢失。分散布局把帧
划分为 `rows x cols` 个单元格，每个单元格放置一份水印实例——任意局部
裁剪都会保留其余区域的实例，四周留白 `margin` 进一步避免贴边实例被
整条裁掉：

- **确定性布局**：位置抖动（`jitter`）与每实例旋转角抖动
  （`angle_jitter`）都由 `seed` 驱动的 `random.Random` 产生，相同种子
  布局完全一致——批量处理多帧时布局统一、结果可复现；
- **实例互不重叠**：抖动幅度不超过单元格内可动范围的一半，水印不大于
  单元格时每个实例完整落在本单元格内；水印大于单元格时按中心居中，
  越界部分交给混合层逐像素裁剪；
- **效率设计**：水印只缩放一次、角度为 0 的实例共享同一份像素；自适应
  蒙版的帧级细节图（Sobel + 积分图）只计算一次供全部实例共享；需要
  旋转的实例逐个「变换 → 混合 → 释放」，任意时刻内存中至多一份旋转
  实例，与实例总数无关。

### 7. 多输入批量调度

`batch.py` 在单文件引擎之上提供批量编排，针对内存与时间开销做了专门设计：

- **水印只解码一次**：全部任务共享同一份只读水印像素，不随任务数增长；
- **帧流式处理**：每个任务独立「读帧 → 叠加 → 写盘 → 释放」，任意时刻内存中
  只有有限帧（顺序模式 1 帧，并发模式至多 `workers` 帧），与任务总数无关；
- **失败隔离**：单个任务的任何异常（输入缺失、PNG 损坏、参数非法）只记录到
  该任务的结果中，其余任务继续执行，报告按任务顺序汇总成功/失败明细；
- **可选并发**：`workers > 1` 启用线程池（PNG 编解码的 zlib 会释放 GIL，
  多线程有实际收益）；默认 `workers=1` 纯顺序执行，零线程开销，
  两种模式产出逐字节一致。

任务通过 JSON 清单描述，相对路径按清单所在目录解析，整个批处理目录可整体迁移。

## 目录结构

```
watermark_overlay/
  image.py        RGBA 像素缓冲 + 手写 PNG 解码(5 种滤波器)/编码
  blend.py        预乘 alpha src-over 逐像素混合
  interpolate.py  预乘空间双线性缩放与目标尺寸解析
  geometry.py     九宫格锚点定位
  mask.py         亮度/Sobel/积分图自适应蒙版
  animation.py    关键帧轨迹、缓动函数、角度归一化、帧率映射
  transform.py    预乘空间旋转 + 亚像素平移重采样
  scatter.py      分散布局（确定性网格 + 抖动）与 ScatterConfig 校验
  config.py       WatermarkConfig 参数校验
  engine.py       流水线编排（静态/动画/分散布局叠加主入口）
  batch.py        批量调度（失败隔离、共享水印、可选并发、清单加载）
  cli.py          命令行入口（单文件 / --batch 批量两种模式）
tests/            229 项 pytest 测试
demo/generate_demo.py
```

## 快速开始

```bash
# 基本叠加（默认居中、原尺寸、完全不透明）
python -m watermark_overlay frame.png watermark.png -o out.png

# 右下角 + 像素偏移 + 70% 透明度 + 等比缩放到宽 200
python -m watermark_overlay frame.png watermark.png -o out.png \
    --anchor bottom-right --offset-x -10 --offset-y -10 \
    --opacity 0.7 --width 200

# 启用内容自适应蒙版，关键区域自动降低水印可见度
python -m watermark_overlay frame.png watermark.png -o out.png \
    --adaptive --mask-strength 0.8 --detail-low 16 --detail-high 96

# 动画叠加：按 motion.json 的轨迹渲染第 12 帧（25fps → t=0.48s）
python -m watermark_overlay frame.png watermark.png -o out.png \
    --motion motion.json --frame-index 12 --fps 25

# 分散布局（抗裁剪）：2x3 网格平铺，抖动 0.4，种子 7，逐实例旋转 ±15°
python -m watermark_overlay frame.png watermark.png -o out.png \
    --scatter --scatter-rows 2 --scatter-cols 3 --scatter-jitter 0.4 \
    --scatter-angle-jitter 15 --scatter-seed 7 --opacity 0.5

# 批量模式：按清单处理全部任务，4 线程并发；单个失败不影响其余
python -m watermark_overlay --batch batch.json --workers 4
```

批量清单 `batch.json` 示例（相对路径按清单所在目录解析）：

```json
{
  "watermark": "wm.png",
  "config": {"anchor": "bottom-right", "opacity": 0.7},
  "motion": "motion.json",
  "fps": 25.0,
  "jobs": [
    {"input": "frames/a.png", "output": "out/a.png"},
    {"input": "frames/b.png", "output": "out/b.png",
     "config": {"opacity": 0.5}, "frame_index": 12}
  ]
}
```

`config` / `motion` / `fps` 为共享默认值，每个任务可用同名字段覆盖；
退出码：全部成功 0，部分或全部失败 1，清单/参数非法 2。
把 `motion` 换成 `scatter` 即批量分散布局（两者互斥）：

```json
{
  "watermark": "wm.png",
  "config": {"opacity": 0.5},
  "scatter": {"rows": 2, "cols": 3, "jitter": 0.4, "seed": 7},
  "jobs": [
    {"input": "frames/a.png", "output": "out/a.png"},
    {"input": "frames/b.png", "output": "out/b.png"}
  ]
}
```

`motion.json` 示例（关键帧时间必须严格递增，x/y/angle 可省略）：

```json
{
  "keyframes": [
    {"time": 0.0, "x": 0, "y": 0, "angle": 0},
    {"time": 1.0, "x": 40, "y": 10, "angle": 90},
    {"time": 2.0, "x": 0, "y": 0, "angle": 360}
  ],
  "easing": "ease-in-out",
  "anchor_x": 0.5,
  "anchor_y": 0.5
}
```

作为库调用：

```python
from watermark_overlay import overlay_image, read_png, write_png, WatermarkConfig

frame = read_png("frame.png")
wm = read_png("watermark.png")
result, info = overlay_image(
    frame,
    wm,
    WatermarkConfig(
        anchor="bottom-right", offset_x=-10, offset_y=-10,
        target_width=200, opacity=0.7,
        adaptive=True, mask_strength=0.8,
    ),
)
write_png("out.png", result)
print(info.blended_pixels, info.origin)
```

批量处理（作为库调用）：

```python
from watermark_overlay import (
    BatchJob, WatermarkConfig, read_png, run_batch,
)

watermark = read_png("wm.png")  # 水印只解码一次，全批共享
jobs = [
    BatchJob(input_path="a.png", output_path="out/a.png"),
    BatchJob(input_path="b.png", output_path="out/b.png",
             config=WatermarkConfig(opacity=0.5)),  # 任务级覆盖
]
report = run_batch(
    jobs, watermark,
    config=WatermarkConfig(anchor="bottom-right", opacity=0.7),
    workers=4,  # 默认 1：纯顺序执行
)
for outcome in report.failures():
    print("失败：", outcome.job.input_path, outcome.error)
print(f"成功 {report.succeeded}/{report.total}，耗时 {report.elapsed:.2f}s")
```

分散布局（作为库调用）：

```python
from watermark_overlay import (
    ScatterConfig, WatermarkConfig, overlay_scattered, read_png, write_png,
)

frame = read_png("frame.png")
wm = read_png("watermark.png")
result, info = overlay_scattered(
    frame, wm,
    WatermarkConfig(opacity=0.5),
    ScatterConfig(rows=2, cols=3, jitter=0.4, angle_jitter=15, seed=7),
)
write_png("out.png", result)
print(info.instance_count, info.blended_pixels)  # 6 个实例
```

动画叠加（逐帧渲染，帧序列可循环调用）：

```python
from watermark_overlay import (
    Keyframe, MotionConfig, WatermarkConfig,
    overlay_animated_frame, read_png, write_png,
)

frame = read_png("frame.png")
wm = read_png("watermark.png")
motion = MotionConfig(
    keyframes=[
        Keyframe(0.0, x=0, y=0, angle=0),
        Keyframe(2.0, x=40, y=10, angle=90),
    ],
    easing="ease-in-out",
    anchor_x=0.5, anchor_y=0.5,  # 旋转锚点（相对水印尺寸）
)
config = WatermarkConfig(anchor="center", opacity=0.8)
for n in range(50):  # 25fps 的 2 秒
    out, info = overlay_animated_frame(
        frame, wm, config, motion, frame_index=n, fps=25.0
    )
    write_png(f"frame_{n:03d}.png", out)
```

## 参数与校验

| 参数 | 范围/取值 | 说明 |
| --- | --- | --- |
| `anchor` | 9 个 `*-*` 锚点 | 未知值直接拒绝 |
| `offset_x/y` | 整数（可负） | 锚点定位后的像素偏移 |
| `target_width/height` | 正整数或 None | 只给一边按宽高比推算 |
| `opacity` | [0.0, 1.0] | 0 时直接返回原帧，不做混合 |
| `mask_strength` | [0.0, 1.0] | 关键区域最大压暗比例 |
| `detail_low/high` | [0, 255] 且 low < high | 蒙版映射区间 |
| `mask_radius` | ≥1 整数 | 局部统计窗口半径 |
| `keyframes` | 时间严格递增，非空 | 运动轨迹关键帧（time/x/y/angle） |
| `easing` | linear / ease-in / ease-out / ease-in-out | 段内缓动函数 |
| `anchor_x/y`（运动） | [0.0, 1.0] | 旋转锚点相对坐标 |
| `frame_index` | ≥0 整数 | 整数帧索引 |
| `fps` | 正有限数（可非整数） | 帧率，t = frame_index / fps |
| `rows` / `cols`（分散） | ≥1 整数 | 网格行/列数，实例数 = rows × cols |
| `margin`（分散） | ≥0 整数 | 四边留白像素，布局区域向内收缩 |
| `jitter`（分散） | [0.0, 1.0] | 单元格内位置抖动幅度（占可动范围比例） |
| `angle_jitter`（分散） | [0, 180] | 每实例随机旋转角幅度（度） |
| `seed`（分散） | 整数 | 布局随机种子，相同种子布局完全一致 |

所有非法参数（越界、类型错误、布尔冒充整数等）在进入算法前统一抛出
`InvalidParameterError`；PNG 损坏/格式不支持抛出 `ImageDecodeError`；
CLI 捕获后输出中文错误并以退出码 2 结束。

## 边界情况覆盖

- 水印/配置全透明：结果与原帧逐字节相同；
- 水印超帧尺寸或负原点：逐像素裁剪，仅帧内部分参与混合；
- 完全无交集：正常写出原图并告警，混合像素数为 0；
- 半透明目标帧：正确累积 alpha（`a_s + a_d - a_s·a_d/255`）；
- 1px 图像、上/下采样、负采样坐标、越界 Sobel 邻居均有专门测试；
- 运动水印整体移出画布：零混合像素，帧保持不变；
- 旋转越界：包围盒扩展 + 逐像素裁剪，空白角全透明；
- 负角度 / 超 360°：归一化到 [0, 360)，450° 与 90° 结果逐字节一致；
- 角度 0 + 位移 0：退化为静态叠加，结果与 `overlay_image` 逐字节一致；
- 分散布局：相同种子两次叠加逐字节一致；jitter=0 时实例严格位于单元格
  中心；jitter=1 时实例仍不越出本单元格；水印大于单元格/帧时逐像素裁剪；
- 分散布局与运动轨迹互斥，同时给出时单文件模式报错（退出码 2）、
  批量模式仅该任务失败。

## 测试与演示

```bash
pip install -r requirements-dev.txt
python -m pytest          # 229 项测试
python demo/generate_demo.py   # 生成固定/自适应/缩放三组对比 PNG
```

## 支持的 PNG 子集

8 位采样、非隔行、RGB(类型 2)/RGBA(类型 6)；解码支持 None/Sub/Up/Average/Paeth
全部 5 种扫描线滤波器并校验每个 chunk 的 CRC。其余位深、色彩类型、Adam7 隔行
明确报错而不是静默产出错误结果。
