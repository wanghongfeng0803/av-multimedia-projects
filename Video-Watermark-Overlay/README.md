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

## 目录结构

```
watermark_overlay/
  image.py        RGBA 像素缓冲 + 手写 PNG 解码(5 种滤波器)/编码
  blend.py        预乘 alpha src-over 逐像素混合
  interpolate.py  预乘空间双线性缩放与目标尺寸解析
  geometry.py     九宫格锚点定位
  mask.py         亮度/Sobel/积分图自适应蒙版
  config.py       WatermarkConfig 参数校验
  engine.py       流水线编排（叠加主入口）
  cli.py          命令行入口
tests/            87 项 pytest 测试
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

所有非法参数（越界、类型错误、布尔冒充整数等）在进入算法前统一抛出
`InvalidParameterError`；PNG 损坏/格式不支持抛出 `ImageDecodeError`；
CLI 捕获后输出中文错误并以退出码 2 结束。

## 边界情况覆盖

- 水印/配置全透明：结果与原帧逐字节相同；
- 水印超帧尺寸或负原点：逐像素裁剪，仅帧内部分参与混合；
- 完全无交集：正常写出原图并告警，混合像素数为 0；
- 半透明目标帧：正确累积 alpha（`a_s + a_d - a_s·a_d/255`）；
- 1px 图像、上/下采样、负采样坐标、越界 Sobel 邻居均有专门测试。

## 测试与演示

```bash
pip install -r requirements-dev.txt
python -m pytest          # 87 项测试
python demo/generate_demo.py   # 生成固定/自适应/缩放三组对比 PNG
```

## 支持的 PNG 子集

8 位采样、非隔行、RGB(类型 2)/RGBA(类型 6)；解码支持 None/Sub/Up/Average/Paeth
全部 5 种扫描线滤波器并校验每个 chunk 的 CRC。其余位深、色彩类型、Adam7 隔行
明确报错而不是静默产出错误结果。
