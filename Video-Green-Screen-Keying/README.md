# Video-Green-Screen-Keying

色度键抠像工具：从画面采样键色（**不写死绿色**，绿/蓝/红/任意纯色幕通用），
输出带 alpha 通道的透明视频/图像序列，或直接做背景替换。纯 Python 标准库
实现，ffmpeg / ffprobe 通过子进程调用。

## 三层架构（层间解耦，各自可单测、可复用）

```
输入 ──ffmpeg 解码 rgb24──> [层1 键控] RGBA
                            [层2 alpha 后处理 + despill 刷新]
                            [层3 背景替换? 或 alpha 直出] ──ffmpeg 编码──> 输出
  └──────────── 音轨不进像素管道，-c:a copy 直通 ────────────┘
```

| 层 | 模块 | 职责 | 复用出口 |
|----|------|------|----------|
| 层 1 键控 | `vkeyer.keyer` / `vkeyer.sampler` | 键色采样、RGB 欧氏距离双阈值软 alpha、despill 去残色 | `Keyer.key_frame` |
| 层 2 alpha 后处理 | `vkeyer.matte` | 阈值硬化、choke 收缩、盒式羽化 | 每个算子独立可调，后续羽化/背景替换直接用 |
| 层 3 输出编码 | `vkeyer.encoder` / `vkeyer.pipeline` | alpha 容器选型、source-over 合成、音轨直通、帧循环编排 | `composite_rgba_frame` 等 |

## 功能

- **键色画面采样，不写死绿色**：默认从顶部条带采样，也支持 8 个边缘
  条带 / 全帧 / 归一化矩形；多帧多像素取**每通道直方图中位数**
  （对前景误入、压缩噪点、幕布阴影稳健），也可 `--key R,G,B` 显式指定。
- **双阈值软 alpha**：RGB 欧氏距离 `d <= similarity` 全透明、
  `d >= similarity+softness` 全不透明，中间线性过渡；`softness=0` 退化为硬键。
- **边缘不残绿（despill）**：只在半透明像素上、按 `1-alpha/255` 权重把
  键色主导通道压到另两通道最大值；不透明像素零损伤。蓝幕自动改压蓝通道。
- **alpha 后处理三件套**（固定顺序：硬化 → choke → 羽化）：
  - 阈值硬化消灭幕布区稀疏噪点；
  - choke 取局部最小值收缩前景边缘，几何上去掉一圈幕布色；
  - 盒式羽化（与 vdenoise 同套截断边界 + 定点四舍五入）抹平滑硬边。
- **两种输出**：
  - alpha 模式：`.mov`(qtrle) / `.mkv`(ffv1) / `.png`、`.webp` 序列；
  - 背景替换：合成到纯色或背景图片/视频，直接出 `.mp4` 等普通视频。
- **音轨 ` -c:a copy` 直通**，无音轨输入也能跑（`?` 可选映射）。
- **内存有界**：在途帧为常数，不随时长增长；背景视频更短时冻结最后一帧。
- **失败不出半成品**：所有参数/格式校验在启动子进程前完成；运行中失败
  删除本次新建的输出，预先存在的文件不动。

## 用法

```bash
# 采样键色（默认顶部条带），输出带 alpha 的 mov
python key.py green.mp4 -o out.mov

# 蓝幕：显式键色 + 收缩 1px + 羽化 1px
python key.py blue.mp4 -o out.mov --key 0,70,255 --choke 1 --feather 1

# 矩形区域采样（归一化坐标）
python key.py g.mp4 -o out.mov --sample-mode rect --sample-rect 0.05,0.05,0.3,0.2

# 背景替换出 mp4（纯色 / 背景素材二选一）
python key.py g.mp4 -o comp.mp4 --bg-color 0,0,0
python key.py g.mp4 -o comp.mp4 --bg-image backdrop.png

# 带 alpha 的 PNG 序列（路径含 %06d 帧号通配）
python key.py g.mp4 -o "seq/f_%06d.png"
```

也可作为库使用：`from vkeyer import key_video, Keyer, MatteConfig, ...`，
公开符号见 `vkeyer/__init__.py` 的 `__all__`。

## 输出格式与 alpha 承载（本机 ffmpeg 9 实测）

| 输出 | 编码器 | 像素格式 | alpha | 说明 |
|------|--------|----------|-------|------|
| `.mov` | qtrle | argb | ✅ 可靠 | 默认推荐，无损 |
| `.mkv` | ffv1 | rgba（ffprobe 报 bgra） | ✅ 可靠 | 无损，bgra/rgba 仅字节序差异 |
| `.webm` | libvpx-vp9 | yuva420p | ⚠️ 不可靠 | 本机构建**静默剥离 alpha**，运行结果 `messages` 给警告 |
| `.png`/`.webp` 序列 | png / libwebp | rgba | ✅ 可靠 | webp 用无损模式 |
| `.mp4` | libx264 | yuv420p | ❌ 不支持 | 仅背景替换模式可用，alpha 模式直接报错（退出码 9） |

## 实测的 ffmpeg 坑

1. **ffmpeg 9 移除 `-vsync`**：`select` 滤镜抽帧后必须用
   `-fps_mode passthrough` 透传选中帧，否则报 `Unrecognized option 'vsync'`。
2. **`-vf` 必须放在全部 `-i` 之后**：夹在两个输入之间会被误绑到后续输入，
   报 `Option vf cannot be applied to input url`。
3. **libvpx-vp9 静默丢 alpha**：编码日志声明 yuva420p，产物却落 yuv420p，
   调 `-auto-alt-ref 0` / `-g 1` / lossless 各组合均无效（仅本机此构建）。
   工具不硬失败，保留出口并在 `messages` 明确警告。
4. **恒不透明 alpha 平面会被编码器剥离**：这是正确优化，验证 alpha 是否
   保留时测试素材必须含真透明像素，不能用全 255 的裸流。

## 测试

```bash
pytest
```

纯逻辑测试（键控/采样/后处理/合成，不依赖 ffmpeg）与真实 ffmpeg 集成测试
分离；无 ffmpeg 时集成测试自动 skip。共 70+ 用例，覆盖双阈值边界、despill
权重、choke/羽化镜像边界、合成两端精确、三种 alpha 容器、背景替换、
音轨直通、帧数对齐、覆盖保护与失败清理。

## 退出码

| 码 | 含义 |
|----|------|
| 1 | 通用 / 参数错误 |
| 2 | 找不到 ffmpeg/ffprobe |
| 3 | 输入文件不存在 |
| 4 | 输入无视频流/缺尺寸 |
| 5 | 输出已存在且未允许覆盖 |
| 6 | ffmpeg 执行失败 |
| 7 | 裸流帧中途截断 |
| 8 | 键色采样失败 |
| 9 | 输出格式无法承载 alpha |
