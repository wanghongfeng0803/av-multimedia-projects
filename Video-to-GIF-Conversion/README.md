# Video-to-GIF-Conversion（vgif）

零三方依赖的视频转 GIF 工具：ffmpeg 只负责解码，调色板量化、帧率采样、
LZW 压缩与 GIF 封装全部用 Python 标准库从零实现。核心目标是**固定帧率
转换时的时序零漂移**与**抽帧 / 量化 / 编码三步严格解耦**。

## 设计约束（本项目要解决的问题）

1. **体积与画质权衡，默认参数写清依据**（见下文「默认参数」）；
2. **帧延迟精确对应原帧率**，不能因调色板生成或帧率换算出现快慢漂移；
3. **抽帧、量化、编码严格分三步解耦**，抽帧 / 量化产物可被后续缩放与
   采样复用，改目标参数不必重新解码源视频。

## 时序零漂移是怎么做到的

GIF 每帧延迟写在图形控制扩展（GCE）里，单位是**厘秒（1/100 秒）**且为
整数。常见实现用 `round(100/fps)` 反推延迟，例如 12fps 时
`100/12≈8.33` 取 8，每帧少 0.33cs，长片段累积成可感知的播放加速。

本包的口径（`vgif/timing.py`）：

- 延迟**只由源时长与输出帧数决定，与调色板、缩放、量化完全无关**；
- 源时长按「末帧 PTS + 末帧持续时间」计算（与 frconv 项目同一口径，
  不能只用末帧 PTS，否则平白少算一帧）；
- 总厘秒数半值向上取整后，用 Bresenham 把余数均匀撒到各帧——每帧延迟
  只可能相差 1 厘秒，**总时长误差严格小于 1 厘秒，且不随帧数累积**；
- 取帧用端点锚定 Bresenham（`divmod(N-1, M-1)`），强压缩（如 60→24）
  也不丢首尾帧，相邻输出帧的源下标间距只可能是 q 或 q+1；
- **安全下限**：历史浏览器 / 查看器普遍把 0、1 厘秒钳制为 10 厘秒。
  均摊延迟低于 2 厘秒（目标 >50fps）时直接报错，不静默产出会被查看器
  改写时序的文件。

实测：24fps / 2.0s 源转 12fps，GIF 24 帧延迟只取 {80, 90}ms 两档，
总时长 2000ms，偏差 0ms（PIL 与 ffprobe 独立校验一致）。

## 三步解耦

| 阶段 | 模块 | 职责 | 产物 |
| --- | --- | --- | --- |
| 抽帧 | `vgif/extract.py` | ffprobe 探测逐帧 PTS/时长；ffmpeg `-fps_mode passthrough` 逐帧 1:1 解码为 rgb24 | `frame_XXXXXX.rgb` + `manifest.json` |
| 量化 | `vgif/sampling.py` `scale.py` `quantize.py` | 端点锚定取帧 → RGB 双线性缩放 → 中位切分全局调色板 → 最近邻索引映射 | 调色板 + 索引帧（可 pickle） |
| 编码 | `vgif/gif_writer.py` | 纯 Python LZW + GIF89a 封装，厘秒延迟原样写入 | `.gif` |

抽帧产物与帧率 / 调色板无关，换 `--fps`、`--max-width`、`--colors`
重跑时通过 `--work-dir` 直接复用；量化产物也可在索引域最近邻缩放
（`--indexed-scale`，适合像素风素材）零颜色损失复用。

## 默认参数与依据

| 参数 | 默认 | 依据 |
| --- | --- | --- |
| `--fps` | `12` | 100/12 经厘秒均摊为 8/9 厘秒交替，节奏均匀且在浏览器安全区；对 24/25/30fps 源为整数级丢帧。体积约为 15fps 的八成、25fps 直转的一半 |
| `--max-width` | `480` | GIF 仅 256 色且 LZW 压缩率有限，480px 是社媒贴图常用上限；再宽体积近似按像素平方增长，而画质受 256 色上限制约，收益递减 |
| `--colors` | `256` | 用满全局颜色表换取最好量化质量；需要压体积时降到 128/64（实测同素材 256→64 色体积约降三分之一） |

实测体积（320×240 / 2s testsrc2）：256色12fps 222KB，128色 182KB，
64色 148KB；同色数下 15fps 256KB、10fps 185KB。

## 用法

完整转换：

```bash
python gif_convert.py 输入.mp4 输出.gif \
    [--fps 12] [--max-width 480] [--colors 256] \
    [--work-dir DIR] [--no-reuse] [--indexed-scale]
```

分步执行（每步产物独立可复用）：

```bash
python gif_convert.py extract  输入.mp4 工作目录/
python gif_convert.py quantize 工作目录/ 索引产物.pkl --fps 12 --max-width 480
python gif_convert.py encode   索引产物.pkl 输出.gif
```

Python API：

```python
from vgif import convert_to_gif
result = convert_to_gif("in.mp4", "out.gif",
                        target_fps=12, max_width=480, colors=256,
                        work_dir="./vgif_work")  # 给出目录即可跨调用复用
print(result.gif_duration, result.duration_error, result.delays_cs)
```

## 工程约定

- Python 侧只用标准库；ffmpeg/ffprobe 通过子进程调用（PIL 仅在测试里
  作独立校验，运行期零依赖）。
- 中文 docstring 解释设计决策，公开符号列入 `__all__` 并由包
  `__init__.py` 重新导出。
- 纯逻辑测试不依赖 ffmpeg；集成测试在无 ffmpeg 时自动 skip。
- 定点缩放（1/65536）与 Bresenham 取帧均为纯整数运算，结果确定性、
  跨平台可复现。

## 测试

```bash
python -m pytest -q
```

95 项测试：时序规划（端点锚定 / 厘秒均摊 / 安全下限）、中位切分量化、
双线性与索引缩放、LZW 往返（自带独立解码器核对）、GIF 字节结构，以及
ffmpeg 端到端（时序零漂移、三步复用、CLI、抽帧 1:1）。

## 模块说明

- `vgif/timing.py`：帧下标 Bresenham 映射 + 厘秒延迟规划（零漂移核心）
- `vgif/extract.py`：ffprobe 探测 + ffmpeg 抽帧（passthrough、stderr 防死锁）
- `vgif/sampling.py`：RGB / 索引两条同构采样路径
- `vgif/scale.py`：RGB 双线性（定点）+ 索引域最近邻
- `vgif/quantize.py`：中位切分全局调色板 + 立方缓存最近邻映射
- `vgif/gif_writer.py`：GIF89a 封装 + LZW 变体编码
- `vgif/pipeline.py`：三步编排、默认参数、工作目录复用
