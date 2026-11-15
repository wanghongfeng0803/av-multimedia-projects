# Video-Scene-Segmentation

基于**相邻帧灰度直方图差异**的场景切换（硬切）检测工具。直方图统计与差异
度量全部从零实现，**不依赖 opencv / numpy / Pillow 等任何图像库**；Python
侧仅使用标准库，ffmpeg/ffprobe 通过子进程调用，负责视频解码与灰度转换。

## 设计目标与达成方式

| 要求 | 实现方式 |
| --- | --- |
| 自研直方图统计，不依赖图像库 | `collections.Counter` 的 C 级计数 + 固定 256 级装箱（`histogram.py`） |
| 单遍流式、不重复解码 | ffmpeg 输出 `rawvideo/gray` 原始字节流，生成器逐帧产出，评分器只保留 **3 条直方图** |
| 核心统计 O(帧数 × 灰度级数) | 每帧统计一次、每边界固定 256 项整数 L1；总耗时随帧数线性增长 |
| 边界：单帧视频 | 零边界合法结果：差异序列为空、阈值走兜底、切点为空，不报错 |
| 边界：长时间无变化 | 全 0 序列使 MAD 退化为 0，阈值走兜底，零误报 |
| 边界：亮度/曝光突变 | 单帧闪光的成对尖峰 + 三帧回返距离判定，同时抑制进入/离开两条边界 |
| 可复现 | 全程整数统计、排序统计无随机量、输出无时间戳字段、键序与浮点格式固定，同输入逐字节一致 |

## 快速开始

```bash
# 终端摘要
python scene_seg.py input.mp4

# 完整差异序列 CSV + 完整 JSON 报告
python scene_seg.py input.mp4 --csv diff.csv --json report.json

# python -m 形式等价
python -m scene_seg_tool input.mp4 -o report.json
```

终端输出示例（76 帧视频：灰场 25 帧 → 1 帧白闪 → 灰场 25 帧 → 黑场 25 帧）：

```
输入: demo_out/demo.mp4
帧数: 76  尺寸: 160x120  帧率: 25.000000 (avg_frame_rate)
判定阈值: 0.500000 [gap_midpoint]
硬切点数: 1
闪光抑制边界: 25, 26

切点帧号        时间(秒)    归一化差异    整数距离
      51      2.040000      1.000000       38400
```

### 命令行参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--mad-k K` | 9.0 | 稳健阈值系数，阈值 = 中位数 + K × 1.4826 × MAD |
| `--fallback-threshold T` | 0.35 | 全静止/单帧等退化场景的兜底归一化阈值，范围 [0, 1] |
| `--min-scene-frames N` | 1 | 最小场景长度（帧），冲突切点按距离做确定性 NMS，1 = 关闭 |
| `--no-flash-suppression` | 关 | 关闭曝光突变抑制 |
| `--csv` / `--json` | — | 写出完整差异序列 / 完整报告 |
| `-o` | — | 按扩展名（.csv/.json）选择单一输出 |
| `--quiet` | 关 | 不打印终端摘要 |

退出码：`0` 正常（含未检出切点）、`2` 找不到 ffmpeg/ffprobe、`3` 输入
不存在、`4` 无视频流/零帧、`6` ffmpeg 执行失败或流被截断。

## 算法

### 1. 灰度化与直方图

ffmpeg 以 `-pix_fmt gray` 解码为 8-bit 单通道灰度（BT.601 加权由 ffmpeg
内部完成），每帧即一串 0..255 字节。直方图 `h[v]` 统计灰度值 `v` 的
像素数，固定 256 级。

统计实现对比（720p 单帧，本机 CPython 3.10）：

| 实现 | 单帧耗时 |
| --- | --- |
| **`Counter(frame)` + 256 次装箱（采用）** | **16.9 ms** |
| 纯 Python 逐像素循环 | 23.9 ms |
| `bytes.count` 扫 256 遍 | 56.0 ms |

`Counter` 的元素计数在 C 级循环（`_count_elements`）中完成，且只扫描
字节流一遍。

### 2. 差异度量：整数 L1 距离

```
d(a, b) = Σ_v |h_a[v] - h_b[v]|          # 整数，范围 [0, 2×像素数]
score    = d / (2 × 像素数)               # 归一化到 [0, 1]，仅在判定前做一次
```

两帧完全相同为 0，灰度分布完全不重叠为 1。中间统计全程整数加法，不存在
浮点舍入，这是可复现性的基础；直方图交（intersection）与 L1 仅差常数倍，
语义等价。

### 3. 自适应阈值（中位数 + MAD）

硬切在差异序列中是少数极端值，普通均值/标准差会被其带偏，因此采用对
异常值稳健的排序统计：

1. `MAD > 0`：`阈值 = median + K × 1.4826 × MAD`（1.4826 是 MAD 对标
   正态标准差的一致性常数，K 默认 9）；
2. `MAD = 0` 但序列有不同取值（离散常态 + 离散异常）：取「大于中位数的
   最小取值」与中位数的**空隙中点**（来源标记 `gap_midpoint`）；
3. 序列为空或全相同（含全 0 的长静止视频、单帧视频）：使用兜底阈值
   0.35（来源标记 `fallback`）。

阈值超过 1.0 时不做截断——此时天然没有边界能通过判定，等价于无硬切。

### 4. 曝光突变（闪光）抑制

单帧闪光（闪光灯、白闪、黑闪）在差异序列上留下**成对尖峰**：进入闪光
帧 `t` 的边界 `t` 与离开的边界 `t+1` 距离都很大，但跳过闪光帧后两侧
画面几乎相同——三帧回返距离 `|h[t-1] - h[t+1]|` 很小。真实硬切
A→B→C 的回退距离 `|h[A] - h[C]|` 通常依然很大，不会被误抑制。

判定（闪光帧 `t`，要求 `1 ≤ t ≤ N-2`）：两条边界归一化距离都达到
阈值，且回返距离 < 两条边界距离最大值的 25%（`FLASH_RATIO`）。
命中后边界 `t` 与 `t+1` 同时标记为 `flash`。

**能力边界**：持续 2 帧及以上的多帧闪光/慢淡入淡出不存在单帧三元组
回返关系，不在抑制范围内，可用 `--min-scene-frames` 缓解尖峰簇。

### 5. 最小场景长度 NMS

两个保留切点至少相隔 N 帧。冲突时按「整数距离更大者优先、距离相同
则边界下标更小者优先」的**全序**规则处理，任何输入下结果唯一。

## 流式处理与复杂度

```
ffmpeg(gray 原始流) → read_exact_frame(精确成帧) → Scorer.push()
                                                       ├─ build_histogram   O(像素数), C 级
                                                       ├─ 相邻 L1           O(256)
                                                       └─ 三帧回返 L1       O(256)
```

- 视频只解码**一遍**，无回放、无 seek、无帧缓存；评分器任意时刻只持有
  3 条直方图（`prev / cur` 及到达中的新帧），直方图内存恒为 O(256)；
- 核心统计复杂度 **O(帧数 × 灰度级数)**，直方图的像素扫描为 O(帧数 ×
  像素数) 但由 C 级循环完成；
- 实测 720p 灰度帧直方图部分约 35–60 帧/秒（纯 Python 评分，随机器负载
  波动），总耗时随帧数线性增长；ffmpeg 解码与评分在两个进程间流水线并行；
- stderr 由独立守护线程持续排空，避免 Windows 管道容量（约 4KB）导致
  的双向死锁；生成器提前关闭时在 `finally` 中终止并回收 ffmpeg 进程；
- 流末尾残留不足一帧的字节时显式报错，不让残缺帧参与统计。

## 输出格式

**CSV**（`--csv`，每条相邻帧边界一行，固定 6 位小数、`\r\n` 行尾）：

```
frame_index,time_seconds,raw_distance,score,return_distance,decision
1,0.040000,0,0.000000,0,below_threshold
25,1.000000,38400,1.000000,0,flash
26,1.040000,38400,1.000000,38400,flash
51,2.040000,38400,1.000000,38400,cut
```

`decision` 取值：`cut`（确认硬切）、`flash`（闪光抑制）、
`suppressed`（最小场景长度 NMS 抑制）、`below_threshold`（未达阈值）。

**JSON**（`--json`）：含探测元数据、阈值与来源、闪光/NMS 抑制边界清单、
全部切点与**完整边界序列**。键顺序固定，UTF-8 无 BOM，无任何运行时刻
字段（时间戳、进程 id 等），同一输入多次运行逐字节相同。

## 库级用法

```python
from scene_seg_tool import detect_scenes, detect_scenes_in_frames

# 文件：内部一次 ffprobe + 一次 ffmpeg 单遍解码
result = detect_scenes("input.mp4", min_scene_frames=12)
for cut in result.cuts:
    print(cut.frame_index, cut.time_seconds, cut.score)

# 内存灰度帧（无需 ffmpeg，便于嵌入其它管线）
result = detect_scenes_in_frames(frame_iter, width=160, height=120, fps=25.0)
```

## 项目结构

```
scene_seg_tool/
├── histogram.py   # 直方图统计、整数 L1、中位数/MAD 阈值、闪光抑制、NMS（纯逻辑）
├── detector.py    # StreamingHistogramScorer 流式评分 + 检测编排（纯逻辑 + 文件入口）
├── frames.py      # ffmpeg gray 原始流单遍解码、精确成帧、stderr 排空
├── probe.py       # ffprobe 元数据探测与帧率解析
├── output.py      # 确定性 CSV/JSON 序列化
├── models.py      # 不可变结果数据模型
├── errors.py      # 统一异常与退出码
├── binaries.py    # ffmpeg/ffprobe 查找（PATH / 环境变量 / 同目录伴随）
├── cli.py         # 命令行入口
└── __main__.py    # python -m scene_seg_tool
tests/             # 77 项测试：63 项纯逻辑（无需 ffmpeg）+ 14 项端到端集成
```

## 测试

```bash
pytest                 # 无 ffmpeg 时集成测试自动 skip，纯逻辑测试照常运行
pytest -q              # 当前环境：77 passed
```

测试覆盖：直方图三种边界、L1 上下界与对称性、阈值三个分支、闪光成对
抑制与真实切换的区分、NMS 全序规则、单帧/长静止/硬切/闪光完整管线、
截断输入不挂死、CSV/JSON 结构与逐字节可复现性。

## 已知能力边界

- 只检测**硬切**；渐变（淡入淡出、叠化）表现为连续小幅差异，不产生
  超过阈值的尖峰，按设计不报告；
- 多帧（≥2 帧）持续闪光无单帧回返关系，不抑制；
- VFR 视频的时间戳由平均帧率换算，为近似值（帧号与差异序列本身精确）；
- 灰度化依赖 ffmpeg 的 BT.601 实现；直方图与差异度量本身完全自研。
