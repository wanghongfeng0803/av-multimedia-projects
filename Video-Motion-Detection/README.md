# Video-Motion-Detection（vmotion）

从零实现的视频运动检测工具：**帧差 / 背景建模两种模式**，共用分块帧差、
局部自适应阈值与光照突变抑制。纯 Python 标准库实现，ffmpeg 经子进程调用，
无第三方运行时依赖。

## 两种检测模式

| | 帧差模式（默认） | 背景建模模式（`--background`） |
| --- | --- | --- |
| 参考模型 | 上一帧，整体滚动 | 逐块运行背景，选择性 EMA 更新 |
| 回答的问题 | 相邻帧间有没有「变化沿」 | 这块区域现在算不算背景 |
| 物体停下 | 立刻与参考一致，停止报警 | 先持续报警，停留足够久后吸收进背景 |
| 缓慢光照 | 逐帧小差异越不过门限，但参考无长期记忆 | 背景块持续 EMA 跟踪，平滑适应且不误报 |
| 开机阶段 | 首帧即开始检测 | 预热期累计平均、不出掩码，杜绝未稳定误报 |

两种模式共用块均值、局部 MAD 门控、形态学精炼与滞后状态机
（`vmotion/detection_common.py`），差异只在参考模型的维护方式；
`compare` 子命令在同一份解码帧流上并排跑两者，产出逐帧不一致清单。

## 背景建模的三个时间尺度目标

背景建模要同时满足三个互相冲突的目标，设计上用三套独立机制分别兜住：

1. **缓慢光照要自适应**：被判为背景的块每帧按
   `B ← (1-α)B + αX` 做指数滑动平均。每帧只差几个灰度级的光照漂移
   越不过门限，块始终按背景更新，背景平滑跟过去且全程零误报。
2. **更新速度可配**：`--learning-rate` 即 α（默认 0.05，约 20 帧
   时间常数）。α 越小跟踪越稳、抗瞬时扰动越强，越大适应越快。
3. **前景停留足够久后吸收**：每块维护连续前景计数，前景块默认
   **不更新**背景（移动物体不污染模型）；连续前景达到
   `--absorb-frames`（默认 60）后转入吸收态开始 EMA，背景追平后
   自动回到背景态——停下的车、放下的包裹最终成为新背景。

**初始化保护**：前 `--warmup-frames` 帧（默认 10，另加首帧）只做精确
累计平均、不出掩码、不推进去抖状态机，背景未稳定时从源头杜绝大面积
误报；预热结束后才切到 EMA 选择性更新。

光照突变 holdoff 在两种模式下语义一致（帧间全局均值差，相对**上一帧**
而非背景模型，否则小 α 时背景追不上跳变会无限重触发）；抑制期背景对
全部块做 EMA，尽快吸收跳变后的新亮度。

## 核心设计：光照变化怎么处理

需求是「光照突变要抑制，但不能用简单全局归一化一棒子打死」——全局亮度
变化要滤掉，车灯、阴影这类**局部光照变化**要能检出。本工具不做任何全局
归一化（归一化会把局部变化一并抹平），而是用三道闸门：

1. **分块帧差**：亮度平面先压成 `block×block` 块均值再做绝对差。块均值
   等价箱式低通，先压传感器噪声，也让「全局/局部」有了空间尺度。
2. **局部 MAD 自适应阈值**：每个块的判定阈值是
   `max(硬阈值, gain × 局部 MAD)`，局部 MAD 在以该块为中心的
   `local_radius` 邻域内计算（积分图 O(块数)）。
   - 全局亮度变化在**所有**邻域同步抬高 MAD → 处处阈值抬高 → 整帧被门控
     滤除；
   - 车灯/阴影只影响局部邻域，邻域外阈值不动，运动仍可越限检出。
3. **光照突变 holdoff**：帧间全局均值变化超过 `illum_change_threshold`
   判定为突变帧，该帧不出掩码；之后 `illum_hold_frames` 帧强制抑制，
   抑制期内再次突变则重新计时。帧差模式抑制期参考帧仍滚动，背景模式
   抑制期全块 EMA，都是为了避免 holdoff 被旧参考无限重新触发。

之后再经形态学开/闭运算（游程表实现的腐蚀膨胀）去噪填洞、4 邻接连通域
分析与面积过滤，最后由滞后状态机（`trigger_frames` 去抖、
`release_frames` 迟滞释放）输出稳定的运动态。

## 三模块解耦

| 模块 | 职责 | 不做什么 |
| --- | --- | --- |
| `vmotion.detector` / `vmotion.background` | 逐帧只出二值掩码与 `Region` 区域列表 | 不可视化、不落盘 |
| `vmotion.render` | 只消费检测 JSON 画区域框/状态行 | 不跑任何检测算法 |
| `vmotion.events` / `vmotion.model` | 只消费区域，沿状态边沿组装事件并落盘 | 不接触像素判定 |

检测 JSON 是三者间唯一交换格式，两种模式产出同构 JSON，标记模块不感知
模式差异（预热帧状态行显示 `WARMUP`）。检测与标记是两次独立的 ffmpeg
解码，同一份 JSON 可反复换样式重渲染。

## 用法

```bash
# 1a) 帧差模式检测（默认，旧行为不变）
python motion.py detect 输入.mp4 检测.json \
    --block 8 --local-radius 3 --pixel-threshold 12 --gain 3.0 \
    --min-area 64 --open-radius 1 --close-radius 1 \
    --trigger-frames 2 --release-frames 3 \
    --illum-change-threshold 25 --illum-hold-frames 2

# 1b) 背景建模模式：加 --background 与三个专有参数
python motion.py detect 输入.mp4 检测.json --background \
    --warmup-frames 10 --learning-rate 0.05 --absorb-frames 60

# 1c) 同输入双模式量化对比，输出逐帧不一致清单
python motion.py compare 输入.mp4 对比.json

# 2) 标记：读 JSON 出带框视频，两种模式的 JSON 都能直接消费
python motion.py mark 输入.mp4 检测.json 标记.mp4 --font-scale 1
```

退出码：`0` 成功且检测到事件（`compare` 成功恒为 0）；`1` 运行错误
（文件/ffmpeg/参数）；`2` 流程成功但全程无运动事件（JSON 仍有效，便于
脚本编排区分）。

也支持 `python -m vmotion detect ...`。

## 对比 JSON 结构

`compare` 在一次 ffmpeg 解码内同时喂两个检测器，输出：

```json
{
  "schema_version": 1,
  "frames": 30,
  "agreement": 0.8,
  "verdict_agreement": 0.7667,
  "counts": { "agree": 12, "frame_diff_only": 1,
              "background_only": 6, "region_count_diff": 11 },
  "disagreements": [ { "frame", "pts_time", "kind",
      "frame_diff_motion", "background_motion",
      "frame_diff_regions", "background_regions",
      "background_stats": { "warmup", "held_off",
                            "fg_blocks", "absorbed_blocks" } } ],
  "per_frame": [ { "frame", "frame_diff_motion", "background_motion", ... } ],
  "events": { "frame_diff": [...], "background": [...] },
  "event_counts": { "frame_diff": 1, "background": 1 },
  "params": { "mode": "compare", "frame_diff": {...}, "background": {...} }
}
```

不一致按去抖/迟滞后的**运动态布尔判定**分类：`frame_diff_only`（仅帧差
报警）、`background_only`（仅背景报警，预热期分歧多落此类）、
`region_count_diff`（双方都报警但区域数不同，单列信息项）。

## 检测 JSON 结构

```json
{
  "schema_version": 1,
  "meta":   { "source", "width", "height", "fps", "fps_float", "mode" },
  "params": { "block", "gain", "pixel_threshold", "...全部检测参数" },
  "stats":  { "frames", "motion_frames", "motion_ratio",
              "held_off_frames", "warmup_frames", "absorbing_frames",
              "events", "regions_total" },
  "frames": [ { "frame", "pts_time", "motion", "event_id",
                "regions": [ {"label","x","y","w","h","area","cx","cy"} ],
                "stats": { "raw_pixels", "gated_pixels", "mad",
                           "local_mad", "mean_delta", "held_off",
                           "illum_level", "warmup", "fg_blocks",
                           "absorbed_blocks", "bg_mean" } } ],
  "events": [ { "event_id", "start_frame", "end_frame",
                "start_pts", "end_pts", "duration", "motion_frames",
                "region_count", "peak_frame", "peak_area", "bbox",
                "end_of_stream" } ]
}
```

掩码本身不写入 JSON（体积不可接受），只存区域几何与逐帧统计；需要逐像素
掩码时重放检测即可。标记视频左上角状态行（`MOTION E1 R3` / `HOLDOFF` /
`WARMUP` / `IDLE`）与事件列表的 `event_id` 逐帧可对照。帧差模式的
stats 不含背景专有的 `warmup`/`fg_blocks` 等键，旧 JSON 读回行为不变。

## 参数说明

| 参数 | 默认 | 适用 | 含义 |
| --- | --- | --- | --- |
| `--block` | 8 | 两者 | 分块边长（像素），越大越抗噪、定位越粗 |
| `--local-radius` | 3 | 两者 | 局部 MAD 邻域半径（单位：块） |
| `--pixel-threshold` | 12 | 两者 | 块均值差硬阈值（0..255），自适应结果不低于它 |
| `--gain` | 3.0 | 两者 | 局部 MAD 增益，越大越保守 |
| `--min-area` | 64 | 两者 | 连通域最小面积（全分辨率像素） |
| `--open-radius` / `--close-radius` | 1 / 1 | 两者 | 形态学开闭运算半径，0 跳过 |
| `--trigger-frames` | 2 | 两者 | 连续命中确认帧数（去抖） |
| `--release-frames` | 3 | 两者 | 连续无命中释放帧数（迟滞） |
| `--illum-change-threshold` | 25 | 两者 | 全局均值差突变阈值（0..255） |
| `--illum-hold-frames` | 2 | 两者 | 突变后强制抑制帧数 |
| `--warmup-frames` | 10 | 背景 | 预热帧数（首帧后累计平均、不出掩码），0 关闭 |
| `--learning-rate` | 0.05 | 背景 | 背景 EMA 学习率 α（(0,1]，越小越稳） |
| `--absorb-frames` | 60 | 背景 | 前景块连续多少帧后吸收进背景 |

## 工程结构

```
motion.py                 CLI 入口（detect / mark / compare 三个子命令）
vmotion/
  detector.py             帧差模式：参考帧滚动、holdoff、状态机
  background.py           背景建模：预热累计平均、选择性 EMA、前景吸收
  detection_common.py     两模式共享：块均值、MAD 门控、形态学、滞后状态机
  comparison.py           同输入双检测器并排对比，产出不一致清单
  morphology.py           游程表腐蚀/膨胀、开闭运算、并查集连通域
  events.py               事件状态机（只消费区域）
  model.py                JSON 数据契约与序列化
  render.py               rgb24 字节平面手工绘图 + ffmpeg 编码
  pipeline.py             ffmpeg gray 解码 → 检测 → JSON 流水线
  io_utils.py             ffprobe/ffmpeg 探测与定长帧读取
  font.py                 内置 5×7 点阵字库（零依赖绘图）
tests/                    pytest 套件，纯逻辑不依赖 ffmpeg，端到端自动 skip
```

## 性能与取舍

纯 Python 实现在 640×480 上实测帧差模式约 12 fps、背景模式约 16 fps
（背景模式每帧只需一次块均值，参考更新是 O(块数) 的轻量 EMA），定位是
算法学习而非实时监控。热点在形态学的全帧游程转换与连通域；块均值已用
逐行累加优化到每像素一次访问，局部 MAD 走积分图 O(块数)。检测只在亮度
平面进行，色度不参与判定（对帧差贡献低且数据量翻倍）。

## 测试

```bash
pytest
```

87 个用例：形态学数学性质、连通域统计、局部/全局光照对照、holdoff 重新
计时、去抖迟滞、面积过滤、事件边沿、JSON 往返校验、字节级绘图，以及
背景建模专有的预热保护、慢光照零误报、学习率跟踪速度、前景吸收/瞬态
不污染、双模式分歧分类，和需要 ffmpeg 的端到端流水线（无 ffmpeg 自动
skip）。
