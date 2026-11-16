# Video-Motion-Detection（vmotion）

从零实现的视频运动检测工具：**分块帧差 + 局部自适应阈值 + 光照突变抑制**。
纯 Python 标准库实现，ffmpeg 经子进程调用，无第三方运行时依赖。

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
   抑制期内再次突变则重新计时。抑制期检测输出冻结，但参考帧仍滚动，
   否则 holdoff 会被旧参考无限重新触发。

之后再经形态学开/闭运算（游程表实现的腐蚀膨胀）去噪填洞、4 邻接连通域
分析与面积过滤，最后由滞后状态机（`trigger_frames` 去抖、
`release_frames` 迟滞释放）输出稳定的运动态。

## 三模块解耦

| 模块 | 职责 | 不做什么 |
| --- | --- | --- |
| `vmotion.detector` | 逐帧只出二值掩码与 `Region` 区域列表 | 不可视化、不落盘 |
| `vmotion.render` | 只消费检测 JSON 画区域框/状态行 | 不跑任何检测算法 |
| `vmotion.events` / `vmotion.model` | 只消费区域，沿状态边沿组装事件并落盘 | 不接触像素判定 |

检测 JSON 是三者间唯一交换格式。区域结构（外接框、面积、质心）同时是
后续背景建模复用的最小信息单元。检测与标记是两次独立的 ffmpeg 解码，
同一份 JSON 可反复换样式重渲染。

## 用法

```bash
# 1) 检测：产出 JSON（不出视频），全部阈值可配
python motion.py detect 输入.mp4 检测.json \
    --block 8 --local-radius 3 --pixel-threshold 12 --gain 3.0 \
    --min-area 64 --open-radius 1 --close-radius 1 \
    --trigger-frames 2 --release-frames 3 \
    --illum-change-threshold 25 --illum-hold-frames 2

# 2) 标记：读 JSON 出带框视频，可重复执行
python motion.py mark 输入.mp4 检测.json 标记.mp4 --font-scale 1
```

退出码：`0` 成功且检测到事件；`1` 运行错误（文件/ffmpeg/参数）；
`2` 流程成功但全程无运动事件（JSON 仍有效，便于脚本编排）。

也支持 `python -m vmotion detect ...`。

## 检测 JSON 结构

```json
{
  "schema_version": 1,
  "meta":   { "source", "width", "height", "fps", "fps_float" },
  "params": { "block", "gain", "pixel_threshold", "...全部检测参数" },
  "stats":  { "frames", "motion_frames", "motion_ratio",
              "held_off_frames", "events", "regions_total" },
  "frames": [ { "frame", "pts_time", "motion", "event_id",
                "regions": [ {"label","x","y","w","h","area","cx","cy"} ],
                "stats": { "raw_pixels", "gated_pixels", "mad",
                           "local_mad", "mean_delta", "held_off",
                           "illum_level" } } ],
  "events": [ { "event_id", "start_frame", "end_frame",
                "start_pts", "end_pts", "duration", "motion_frames",
                "region_count", "peak_frame", "peak_area", "bbox",
                "end_of_stream" } ]
}
```

掩码本身不写入 JSON（体积不可接受），只存区域几何与逐帧统计；需要逐像素
掩码时重放检测即可。标记视频左上角状态行（`MOTION E1 R3` / `HOLDOFF` /
`IDLE`）与事件列表的 `event_id` 逐帧可对照。

## 参数说明

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--block` | 8 | 分块边长（像素），越大越抗噪、定位越粗 |
| `--local-radius` | 3 | 局部 MAD 邻域半径（单位：块） |
| `--pixel-threshold` | 12 | 块均值差硬阈值（0..255），自适应结果不低于它 |
| `--gain` | 3.0 | 局部 MAD 增益，越大越保守 |
| `--min-area` | 64 | 连通域最小面积（全分辨率像素） |
| `--open-radius` / `--close-radius` | 1 / 1 | 形态学开闭运算半径，0 跳过 |
| `--trigger-frames` | 2 | 连续命中确认帧数（去抖） |
| `--release-frames` | 3 | 连续无命中释放帧数（迟滞） |
| `--illum-change-threshold` | 25 | 全局均值差突变阈值（0..255） |
| `--illum-hold-frames` | 2 | 突变后强制抑制帧数 |

## 工程结构

```
motion.py                 CLI 入口（detect / mark 两个子命令）
vmotion/
  detector.py             检测核心：块均值、局部 MAD 门控、holdoff、状态机
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

纯 Python 实现在 640×480 上约 13 fps（开发机实测），定位是算法学习而非实时
监控。热点在形态学的全帧游程转换与连通域；块均值已用逐行累加优化到每像素
一次访问，局部 MAD 走积分图 O(块数)。检测只在亮度平面进行，色度不参与
判定（对帧差贡献低且数据量翻倍）。

## 测试

```bash
pytest
```

53 个用例：形态学数学性质、连通域统计、局部/全局光照对照、holdoff 重新
计时、去抖迟滞、面积过滤、事件边沿、JSON 往返校验、字节级绘图，以及
需要 ffmpeg 的端到端流水线（无 ffmpeg 自动 skip）。
