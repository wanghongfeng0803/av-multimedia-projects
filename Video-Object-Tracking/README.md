# Video-Object-Tracking（vtrack）

逐帧模板匹配的单目标追踪。追踪与渲染**严格分离**：追踪阶段只产出轨迹
JSON，渲染阶段只读 JSON 画框出视频，同一条轨迹可反复换样式重渲染，
JSON 也是后续光流追踪、多目标追踪复用的数据契约。零三方依赖
（Python 标准库 + ffmpeg/ffprobe 子进程）。

核心承诺：

- **跟丢要报告、不乱跟**：匹配分数低于阈值立即标记 `lost`，跟踪框
  冻结在最后已知位置，绝不跳到未达标的候选上；
- **恢复后重新搜索**：丢失后搜索区逐帧扩大直至铺满全帧，找到候选还要
  连续多帧确认才恢复，防止误匹配把目标「跟丢到别的东西上」；
- **追踪与渲染分离**：中间产物只有一份轨迹 JSON，渲染不做任何匹配。

## 用法

```bash
# 1) 追踪：逐帧模板匹配，产出轨迹 JSON（不产生视频）
python track.py track 输入.mp4 轨迹.json --box x y w h
python track.py track 输入.mp4 轨迹.json --box 12 12 16 16 \
    --method sad --threshold 30 --search-margin 12 \
    --reacquire-frames 2 --lost-growth 2

# 外部图片作为模板（替代从视频帧裁剪），仍需 --box 给出锚点位置
python track.py track 输入.mp4 轨迹.json --template-image tpl.png --box 12 12 16 16

# 2) 渲染：读轨迹 JSON 画框出视频，可对同一轨迹反复重渲染
python track.py render 输入.mp4 轨迹.json 输出.mp4 [--font-scale 2] [--last-known]
```

参数：

- `--box x y w h`：初始模板框（第 0 帧坐标 + 宽高），外部图片模板时
  取其 x/y 作为锚点；
- `--method`：`sad`（默认，绝对差均值）/ `ssd`（平方差均值）/
  `zncc`（归一化相关，抗线性光照变化）；
- `--threshold`：接受阈值。SAD/SSD 为每像素均值（越小越好，默认
  30 / 900），ZNCC 为相关系数（越大越好，默认 0.7）；
- `--search-margin`：追踪态局部搜索区相对最后已知框外扩的像素边距，
  默认 12；
- `--lost-growth`：丢失后搜索区每帧的复利增长因子，默认 1.5，铺满
  全帧后保持全帧；
- `--reacquire-frames`：恢复所需的连续达标帧数，默认 3；
- `--reacquire-threshold`：重确认专用阈值，缺省沿用主阈值，可设更严；
- `--update-template`：启用工作模板自适应（默认关闭，全程用初始
  模板防漂移），恢复时工作模板会重置回初始模板；
- `--template-image` / `--template-frame`：外部模板图片，或从指定
  视频帧裁剪（默认第 0 帧）；
- 渲染侧 `--font-scale` 点阵文字放大倍数，`--last-known` 给丢失帧
  额外画黄色虚线框标出最后已知位置。

**退出码**：0 成功；1 运行错误；2 追踪完成但视频结束时目标仍处于
丢失状态（轨迹 JSON 仍有效）。

也可作为库使用：

```python
from vtrack import track_video, render_video, load_track_json

# 追踪 → JSON
result = track_video("in.mp4", "track.json", (12, 12, 16, 16),
                     method="sad", threshold=30.0, search_margin=12)
for ev in result.loss_events:        # 丢失区间（含片末未恢复事件）
    print(ev.lost_frame, ev.recovered_frame, ev.frames_lost)

# 渲染（与追踪解耦，可多次调用）
render_video("in.mp4", "out.mp4", "track.json", show_last_known=True)

# 后续算法直接消费同一契约
track = load_track_json("track.json")
for rec in track.frames:             # rec.status / rec.bbox / rec.score ...
    ...
```

## 轨迹 JSON 契约

```json
{
  "schema_version": 1,
  "meta":   { "source": "...", "target_id": "target_0", "width": 128,
              "height": 96, "fps": "10/1", "initial_box": [12,12,16,16],
              "template_source": "frame:0" },
  "params": { "method": "sad", "threshold": 30.0, "search_margin": 12,
              "lost_growth": 1.5, "reacquire_frames": 3, ... },
  "frames": [
    { "frame": 0, "pts_time": 0.0, "status": "tracked",
      "x": 12, "y": 12, "w": 16, "h": 16,
      "score": 0.0, "confirmed": true, "search": [12,12,12,12] }
  ],
  "loss_events": [
    { "lost_frame": 8, "recovered_frame": 13, "lost_pts": 0.8,
      "recovered_pts": 1.3, "frames_lost": 5, "end_of_stream": false }
  ],
  "stats": { "frames": 20, "tracked_frames": 15, "lost_frames": 5,
             "lost_ratio": 0.25, "loss_events": 1,
             "still_lost_at_end": false,
             "score_min": 0.0, "score_max": 3.45, "score_mean": 1.61 }
}
```

- 帧状态对外只有 `tracked` / `lost` 两态；重确认（内部第三态）期间
  对外报 `lost` 且 `confirmed=false`；
- `lost` 帧的 `x/y/w/h` 是**最后已知位置**（不是当前位置），渲染默认
  不在此画跟踪框，避免误导；
- 片末仍未恢复时生成 `end_of_stream=true`、无 `recovered_frame` 的事件。

## 设计要点

- **三种匹配分数**（`vtrack/template.py`）：SAD/SSD 在搜索区内穷举、
  分数按窗口像素数归一化，阈值即「每像素灰度差 / 平方差」；ZNCC 走
  块平均下采样粗搜 + 原分辨率 ±2px 精搜的两级搜索，零方差窗按 −1
  处理避免除零。另随包提供积分图工具（`integral_image`/`rect_sum`）
  供后续盒式加速复用。
- **三态状态机**（`vtrack/tracker.py`）：
  - `TRACKED` 只在最后已知框的局部窗内匹配，达标才移动框；
  - 分数不达标立即冻结框进入 `LOST`，搜索边距取
    `max(m0·k, m0·growth**(k-1))`——线性项保证小边距、少帧数时每帧
    至少实打实外扩一个边距，复利项在长丢失段加速铺满全帧；
  - 重搜一律用**初始模板（anchor）**，候选须连续 `reacquire_frames`
    帧在锚点邻域内位置连贯且达标才恢复；任一帧不达标退回 `LOST`
    重新计数；恢复后工作模板重置回 anchor，杜绝漂移累积。
- **追踪与渲染分离**（`vtrack/pipeline.py` / `vtrack/render.py`）：
  追踪解码 `gray` 亮度平面（省 2/3 数据量），渲染独立解码 `rgb24`
  画彩色叠加层，两次 ffmpeg 流水线之间只交换 JSON；帧数与轨迹记录数
  不一致直接报错，不静默截断或补帧。
- **零依赖绘图**（`vtrack/font.py` / `vtrack/render.py`）：内置 5×7
  ASCII 点阵字库，矩形/虚线框/文字全部在 rgb24 字节平面上手写；文字
  两遍绘制（先收集前景坐标，描边环跳过前景再统一写色），避免后画
  像素的描边覆盖相邻已画前景。
- **子进程约定**：与兄弟模块一致，stderr 一律 DEVNULL 防管道死锁，
  ffprobe/ffmpeg 缺失或不可读统一抛 `PipelineError`。

## 测试

```bash
python -m pytest
```

72 项：纯逻辑（三种匹配方法的零误差定位、归一化分数、ZNCC 粗精搜索
与零方差处理、积分图、下采样、状态机的跟丢冻结/扩区重搜/连续确认/
误恢复拒绝/模板重置、JSON 契约往返与校验、点阵字库、rgb24 绘图字节）
不依赖 ffmpeg；集成测试（移动方块端到端、遮挡跟丢与恢复、片末未恢复
事件、音轨直通、帧数不匹配拒绝、外部模板图片、SAD/SSD/ZNCC 三方法、
JSON 复用字段）在无 ffmpeg 时自动 skip。
