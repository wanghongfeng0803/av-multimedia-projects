# fMP4 HLS 打包器 (HLS fMP4 Packager)

## 项目背景

要上 HLS，CDN 才给开边缘缓存。需要做一个基于 fMP4 分片的 HLS 打包器，**不要直接调 `ffmpeg -f hls`**，切点要可控。

## 需求

### 分片规格

- 分片时长 **2 秒**
- **GOP 与分片边界对齐**，切点只落在 **IDR 帧**
- 首片独立起播（第一个分片必须从 IDR 开始，不依赖前面的初始化段以外的数据）
- 单码率输出

### 断流续传

- 输入流中断后重新连接，能从**断点继续**打包，不从头开始
- `EXT-X-DISCONTINUITY` 位置正确（断流续接处、编码参数变化处）
- 时间戳连续计算，不出现回退或跳变

### m3u8 生成

- 自己写 m3u8 播放列表，不依赖 ffmpeg 输出
- 校验 m3u8 格式正确性（EXT-X-VERSION、EXT-X-TARGETDURATION、EXTINF 精度等）
- 输出每片的元数据：`start_time`（浮点秒）、`duration`、`sequence_number`

### 验收标准

- 用 `ffplay` 冷启动播放，**起播时间 3 秒以内**
- 播放过程中无卡顿、无花屏、无声画不同步
- 用 `ffprobe` 验证每个分片首帧为 IDR
- HLS 版本至少为 7（支持 fMP4）

### 技术约束

- 可以用 ffmpeg 做编解码和分片封装底层，但 **分片切割逻辑和 m3u8 生成必须自己控制**
- 不能直接用 `ffmpeg -f hls` 的输出结果当最终产物
- 初始化段（init segment）单独生成，每个分片引用同一个 init

## 技术栈建议

- Python + ffmpeg（子进程管道方式，读取 ffmpeg 输出的分片事件）
- 或直接操作 fMP4 盒子结构（moov/moof/mdat）

（也可自由选择其他技术栈）

## 交付要求

- 命令行工具：`hls-packager input.mp4 output_dir/ [--segment-duration 2]`
- 输出：`master.m3u8` + `init.mp4` + `seg0.m4s`、`seg1.m4s`...
- README 说明分片策略、IDR 对齐原理、断流续传机制、DISCONTINUITY 使用场景
- 附带一个校验脚本，检查每个分片首帧是否为 IDR
