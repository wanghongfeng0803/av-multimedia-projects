#!/usr/bin/env python3
"""视频运动检测命令行入口（detect / mark / compare 三个子命令）。

用法：
    # 1a) 帧差模式检测（默认，旧行为不变）：分块帧差 + 局部自适应阈值
    python motion.py detect 输入.mp4 检测.json
                          [--block 8] [--local-radius 3]
                          [--pixel-threshold 12] [--gain 3.0]
                          [--min-area 64] [--open-radius 1] [--close-radius 1]
                          [--trigger-frames 2] [--release-frames 3]
                          [--illum-change-threshold 25] [--illum-hold-frames 2]

    # 1b) 背景建模模式检测：加 --background，另有预热/学习率/吸收参数
    python motion.py detect 输入.mp4 检测.json --background
                          [--warmup-frames 10] [--learning-rate 0.05]
                          [--absorb-frames 60]

    # 1c) 同输入并排跑两种模式，输出逐帧不一致清单
    python motion.py compare 输入.mp4 对比.json

    # 2) 标记：读检测 JSON 画区域框/状态行出视频，可对同一 JSON 反复重渲染
    python motion.py mark 输入.mp4 检测.json 标记输出.mp4 [--font-scale 1]

检测、标记、事件三模块严格解耦；帧差与背景建模两种模式共享门控/
形态学/滞后机制，差异只在参考模型。区域结构同时供上层复用。
退出码：0 成功（有事件）；1 运行错误；2 成功但全程无运动事件；
compare 只做对比，成功恒为 0。
"""

import sys

from vmotion.cli import main

if __name__ == "__main__":
    sys.exit(main())
