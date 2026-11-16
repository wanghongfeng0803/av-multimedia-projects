#!/usr/bin/env python3
"""视频运动检测命令行入口（detect / mark 两个子命令）。

用法：
    # 1) 检测：逐帧分块帧差 + 局部自适应阈值，产出检测 JSON（不出视频）
    python motion.py detect 输入.mp4 检测.json
                          [--block 8] [--local-radius 3]
                          [--pixel-threshold 12] [--gain 3.0]
                          [--min-area 64] [--open-radius 1] [--close-radius 1]
                          [--trigger-frames 2] [--release-frames 3]
                          [--illum-change-threshold 25] [--illum-hold-frames 2]

    # 2) 标记：读检测 JSON 画区域框/状态行出视频，可对同一 JSON 反复重渲染
    python motion.py mark 输入.mp4 检测.json 标记输出.mp4 [--font-scale 1]

检测、标记、事件三模块严格解耦：检测只出运动掩码与区域，标记只做
可视化，事件只消费区域落盘；区域结构同时供后续背景建模复用。
退出码：0 成功（有事件）；1 运行错误；2 成功但全程无运动事件。
"""

import sys

from vmotion.cli import main

if __name__ == "__main__":
    sys.exit(main())
