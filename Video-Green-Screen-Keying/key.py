#!/usr/bin/env python3
"""色度键抠像命令行入口（键控 / alpha 后处理 / 输出编码 三层）。

用法：
    python key.py 输入 -o 输出
    python key.py green.mp4 -o out.mov                      # 采样键色，透明 mov
    python key.py green.mp4 -o out.mp4 --bg-color 0,0,0     # 背景替换出 mp4
    python key.py blue.mp4 -o out.mov --key 0,70,255 --feather 2 --choke 1
    python key.py green.mp4 -o "seq/f_%06d.png"             # 带 alpha 的 PNG 序列

键色默认从画面顶部条带采样（不写死绿色），蓝幕/红幕/任意纯色幕
同理可用；处理接在 ffmpeg 解码与编码之间，音轨原样直通。
"""

import sys

from vkeyer.cli import main


if __name__ == "__main__":
    sys.exit(main())
