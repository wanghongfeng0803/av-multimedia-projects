"""便捷入口：python burn.py video.mp4 subs.srt -o out.mp4

未显式给出 ``burn`` 子命令时自动补上，使最常见的调用可以省略子命令。
"""

import sys

from subburn.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] not in ("burn", "-h", "--help", "--version"):
        argv = ["burn", *argv]
    raise SystemExit(main(argv))
