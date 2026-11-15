"""``python compose.py 左视频 右视频 ...`` 的便捷入口（等价于 python -m compose_tool）。"""

from compose_tool.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
