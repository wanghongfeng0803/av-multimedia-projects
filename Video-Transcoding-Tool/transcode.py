"""便捷入口：python transcode.py input.mp4 -o output.mp4"""

from transcode_tool.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
