"""支持 ``python -m bvp`` 调用命令行入口。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
