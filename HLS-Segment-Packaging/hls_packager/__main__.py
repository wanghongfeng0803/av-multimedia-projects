"""支持 ``python -m hls_packager`` 方式调用。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
