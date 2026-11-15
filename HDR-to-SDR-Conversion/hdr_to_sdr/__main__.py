"""支持 ``python -m hdr_to_sdr`` 方式调用 CLI。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
