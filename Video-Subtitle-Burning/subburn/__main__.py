"""支持 ``python -m subburn`` 方式调用。"""

from .cli import main

raise SystemExit(main())
