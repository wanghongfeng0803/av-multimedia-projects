#!/usr/bin/env python3
"""Repository-local entry point (no installation required)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abr_ladder.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
