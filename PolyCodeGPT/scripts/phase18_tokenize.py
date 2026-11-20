#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polycodegpt.data_v2.tokenizer import main_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main_cli())
