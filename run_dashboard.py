#!/usr/bin/env python3
"""Entry point for the read-only live pipeline dashboard (implementation in pipeline/).

    python run_dashboard.py [--host 0.0.0.0] [--port 8770]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))

from dashboard import main  # noqa: E402  (sys.path set above)

if __name__ == "__main__":
    main()
