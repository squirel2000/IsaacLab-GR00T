#!/usr/bin/env python3
"""Entry point for the IsaacLab-GR00T automation pipeline.

The implementation lives in the ``pipeline/`` package; this thin shim puts both the repo
root (for the reused tools pegasus.py / run_finetune.py / wifi_switch.py) and ``pipeline/``
on sys.path, then dispatches to the runner.

    python run_pipeline.py --reset | --resume | --status [--profile n1d5|n1d7]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))   # pipeline_* + automation modules
sys.path.insert(0, str(ROOT))                 # pegasus / run_finetune / wifi_switch / project_paths

from pipeline_runner import main  # noqa: E402  (sys.path set above)

if __name__ == "__main__":
    main()
