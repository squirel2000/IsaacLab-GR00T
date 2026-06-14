"""Single source of truth for every path the pipeline touches.

Code lives in this ``scripts/pipeline/`` directory alongside ``config.yaml``; generated
runtime artifacts (run state, logs, reports) live at the REPO ROOT so they stay visible
and stable. Every other module imports its paths from here instead of recomputing
``Path(__file__).parent``.
"""
from __future__ import annotations

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent          # .../IsaacLab-GR00T/scripts/pipeline
REPO_ROOT = PKG_DIR.parents[1]                       # .../IsaacLab-GR00T  (scripts/pipeline -> root)

# --- config travels with the pipeline code; generated artifacts stay at the repo root ---
CONFIG_PATH = PKG_DIR / "config.yaml"               # scripts/pipeline/config.yaml (gitignored)
STATE_PATH = REPO_ROOT / "pipeline_state.json"
LOGS_DIR = REPO_ROOT / "logs"
METRICS_PATH = LOGS_DIR / "metrics.jsonl"
PROGRESS_PATH = LOGS_DIR / "progress.json"
REPORT_DIR = REPO_ROOT                              # report_<ts>.html lands here

# --- bundled assets (travel with the code) ---
VENDOR_DIR = PKG_DIR / "vendor"
CHARTJS_PATH = VENDOR_DIR / "chart.umd.min.js"
DASHBOARD_HTML = PKG_DIR / "dashboard.html"
