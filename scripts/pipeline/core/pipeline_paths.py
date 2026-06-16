"""Single source of truth for every path the pipeline touches.

Code lives under ``scripts/pipeline/`` (this module is in ``core/``); the dashboard assets
are in ``web/`` and the config in ``config/``. Generated runtime artifacts (run state, logs,
reports) live at the REPO ROOT so they stay visible and stable. Every other module imports
its paths from here instead of recomputing ``Path(__file__).parent``.
"""
from __future__ import annotations

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent          # .../scripts/pipeline/core
PIPELINE_DIR = PKG_DIR.parent                       # .../scripts/pipeline
REPO_ROOT = PIPELINE_DIR.parents[1]                 # .../IsaacLab-GR00T  (scripts/pipeline -> root)

# --- config + assets travel with the pipeline code; generated artifacts stay at repo root ---
CONFIG_PATH = PIPELINE_DIR / "config" / "config.yaml"   # scripts/pipeline/config/config.yaml (gitignored)
STATE_PATH = REPO_ROOT / "pipeline_state.json"
LOGS_DIR = REPO_ROOT / "logs"
METRICS_PATH = LOGS_DIR / "metrics.jsonl"
PROGRESS_PATH = LOGS_DIR / "progress.json"
REPORT_DIR = REPO_ROOT                              # report_<ts>.html lands here

# --- bundled web assets (travel with the code, under web/) ---
VENDOR_DIR = PIPELINE_DIR / "web" / "vendor"
CHARTJS_PATH = VENDOR_DIR / "chart.umd.min.js"
DASHBOARD_HTML = PIPELINE_DIR / "web" / "dashboard.html"
