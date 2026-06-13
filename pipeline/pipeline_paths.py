"""Single source of truth for every path the pipeline touches.

Code lives in this ``pipeline/`` directory; runtime artifacts (the user-edited config,
run state, logs, reports) live at the REPO ROOT so they stay visible and stable. Every
other module imports its paths from here instead of recomputing ``Path(__file__).parent``.
"""
from __future__ import annotations

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent          # .../IsaacLab-GR00T/pipeline
REPO_ROOT = PKG_DIR.parent                          # .../IsaacLab-GR00T

# --- user/runtime artifacts (repo root) ---
CONFIG_PATH = REPO_ROOT / "config.yaml"
STATE_PATH = REPO_ROOT / "pipeline_state.json"
LOGS_DIR = REPO_ROOT / "logs"
METRICS_PATH = LOGS_DIR / "metrics.jsonl"
PROGRESS_PATH = LOGS_DIR / "progress.json"
REPORT_DIR = REPO_ROOT                              # report_<ts>.html lands here

# --- bundled assets (travel with the code) ---
VENDOR_DIR = PKG_DIR / "vendor"
CHARTJS_PATH = VENDOR_DIR / "chart.umd.min.js"
DASHBOARD_HTML = PKG_DIR / "dashboard.html"
