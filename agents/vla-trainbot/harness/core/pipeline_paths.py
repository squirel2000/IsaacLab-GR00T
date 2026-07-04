"""Single source of truth for every path the pipeline touches.

Code lives under ``agents/vla-trainbot/harness/`` (this module is in ``core/``); the dashboard
assets are in ``web/`` and the config in ``config/``. Generated runtime artifacts (run state,
logs, reports) live in the owning agent's ``var/`` (gitignored) so the workspace root stays
clean. Every other module imports its paths from here instead of recomputing
``Path(__file__).parent``.
"""
from __future__ import annotations

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent          # .../harness/core
PIPELINE_DIR = PKG_DIR.parent                       # .../harness


def _find_workspace_root() -> Path:
    """Walk up until the workspace.yaml root marker (survives directory moves)."""
    for d in (PIPELINE_DIR, *PIPELINE_DIR.parents):
        if (d / "workspace.yaml").is_file():
            return d
    return PIPELINE_DIR.parents[1]                  # legacy fallback (scripts/pipeline -> root)


REPO_ROOT = _find_workspace_root()                  # .../IsaacLab-GR00T

# --- config + assets travel with the harness code; runtime artifacts go to the agent var/ ---
CONFIG_PATH = PIPELINE_DIR / "config" / "config.yaml"   # harness/config/config.yaml (gitignored)
VAR_DIR = PIPELINE_DIR.parent / "var"               # agents/vla-trainbot/var (gitignored)
STATE_PATH = VAR_DIR / "pipeline_state.json"
LOGS_DIR = VAR_DIR / "logs"
METRICS_PATH = LOGS_DIR / "metrics.jsonl"
PROGRESS_PATH = LOGS_DIR / "progress.json"
REPORT_DIR = VAR_DIR                                # report_<ts>.html lands here

# --- bundled web assets (travel with the code, under web/) ---
VENDOR_DIR = PIPELINE_DIR / "web" / "vendor"
CHARTJS_PATH = VENDOR_DIR / "chart.umd.min.js"
DASHBOARD_HTML = PIPELINE_DIR / "web" / "dashboard.html"
