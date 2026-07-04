"""IsaacLab import shim for the StarVLA WebSocket client adapter.

The implementation lives in the StarVLA checkout so it can evolve with that
project.  This tiny shim lets ``gr00t_infer_agent.py`` import it through the
same local ``utils`` package as the GR00T adapter.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_workspace_root() -> Path:
    """Walk up until the workspace.yaml root marker (survives directory moves)."""
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        if (d / "workspace.yaml").is_file():
            return d
    return here.parents[2]                    # legacy fallback (scripts/eval/utils -> root)


def _ws_get(root: Path, key: str, default: str) -> str:
    """Read one flat `key: value` line from workspace.yaml (no yaml dependency)."""
    marker = root / "workspace.yaml"
    if marker.is_file():
        for line in marker.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{key}:"):
                return line.split(":", 1)[1].strip()
    return default


PROJECT_ROOT = _find_workspace_root()
STARVLA_REPO = PROJECT_ROOT / _ws_get(PROJECT_ROOT, "starvla", "starVLA")

if str(STARVLA_REPO) not in sys.path:
    sys.path.insert(0, str(STARVLA_REPO))

from deployment.isaaclab.starvla_client_adapter import (  # noqa: E402,F401
    DEFAULT_ACTION_SPLIT,
    StarVLAClientAdapter,
)

__all__ = ["DEFAULT_ACTION_SPLIT", "StarVLAClientAdapter"]
