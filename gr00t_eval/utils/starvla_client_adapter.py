"""IsaacLab import shim for the StarVLA WebSocket client adapter.

The implementation lives in the StarVLA checkout so it can evolve with that
project.  This tiny shim lets ``gr00t_infer_agent.py`` import it through the
same local ``utils`` package as the GR00T adapter.
"""

from __future__ import annotations

import sys
from pathlib import Path


# gr00t_eval/utils/starvla_client_adapter.py -> parents[2] == IsaacLab-GR00T repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STARVLA_REPO = PROJECT_ROOT / "starVLA"

if str(STARVLA_REPO) not in sys.path:
    sys.path.insert(0, str(STARVLA_REPO))

from deployment.isaaclab.starvla_client_adapter import (  # noqa: E402,F401
    DEFAULT_ACTION_SPLIT,
    StarVLAClientAdapter,
)

__all__ = ["DEFAULT_ACTION_SPLIT", "StarVLAClientAdapter"]
