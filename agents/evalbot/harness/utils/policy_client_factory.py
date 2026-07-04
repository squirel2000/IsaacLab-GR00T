"""Policy-client factory for IsaacLab rollout scripts.

The rollout loop should only know that a policy client exposes
``get_action(obs)``.  Backend-specific transport, normalization, and import
details live in adapters selected here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Workspace root that hosts shared datasets / checkpoints — walk up until the
# workspace.yaml root marker (survives directory moves).
# ---------------------------------------------------------------------------
def _find_workspace_root() -> Path:
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        if (d / "workspace.yaml").is_file():
            return d
    return here.parents[2]                    # legacy fallback (<harness>/utils -> root)


PROJECT_ROOT = _find_workspace_root()

SCRIPT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_CONFIGS = {
    "starvla": SCRIPT_DIR / "configs" / "starvla_openarm_o6.json",
    "gr00t": SCRIPT_DIR / "configs" / "gr00t_n15_openarm_o6.json",
}


def _resolve_path(value: str | os.PathLike[str]) -> Path:
    """Expand env vars / `~` and resolve relative paths against PROJECT_ROOT."""
    path = Path(os.path.expandvars(os.fspath(value))).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_policy_config(policy: str, config_path: str | None) -> dict[str, Any]:
    path = _resolve_path(config_path) if config_path else DEFAULT_POLICY_CONFIGS.get(policy)
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Policy config not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                f"Reading YAML policy configs requires PyYAML. Either install it or "
                f"convert {path} to JSON."
            ) from e
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    raise ValueError(f"Unsupported policy config format: {path}")


def _override(value: Any, fallback: Any) -> Any:
    return fallback if value is None or value == "" else value


def build_policy_client(args_cli):
    """Build the selected policy client from generic rollout CLI args."""
    policy = str(args_cli.policy).lower()
    config = _load_policy_config(policy, getattr(args_cli, "policy_config", None))

    if policy == "gr00t":
        from utils.gr00t_client_adapter import Gr00tClientAdapter

        version = config.get("version", args_cli.gr00t_ver)
        host = _override(args_cli.host, config.get("host", "localhost"))
        port = int(_override(args_cli.port, config.get("port", 5555)))
        return Gr00tClientAdapter(version=version, host=host, port=port)

    if policy == "starvla":
        from utils.starvla_client_adapter import StarVLAClientAdapter

        host = _override(args_cli.host, config.get("host", "127.0.0.1"))
        port = int(_override(args_cli.port, config.get("port", 10093)))
        action_split = [tuple(item) for item in config.get("action_split", [["right_arm", 7], ["right_hand", 6]])]

        return StarVLAClientAdapter(
            host=host,
            port=port,
            stats_path=_resolve_path(config["stats_path"]),
            embodiment_key=config.get("embodiment_key"),
            action_split=action_split,
            starvla_repo=_resolve_path(config["starvla_repo"]),
        )

    raise ValueError(
        f"Unsupported policy '{args_cli.policy}'. Add an adapter and register it in "
        "utils/policy_client_factory.py."
    )
