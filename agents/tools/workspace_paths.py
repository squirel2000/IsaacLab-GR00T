"""Resolve workspace-relative paths through the root workspace.yaml.

The monorepo root is discovered by walking up from a start directory until a
``workspace.yaml`` is found (that file is the root marker). Keys map to
directories that may move during the agents/engines restructure — consumers
resolve through here (or through ``paths.env`` in bash) instead of hardcoding.

No dependencies: workspace.yaml is a flat ``key: value`` file parsed with plain
string handling, so this works in every env (uv, conda env_isaaclab, system).
"""
from __future__ import annotations

import os
from pathlib import Path

_MARKER = "workspace.yaml"


def find_workspace_root(start: str | os.PathLike | None = None) -> Path:
    """Walk up from *start* (default: this file) until workspace.yaml is found."""
    env = os.environ.get("WS_ROOT")
    if env and (Path(env) / _MARKER).is_file():
        return Path(env).resolve()
    base = Path(start).resolve() if start else Path(__file__).resolve().parent
    for d in (base, *base.parents):
        if (d / _MARKER).is_file():
            return d
    raise FileNotFoundError(f"{_MARKER} not found walking up from {base}")


def load_map(root: Path | None = None) -> dict[str, str]:
    root = root or find_workspace_root()
    out: dict[str, str] = {}
    for line in (root / _MARKER).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def ws_path(key: str, start: str | os.PathLike | None = None) -> Path:
    """Absolute path for a workspace.yaml *key* (e.g. ws_path('eval_harness'))."""
    root = find_workspace_root(start)
    mapping = load_map(root)
    if key not in mapping:
        raise KeyError(f"'{key}' not in {root / _MARKER} (known: {sorted(mapping)})")
    return (root / mapping[key]).resolve()
