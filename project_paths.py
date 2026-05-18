"""Shared path helpers for the IsaacLab-GR00T workspace.

The root workspace owns project-level VLA datasets and fine-tuned checkpoints.
Individual upstream repositories can keep their own examples, assets, logs,
and package-local defaults.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT_ENV = "ISAACLAB_GR00T_ROOT"
DATASETS_ENV = "VLA_DATASETS_ROOT"
CHECKPOINTS_ENV = "VLA_CHECKPOINTS_ROOT"


def project_root(start: Path | None = None) -> Path:
    """Return the workspace root, honoring ISAACLAB_GR00T_ROOT if set."""
    env_root = os.environ.get(ROOT_ENV)
    if env_root:
        return Path(os.path.expandvars(env_root)).expanduser().resolve()

    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent

    for path in (cur, *cur.parents):
        if (path / "Isaac-GR00T").exists() and (path / "IsaacLab").exists() and (path / "starVLA").exists():
            return path

    return Path(__file__).resolve().parent


def datasets_root(root: Path | None = None) -> Path:
    """Return the root directory for shared project fine-tuning datasets."""
    env_root = os.environ.get(DATASETS_ENV)
    if env_root:
        return Path(os.path.expandvars(env_root)).expanduser().resolve()
    return (root or project_root()) / "datasets"


def checkpoints_root(root: Path | None = None) -> Path:
    """Return the root directory for shared project fine-tuned checkpoints."""
    env_root = os.environ.get(CHECKPOINTS_ENV)
    if env_root:
        return Path(os.path.expandvars(env_root)).expanduser().resolve()
    return (root or project_root()) / "artifacts" / "checkpoints"


def resolve_project_path(value: str | os.PathLike[str], root: Path | None = None) -> Path:
    """Resolve absolute, env-var, user, or workspace-relative paths."""
    path = Path(os.path.expandvars(os.fspath(value))).expanduser()
    if path.is_absolute():
        return path
    return (root or project_root()) / path


def dataset_path(name: str, root: Path | None = None) -> Path:
    """Return a shared dataset path by name."""
    return datasets_root(root) / name


def checkpoint_path(model: str, *parts: str, root: Path | None = None) -> Path:
    """Return a shared checkpoint path under artifacts/checkpoints/<model>/."""
    return checkpoints_root(root) / model / Path(*parts)
