"""Configuration loading + profile resolution for the automation pipeline.

Reads ``config.yaml`` (see ``config.example.yaml``), deep-merges it over built-in
defaults, resolves the active training profile (N1.5 conda / N1.7 uv), and builds the
concrete training command by substituting the detected GPU id and paths into the
profile's ``train_cmd_template``.

Pure helpers (``deep_merge`` / ``resolve_profile`` / ``build_train_cmd``) take plain
dicts so they can be unit-tested without touching the filesystem.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "config.yaml"
DEFAULT_PROFILE = "n1d7"

# Built-in fallbacks for every non-secret, non-path knob. A user config.yaml only needs
# to specify what differs; anything omitted falls back to these.
DEFAULTS: dict = {
    "pegasus": {"state_root": "/data/VLA/tingying/pegasus_runs"},
    "training": {
        "gpu_idle_threshold_util": 10,
        "gpu_idle_threshold_mem_gb": 5.0,
        "gpu_poll_interval_min": 30,
        "monitor_interval_sec": 300,
        "connect_retry_max": 3,
        "connect_retry_interval_sec": 10,
    },
    "asus4090": {"retry_max": 3},
    "wifi": {
        "external": "cj",
        "local": "omap",
        "internet_check_ip": "8.8.8.8",
        "net_ready_timeout_sec": 60,
    },
}

_PLACEHOLDERS = ("gpu", "dataset_path", "output_dir", "max_steps")


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: ``override`` layered over ``base``, recursing into sub-dicts."""
    out = dict(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str | os.PathLike[str] | None = None) -> dict:
    """Load config.yaml and deep-merge it over :data:`DEFAULTS`."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise SystemExit(
            f"config not found: {path}\n"
            f"copy config.example.yaml -> config.yaml and edit it."
        )
    user = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return deep_merge(DEFAULTS, user)


def resolve_profile(config: dict, cli_profile: str | None = None) -> tuple[str, dict]:
    """Pick the active profile: CLI flag > config['active_profile'] > 'n1d7'.

    Returns ``(name, profile_dict)`` and fails loudly if the name is unknown.
    """
    profiles = config.get("profiles") or {}
    name = cli_profile or config.get("active_profile") or DEFAULT_PROFILE
    if name not in profiles:
        raise SystemExit(
            f"profile '{name}' not in config.profiles "
            f"(have: {', '.join(sorted(profiles)) or 'none'})"
        )
    return name, profiles[name]


def build_train_cmd(profile: dict, gpu_id: int) -> str:
    """Substitute {gpu}/{dataset_path}/{output_dir}/{max_steps} into the profile template.

    Collapses the YAML folded-scalar whitespace into a single clean command line, then
    replaces each placeholder by literal string substitution (so other shell braces, if
    any, are never touched).
    """
    template = profile.get("train_cmd_template")
    if not template:
        raise SystemExit("profile has no 'train_cmd_template'")
    values = {
        "gpu": str(gpu_id),
        "dataset_path": str(profile["dataset_path"]),
        "output_dir": str(profile["output_dir"]),
        "max_steps": str(profile["max_steps"]),
    }
    cmd = " ".join(str(template).split())  # collapse newlines/extra spaces from `>` block
    for key in _PLACEHOLDERS:
        cmd = cmd.replace("{" + key + "}", values[key])
    return cmd


def get_secret(env_name: str, fallback: str | None = None) -> str | None:
    """Return a secret from the environment, falling back to the config value."""
    return os.environ.get(env_name) or fallback
