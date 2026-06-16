"""Tiny shared progress file the long-running stages write and the dashboard reads.

Stages (download / deploy / train) call :func:`update` with live counters; the dashboard
polls :func:`read`. Single small JSON at ``logs/progress.json``, written atomically so a
reader never sees a half-written file.

    progress.update("download", done=12_000_000_000, total=26_000_000_000,
                    pct=46.1, rate_mbps=88.0, eta_sec=160)
"""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

from pipeline_paths import LOGS_DIR, PROGRESS_PATH


def read(path: Path | None = None) -> dict:
    """Return the current progress dict (empty if absent/unreadable)."""
    p = Path(path) if path else PROGRESS_PATH
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                            # noqa: BLE001 - missing/partial -> empty
        return {}


def update(phase: str, **fields) -> None:
    """Merge ``fields`` into ``progress[phase]`` and mark ``phase`` active. Atomic write."""
    LOGS_DIR.mkdir(exist_ok=True)
    data = read()
    data["phase"] = phase
    data["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    data[phase] = {**data.get(phase, {}), **fields}
    tmp = PROGRESS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, PROGRESS_PATH)               # atomic on Windows + POSIX


def clear() -> None:
    """Remove the progress file (e.g. on --reset)."""
    try:
        PROGRESS_PATH.unlink()
    except FileNotFoundError:
        pass
