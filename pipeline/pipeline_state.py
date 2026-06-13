"""Persistent state machine for the automation pipeline.

Backed by ``pipeline_state.json``. Records the current stage plus, per stage, its
``status`` / timestamps / ``attempts`` / last ``error``, and a shared ``data`` dict for
values that flow between stages (gpu_id, runid, local zip path, remote deploy path, ...).

On restart, :meth:`PipelineState.load` reads the file and the runner resumes from
``state.current`` — completed stages are skipped, the interrupted stage is retried.

    state = PipelineState.load()
    while state.current is not Stage.DONE:
        stage = state.current
        state.begin(stage)
        ...do the work...            # may call state.record_output(...)
        state.complete(stage)        # marks done + advances `current`
"""
from __future__ import annotations

import datetime
import json
from enum import Enum
from pathlib import Path

from pipeline_paths import STATE_PATH as DEFAULT_STATE_PATH


class Stage(str, Enum):
    IDLE = "IDLE"
    GPU_WAIT = "GPU_WAIT"
    TRAINING = "TRAINING"
    DOWNLOADING = "DOWNLOADING"
    DEPLOYING = "DEPLOYING"
    REPORTING = "REPORTING"
    DONE = "DONE"


# Linear order the pipeline walks. (SIMULATING is intentionally omitted — run manually.)
ORDER: list[Stage] = [
    Stage.IDLE, Stage.GPU_WAIT, Stage.TRAINING, Stage.DOWNLOADING,
    Stage.DEPLOYING, Stage.REPORTING, Stage.DONE,
]


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def next_stage(stage: Stage) -> Stage:
    """Return the stage after ``stage`` (DONE maps to itself)."""
    i = ORDER.index(stage)
    return ORDER[min(i + 1, len(ORDER) - 1)]


class PipelineState:
    def __init__(self, path: Path, current: Stage = Stage.IDLE,
                 profile: str | None = None, stages: dict | None = None,
                 data: dict | None = None, history: list | None = None):
        self.path = Path(path)
        self.current = Stage(current)
        self.profile = profile
        self.stages: dict[str, dict] = stages or {}
        self.data: dict = data or {}
        self.history: list[dict] = history or []

    # ----- persistence ------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path = DEFAULT_STATE_PATH) -> "PipelineState":
        """Load existing state, or return a fresh IDLE machine if the file is absent."""
        path = Path(path)
        if not path.exists():
            return cls(path)
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(path, current=d.get("current", Stage.IDLE.value),
                   profile=d.get("profile"), stages=d.get("stages", {}),
                   data=d.get("data", {}), history=d.get("history", []))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "current": self.current.value,
            "profile": self.profile,
            "updated_at": _now(),
            "stages": self.stages,
            "data": self.data,
            "history": self.history,
        }, indent=2), encoding="utf-8")

    # ----- per-stage record helpers ----------------------------------------
    def _rec(self, stage: Stage) -> dict:
        return self.stages.setdefault(stage.value, {"status": "pending", "attempts": 0})

    def attempts(self, stage: Stage) -> int:
        return self._rec(stage).get("attempts", 0)

    def status(self, stage: Stage) -> str:
        return self._rec(stage).get("status", "pending")

    def begin(self, stage: Stage) -> None:
        """Mark ``stage`` in_progress, stamp the start time, bump the attempt counter."""
        rec = self._rec(stage)
        rec["status"] = "in_progress"
        rec["attempts"] = rec.get("attempts", 0) + 1
        rec["started_at"] = _now()
        rec.pop("error", None)
        self.current = stage
        self.history.append({"stage": stage.value, "event": "begin",
                             "attempt": rec["attempts"], "at": rec["started_at"]})
        self.save()

    def complete(self, stage: Stage) -> Stage:
        """Mark ``stage`` done and advance ``current`` to the next stage. Returns it."""
        rec = self._rec(stage)
        rec["status"] = "done"
        rec["finished_at"] = _now()
        self.current = next_stage(stage)
        self.history.append({"stage": stage.value, "event": "done", "at": rec["finished_at"]})
        self.save()
        return self.current

    def fail(self, stage: Stage, error: str) -> None:
        """Mark ``stage`` failed and record the error (``current`` stays put for retry)."""
        rec = self._rec(stage)
        rec["status"] = "failed"
        rec["error"] = str(error)
        rec["failed_at"] = _now()
        self.history.append({"stage": stage.value, "event": "failed",
                             "error": str(error), "at": rec["failed_at"]})
        self.save()

    # ----- shared cross-stage data -----------------------------------------
    def record_output(self, key: str, value) -> None:
        self.data[key] = value
        self.save()

    def record_outputs(self, **kwargs) -> None:
        """Set several shared outputs and persist once (one write, not one per key)."""
        self.data.update(kwargs)
        self.save()

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set_profile(self, name: str) -> None:
        self.profile = name
        self.save()

    # ----- reset ------------------------------------------------------------
    def reset(self) -> None:
        """Wipe progress back to IDLE (keeps the file for the next run)."""
        self.current = Stage.IDLE
        self.stages = {}
        self.data = {}
        self.history = [{"event": "reset", "at": _now()}]
        self.save()
