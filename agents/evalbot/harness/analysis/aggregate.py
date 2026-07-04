"""Parse closed-loop eval logs and training-state curves.

The per-episode source of truth is ``<tag>_combined_episodes.log`` (the run_manifest
JSON only covers one attempt and isn't written on a crash). Training curves come from
each checkpoint's ``trainer_state.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# "Episode N finished after S steps (Success: B, Terminated: tensor([B]), Truncated: tensor([B]))"
_EP = re.compile(r"Success:\s*(True|False).*?Terminated:.*?\b(True|False)\b.*?Truncated:.*?\b(True|False)\b")


def aggregate(log_path) -> dict:
    """success / terminated(unsafe) / truncated(timeout) counts from one combined log."""
    succ = term = trunc = n = 0
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        m = _EP.search(line)
        if not m:
            continue
        n += 1
        succ += m.group(1) == "True"
        term += m.group(2) == "True"
        trunc += m.group(3) == "True"
    return {"n": n, "success": succ, "terminated": term, "truncated": trunc, "rate": succ / n if n else 0.0}


def count_eps(log_path) -> int:
    p = Path(log_path)
    return aggregate(p)["n"] if p.exists() else 0


def load_curve(trainer_state):
    """(steps, loss, grad_norm) lists from a trainer_state.json (empty if absent)."""
    p = Path(trainer_state)
    if not p.exists():
        return [], [], []
    hist = [h for h in json.loads(p.read_text()).get("log_history", []) if "loss" in h]
    return ([h["step"] for h in hist],
            [h["loss"] for h in hist],
            [h.get("grad_norm") for h in hist])


def final_train_loss(trainer_state):
    _, loss, _ = load_curve(trainer_state)
    return loss[-1] if loss else None
