#!/usr/bin/env python3
"""Thin bridge: point vla-trainbot's training_monitor at an RLDX-1 training log.

vla-trainbot already solved live training monitoring — `stages/training_monitor.py` parses
tqdm progress plus HF Trainer's ``{'loss': ..., 'grad_norm': ..., 'learning_rate': ...}``
lines into ``metrics.jsonl``, and ``web/dashboard.py`` serves them. RLDX-1 trains through the
same HF Trainer, so its log matches those regexes verbatim — there is nothing to reimplement,
only a path to redirect. Reusing it also keeps the repo's convention that a dashboard belongs
to the owning agent's harness rather than becoming an agent of its own.

This module deliberately does NOT copy training_monitor's regexes; it imports them, so the
two can never drift apart.

Usage:
    python training_bridge.py --log <remote-or-local training log> --out <metrics.jsonl>
    python training_bridge.py --runs            # list known runs as JSON (for the index page)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _find_workspace_root() -> Path:
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        if (d / "workspace.yaml").is_file():
            return d
    return here.parents[4]


ROOT = _find_workspace_root()
VLA_MONITOR = ROOT / "agents" / "vla-trainbot" / "harness" / "stages"
VLA_CORE = ROOT / "agents" / "vla-trainbot" / "harness" / "core"
# training_monitor.py imports `pegasus` and its sibling `pipeline_*` modules at module scope,
# so both directories have to be importable before it can be loaded.
COMMON_TOOLS = ROOT / "agents" / "tools" / "common"

# Runs this agent knows about — pulled from runs.py's RunSpec registry, not duplicated here.
# A second copy of this table (paths included) drifted from runs.py: it still pointed at the
# stale chain/ logs after the real runs moved to chain2/, which is why the right_only tab
# showed as empty. One source of truth now; both this and runs.py's tabs describe the same runs.
def _known_runs() -> list[dict]:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import runs as runs_mod  # noqa: PLC0415

    return [r.as_dict() for r in runs_mod.RUNS]


KNOWN_RUNS = _known_runs()


def _import_monitor():
    """Import vla-trainbot's metric extractor, with its own imports satisfied."""
    for p in (VLA_MONITOR, VLA_CORE, COMMON_TOOLS):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import training_monitor  # noqa: PLC0415  (path must be set up first)

    return training_monitor


def extract(log_text: str, max_steps: int | None = None) -> list[dict]:
    """Parse a training log into metric records using vla-trainbot's own extractor."""
    tm = _import_monitor()
    records, _last_step, _last_total = tm.extract_metrics(log_text, train_total=max_steps)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, help="training log to parse")
    ap.add_argument("--out", type=Path, help="where to write metrics.jsonl")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--runs", action="store_true", help="print KNOWN_RUNS as JSON and exit")
    args = ap.parse_args()

    if args.runs:
        print(json.dumps(KNOWN_RUNS, indent=2))
        return 0

    if not args.log:
        ap.error("--log is required unless --runs is given")
    if not args.log.exists():
        print(f"log not found: {args.log}", file=sys.stderr)
        return 1

    records = extract(args.log.read_text(encoding="utf-8", errors="replace"), args.max_steps)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(records)} records -> {args.out}")
    else:
        print(json.dumps(records[-5:], indent=2))
        print(f"({len(records)} records total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
