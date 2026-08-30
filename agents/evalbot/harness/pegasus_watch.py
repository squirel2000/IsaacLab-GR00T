#!/usr/bin/env python3
"""Poll a run_eval.py scheduler log on Pegasus until it finishes (or clearly fails), printing
each run's own milestone line once as it appears. Meant to run detached, in the background —
launch it and let it notify once, rather than polling manually.

A full eval sweep can take hours; this stays deliberately minimal (one thing: read a log,
recognise a few markers) so a bug in a richer monitor can't take the detection down too.

Usage:
    python agents/evalbot/harness/pegasus_watch.py --log /data/VLA/tingying/pegasus_runs/evalbot_phase1_comparison/scheduler.log
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _root() -> Path:
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        if (d / "workspace.yaml").is_file():
            return d
    return here.parents[2]


ROOT = _root()
sys.path.insert(0, str(ROOT / "agents" / "tools" / "common"))

import pegasus  # noqa: E402

DONE_MARKER = "comparison chart ->"     # run_eval.py's own last line, printed once, at the end
MILESTONE_TOKENS = ("DONE --", "SERVER NOT READY", "abort")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="scheduler log path on Pegasus")
    ap.add_argument("--interval", type=int, default=900, help="poll interval, seconds")
    args = ap.parse_args()

    s = pegasus.connect()
    fails = 0
    reported: set[str] = set()
    while True:
        try:
            out, _ = pegasus.sh(s, f"cat {args.log} 2>/dev/null", timeout=90)
            if DONE_MARKER in out:
                print(out.strip())
                return 0
            for ln in out.strip().splitlines():
                if any(t in ln for t in MILESTONE_TOKENS) and ln not in reported:
                    print(ln)
                    reported.add(ln)
            fails = 0
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"probe error ({type(e).__name__}), retry {fails}/5")
            if fails >= 5:
                print("giving up after 5 consecutive probe errors — check manually")
                return 2
            try:
                s = pegasus.connect()
            except Exception:  # noqa: BLE001
                pass
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
