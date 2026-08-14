"""Dump TensorBoard scalar tags from an RLinf run to JSON.

Runs ON PEGASUS inside the RLinf venv (which has tensorboard installed), so the
local side never needs a TB parser. Dumps ALL scalar tags (no guessing tag names);
report.py picks the ones it knows (env/success_once, reward*, loss*, kl*).

Usage (remote):
  python extract_metrics.py <log_dir> <out_json>
"""
import glob
import json
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _event_dir(log_dir):
    """Return the dir actually containing events.out.tfevents* (RLinf writes them
    under <log_dir>/tensorboard/). Falls back to log_dir."""
    hits = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents*"), recursive=True)
    return os.path.dirname(sorted(hits)[-1]) if hits else log_dir


def main():
    log_dir, out_json = sys.argv[1], sys.argv[2]
    log_dir = _event_dir(log_dir)
    print("event dir:", log_dir)
    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    data = {tag: [[s.step, float(s.value)] for s in ea.Scalars(tag)] for tag in tags}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"WROTE {out_json} with {len(tags)} scalar tags")
    for t in tags:
        print("  tag:", t, f"({len(data[t])} pts)")


if __name__ == "__main__":
    main()
