#!/usr/bin/env python3
"""Compare GR00T inference runs from their run_manifest.json files.

Each eval writes output/infer_record/<ckpt_id>/<timestamp>/run_manifest.json. This tool
picks the latest *completed* manifest per checkpoint (or a specific one) and prints a
side-by-side of success rate and the failure-mode breakdown (terminated vs truncated),
which distinguishes unsafe behavior (terminated) from not-finishing-in-time (truncated).

Usage:
  python compare_runs.py                      # scan default infer_record root
  python compare_runs.py --root <dir>
  python compare_runs.py <manifest.json> <manifest.json> ...   # explicit files
"""

import argparse
import json
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "output" / "infer_record"


def load(path):
    with open(path) as f:
        return json.load(f)


def latest_completed_manifest(ckpt_dir: Path):
    """Newest manifest with a populated result; fall back to newest of any."""
    manifests = sorted(ckpt_dir.glob("*/run_manifest.json"))
    completed = [m for m in manifests if (load(m).get("result") is not None)]
    pool = completed or manifests
    return pool[-1] if pool else None


def summarize(path):
    d = load(path)
    r = d.get("result") or {}
    return {
        "ckpt": d.get("checkpoint_name"),
        "ver": d.get("gr00t_ver"),
        "eps": r.get("episodes_total"),
        "success": r.get("success"),
        "terminated": r.get("terminated"),
        "truncated": r.get("truncated"),
        "rate": r.get("success_rate"),
        "infer_avg": r.get("inference_time_avg_s"),
        "complete": d.get("result") is not None,
        "path": str(path),
    }


def fmt_row(s):
    rate = f"{100*s['rate']:.1f}%" if s["rate"] is not None else "  n/a"
    eps = s["eps"] if s["eps"] is not None else "?"
    inf = f"{s['infer_avg']:.3f}s" if s["infer_avg"] is not None else "n/a"
    flag = "" if s["complete"] else "  (INCOMPLETE)"
    return (f"  {s['ver']:<5} | rate {rate:>6} | "
            f"succ {str(s['success']):>3} / term {str(s['terminated']):>3} / "
            f"trunc {str(s['truncated']):>3}  (n={eps}) | infer {inf}{flag}")


def main():
    p = argparse.ArgumentParser(description="Compare GR00T inference run manifests.")
    p.add_argument("manifests", nargs="*", help="Explicit run_manifest.json paths.")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="infer_record root to scan.")
    args = p.parse_args()

    if args.manifests:
        paths = [Path(m) for m in args.manifests]
    else:
        paths = []
        for ckpt_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
            m = latest_completed_manifest(ckpt_dir)
            if m:
                paths.append(m)

    if not paths:
        print(f"No run_manifest.json found under {args.root}")
        return

    rows = [summarize(p) for p in paths]
    print("\n=== GR00T run comparison ===")
    for s in rows:
        print(f"\n[{s['ckpt']}]")
        print(fmt_row(s))

    completed = [s for s in rows if s["complete"] and s["rate"] is not None]
    if len(completed) >= 2:
        completed.sort(key=lambda s: s["rate"], reverse=True)
        best, worst = completed[0], completed[-1]
        print("\n--- delta (best vs worst completed) ---")
        print(f"  {best['ver']} {100*best['rate']:.1f}%  vs  {worst['ver']} {100*worst['rate']:.1f}%"
              f"   gap = {100*(best['rate']-worst['rate']):.1f} pts")
        for s in (best, worst):
            n = s["eps"] or 0
            tr = (s["truncated"] or 0) / n * 100 if n else 0
            te = (s["terminated"] or 0) / n * 100 if n else 0
            print(f"  {s['ver']}: truncated(timeouts) {tr:.0f}% | terminated(unsafe) {te:.0f}%")


if __name__ == "__main__":
    main()
