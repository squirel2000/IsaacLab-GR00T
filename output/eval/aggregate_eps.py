#!/usr/bin/env python3
"""Aggregate success/terminated/truncated from combined per-episode log lines.

The crash-resilient orchestrator concatenates each client attempt's
"Episode N finished after S steps (Success: B, Terminated: tensor([B]...), Truncated: tensor([B]...))"
lines into <tag>_combined_episodes.log. The run_manifest JSON only covers a single
attempt and isn't written on crash, so these lines are the source of truth.

Usage: aggregate_eps.py <combined_episodes.log> [cap]   # cap = max episodes to count
"""
import re
import sys

PAT = re.compile(
    r"Success:\s*(True|False).*?Terminated:.*?\b(True|False)\b.*?Truncated:.*?\b(True|False)\b"
)


def aggregate(path, cap=None):
    succ = term = trunc = n = 0
    with open(path) as f:
        for line in f:
            m = PAT.search(line)
            if not m:
                continue
            n += 1
            if cap and n > cap:
                n = cap
                break
            succ += m.group(1) == "True"
            term += m.group(2) == "True"
            trunc += m.group(3) == "True"
    rate = succ / n if n else 0.0
    return {"n": n, "success": succ, "terminated": term, "truncated": trunc, "rate": rate}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else None
    s = aggregate(sys.argv[1], cap)
    print(f"{sys.argv[1]}")
    print(f"  n={s['n']}  success={s['success']}  terminated={s['terminated']}  truncated={s['truncated']}")
    print(f"  success_rate={100*s['rate']:.1f}%  "
          f"(timeouts {100*s['truncated']/s['n'] if s['n'] else 0:.0f}% / "
          f"unsafe {100*s['terminated']/s['n'] if s['n'] else 0:.0f}%)")


if __name__ == "__main__":
    main()
