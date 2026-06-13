#!/usr/bin/env python3
"""Aggregate + compare closed-loop eval results from the per-episode logs, and
render a success-rate comparison chart.

Each run's outcomes are accumulated by run_eval.py into
``<logdir>/<tag>_combined_episodes.log`` (the crash-resilient source of truth:
the run_manifest JSON only covers a single attempt and isn't written on a crash).
This module reads the run list from eval_config.yaml so the comparison always
matches what run_eval.py produced.

run_eval.py imports ``collect`` / ``print_summary`` / ``write_chart`` and calls
them automatically after an eval. Run this standalone to re-summarise / re-chart
existing logs without launching any eval:

  python scripts/eval/compare_runs.py [--config eval_config.yaml] [--headless]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# "Episode N finished after S steps (Success: B, Terminated: tensor([B]...), Truncated: tensor([B]...))"
_PAT = re.compile(r"Success:\s*(True|False).*?Terminated:.*?\b(True|False)\b.*?Truncated:.*?\b(True|False)\b")

_COLORS = {"success": "#2ca02c", "timeout": "#ff9800", "unsafe": "#d62728"}


def aggregate(path, cap=None) -> dict:
    """success / terminated(unsafe) / truncated(timeout) counts from one combined log."""
    succ = term = trunc = n = 0
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = _PAT.search(line)
        if not m:
            continue
        n += 1
        if cap and n > cap:
            n = cap
            break
        succ += m.group(1) == "True"
        term += m.group(2) == "True"
        trunc += m.group(3) == "True"
    return {"n": n, "success": succ, "terminated": term, "truncated": trunc, "rate": succ / n if n else 0.0}


def count_eps(path: Path) -> int:
    return aggregate(path)["n"] if Path(path).exists() else 0


def collect(runs: list[dict], logdir: Path) -> list[dict]:
    """Aggregate every run that has a combined log under ``logdir``."""
    out = []
    for run in runs:
        log = logdir / f"{run['tag']}_combined_episodes.log"
        if count_eps(log) == 0:
            continue
        out.append({"tag": run["tag"], **aggregate(log)})
    return out


def print_summary(results: list[dict]) -> None:
    if not results:
        print("No eval logs found to summarise.")
        return
    print("\n=== Eval comparison (success / timeout / unsafe) ===")
    print(f"  {'tag':<28} {'success':>9}  {'timeout':>7}  {'unsafe':>6}  {'rate':>5}")
    for r in results:
        print(f"  {r['tag']:<28} {r['success']:>4}/{r['n']:<4} {r['truncated']:>7}  "
              f"{r['terminated']:>6}  {100 * r['rate']:>4.0f}%")


def write_chart(results: list[dict], out_path, title: str = "Closed-loop success rate") -> None:
    """Stacked bar per run (success/timeout/unsafe), height = 100% of n."""
    if not results:
        return
    x0, bw, gap, plot_h, base_y = 90, 60, 50, 230, 290
    width = x0 + len(results) * (bw + gap) + 20
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 380" font-family="sans-serif">',
         f'<text x="{width / 2:.0f}" y="28" text-anchor="middle" font-size="18" font-weight="bold">{title}</text>']
    for pct in (0, 25, 50, 75, 100):
        y = base_y - plot_h * pct / 100
        s.append(f'<line x1="{x0 - 10}" y1="{y:.1f}" x2="{width - 20}" y2="{y:.1f}" stroke="#e0e0e0"/>')
        s.append(f'<text x="{x0 - 15}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#666">{pct}%</text>')
    for i, r in enumerate(results):
        x = x0 + i * (bw + gap)
        n = r["n"] or 1
        y = base_y
        for name, val in (("success", r["success"]), ("timeout", r["truncated"]), ("unsafe", r["terminated"])):
            h = plot_h * val / n
            y -= h
            if h > 0.5:
                s.append(f'<rect x="{x}" y="{y:.1f}" width="{bw}" height="{h:.1f}" fill="{_COLORS[name]}"/>')
        cx = x + bw / 2
        s.append(f'<text x="{cx:.0f}" y="{base_y - plot_h - 8}" text-anchor="middle" font-size="14" font-weight="bold">{100 * r["rate"]:.0f}%</text>')
        s.append(f'<text x="{cx:.0f}" y="{base_y + 16}" text-anchor="end" font-size="11" '
                 f'transform="rotate(-30 {cx:.0f} {base_y + 16})">{r["tag"]} (n={r["n"]})</text>')
    lx = x0
    for name, c in _COLORS.items():
        s.append(f'<rect x="{lx}" y="362" width="11" height="11" fill="{c}"/>')
        s.append(f'<text x="{lx + 15}" y="372" font-size="11">{name}</text>')
        lx += 95
    s.append("</svg>")
    Path(out_path).write_text("\n".join(s))


def _logdir(spec: dict, headless: bool) -> Path:
    base = spec["defaults"]["logdir"] + ("_headless" if headless else "")
    p = Path(base)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    p = argparse.ArgumentParser(description="Summarise + chart existing eval logs (no eval run).")
    p.add_argument("--config", type=Path, default=HERE / "eval_config.yaml", help="eval_config.yaml path.")
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, help="Read the headless logdir.")
    args = p.parse_args()
    spec = yaml.safe_load(args.config.read_text())
    headless = args.headless if args.headless is not None else spec["defaults"]["headless"]
    logdir = _logdir(spec, headless)
    results = collect(spec["runs"], logdir)
    print_summary(results)
    chart = ROOT / "output" / "analysis" / f"eval_results{'_headless' if headless else ''}.svg"
    chart.parent.mkdir(parents=True, exist_ok=True)
    write_chart(results, chart)
    print(f"\nchart -> {chart}")


if __name__ == "__main__":
    main()
