"""Aggregate eval runs into comparison rows, print a table, and render the figure.

Imported by run_eval.py after an eval (``report`` is the one call it makes). Pure
analysis — takes resolved (tag, log, checkpoint) paths, does no eval and no launching.
"""

from __future__ import annotations

from pathlib import Path

from .aggregate import aggregate, final_train_loss, load_curve
from .make_loss_svg import render

# Per-run line colours for the loss / grad curves.
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def build_rows(items):
    """items: [(tag, combined_log_path, ckpt_dir)] -> comparison rows."""
    rows = []
    for i, (tag, log, ckpt) in enumerate(items):
        rows.append({"tag": tag, **aggregate(log), "ckpt": Path(ckpt),
                     "loss": final_train_loss(Path(ckpt) / "trainer_state.json"),
                     "color": PALETTE[i % len(PALETTE)]})
    return rows


def print_summary(rows):
    if not rows:
        print("No eval logs to summarise.")
        return
    print("\n=== Eval comparison ===")
    print(f"  {'tag':<28} {'success':>9}  {'timeout':>7}  {'unsafe':>6}  {'rate':>5}  {'loss':>8}")
    for r in rows:
        loss = f"{r['loss']:.4f}" if r["loss"] is not None else "—"
        print(f"  {r['tag']:<28} {r['success']:>4}/{r['n']:<4} {r['truncated']:>7}  "
              f"{r['terminated']:>6}  {100 * r['rate']:>4.0f}%  {loss:>8}")


def report(items, chart_path):
    """Print the comparison table and write the comparison figure. Returns the rows."""
    rows = build_rows(items)
    print_summary(rows)
    curves = {r["tag"]: load_curve(r["ckpt"] / "trainer_state.json") for r in rows}
    render(rows, curves, Path(chart_path))
    return rows
