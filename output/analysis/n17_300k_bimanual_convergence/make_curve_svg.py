#!/usr/bin/env python3
"""Plot train-loss (log) + grad_norm from an HF trainer_state.json, no wandb needed.

The trainer_state.json bundled in every checkpoint already contains the full
log_history (loss / grad_norm / lr per logging step), so the training curves can
be rebuilt locally even when the original wandb run lives on another account.

Usage:
    python make_curve_svg.py /path/to/checkpoint-XXXX/trainer_state.json [out.svg]
"""
import json
import math
import sys
from pathlib import Path

W, H = 1100, 720
PADL, PADR, PADT, PADB = 90, 30, 50, 60
PANEL_GAP = 70
PANEL_H = (H - PADT - PADB - PANEL_GAP) // 2


def _x(step, smax):
    return PADL + (W - PADL - PADR) * step / smax


def _y(val, vmin, vmax, top, h, log=False):
    if log:
        val = math.log10(max(val, 1e-9))
        vmin = math.log10(max(vmin, 1e-9))
        vmax = math.log10(max(vmax, 1e-9))
    return top + h - h * (val - vmin) / (vmax - vmin + 1e-12)


def _poly(xs, ys):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def main():
    state_path = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else state_path.parent / "train_curve.svg"
    d = json.loads(state_path.read_text())
    Hh = [e for e in d["log_history"] if "loss" in e]
    steps = [e["step"] for e in Hh]
    loss = [e["loss"] for e in Hh]
    gn = [e.get("grad_norm", float("nan")) for e in Hh]
    smax = max(steps)

    # loss panel (log). Clip the warmup spike for readability of the tail.
    lo_top = PADT
    lmin = min(v for v in loss if v > 0)
    lmax = max(loss)
    lx = [_x(s, smax) for s in steps]
    ly = [_y(v, lmin, lmax, lo_top, PANEL_H, log=True) for v in loss]

    # grad_norm panel (linear)
    gn_top = PADT + PANEL_H + PANEL_GAP
    gvals = [g for g in gn if g == g]
    gmin, gmax = 0.0, max(gvals)
    gx = [_x(s, smax) for s in steps]
    gy = [_y(g if g == g else 0, gmin, gmax, gn_top, PANEL_H) for g in gn]

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'font-family="sans-serif" font-size="13">']
    svg.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    svg.append(f'<text x="{W/2}" y="24" text-anchor="middle" font-size="17" '
               f'font-weight="bold">N1.7 FFT 300K (bimanual) — train loss &amp; grad_norm '
               f'(from trainer_state.json)</text>')

    # axes + grid for both panels
    for top, label, vmin, vmax, log in [
        (lo_top, "train loss (log10)", lmin, lmax, True),
        (gn_top, "grad_norm", gmin, gmax, False),
    ]:
        svg.append(f'<rect x="{PADL}" y="{top}" width="{W-PADL-PADR}" height="{PANEL_H}" '
                   f'fill="none" stroke="#ccc"/>')
        svg.append(f'<text x="20" y="{top+PANEL_H/2}" text-anchor="middle" '
                   f'transform="rotate(-90 20 {top+PANEL_H/2})">{label}</text>')
        # y ticks (5)
        for k in range(5):
            frac = k / 4
            yv = top + PANEL_H - PANEL_H * frac
            if log:
                val = 10 ** (math.log10(vmin) + (math.log10(vmax) - math.log10(vmin)) * frac)
                txt = f"{val:.4f}"
            else:
                val = vmin + (vmax - vmin) * frac
                txt = f"{val:.2f}"
            svg.append(f'<line x1="{PADL}" y1="{yv:.1f}" x2="{W-PADR}" y2="{yv:.1f}" '
                       f'stroke="#eee"/>')
            svg.append(f'<text x="{PADL-6}" y="{yv+4:.1f}" text-anchor="end" '
                       f'fill="#555">{txt}</text>')
        # x ticks
        for k in range(7):
            sv = smax * k / 6
            xv = _x(sv, smax)
            svg.append(f'<line x1="{xv:.1f}" y1="{top}" x2="{xv:.1f}" y2="{top+PANEL_H}" '
                       f'stroke="#f3f3f3"/>')
            svg.append(f'<text x="{xv:.1f}" y="{top+PANEL_H+18}" text-anchor="middle" '
                       f'fill="#555">{int(sv/1000)}k</text>')

    svg.append(f'<polyline fill="none" stroke="#1f77b4" stroke-width="1.4" '
               f'points="{_poly(lx, ly)}"/>')
    svg.append(f'<polyline fill="none" stroke="#d62728" stroke-width="1.2" '
               f'points="{_poly(gx, gy)}"/>')
    svg.append(f'<text x="{W/2}" y="{H-18}" text-anchor="middle" fill="#555">'
               f'training step (final loss={loss[-1]:.4f} @ {steps[-1]}; '
               f'NOTE: train velocity-field loss only, NOT a held-out / overfitting signal)</text>')
    svg.append("</svg>")
    out.write_text("\n".join(svg))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
