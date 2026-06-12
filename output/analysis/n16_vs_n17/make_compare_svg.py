#!/usr/bin/env python3
"""Render N1.6 vs N1.7 fine-tune comparison as a 3-panel SVG.

Panels share the X axis (training step) so you can read vertically and spot
overfitting: the point where *train loss keeps dropping* while the *held-out
open-loop action MSE turns back up*.

  Panel 1  Training loss          (log Y)   <- velocity-field MSE on train data
  Panel 2  Held-out open-loop MSE (log Y)   <- action-space MSE on unseen trajs
  Panel 3  Gradient norm          (linear)

Panel 2 reads one CSV per run (`step,mse,mae`, the format emitted by
output/eval_compare_logs/openloop_mse_sweep.sh). If a CSV is missing the panel
draws a placeholder telling you how to populate it. Pure-stdlib, no matplotlib.

Usage:
    make_compare_svg.py [out.svg]
    # CSVs are looked up next to this script:
    #   n16_openloop_mse.csv   n17_openloop_mse.csv
"""
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/asus/Gits/IsaacLab-GR00T/artifacts/checkpoints/gr00t"
RUNS = {
    "N1.6-150k": {
        "trainer_state": (f"{ROOT}/openarm_linkerhando6_multitask_N16_150k_dataset_0403_no_tune_visual"
                          "/checkpoint-150000/trainer_state.json"),
        "openloop_csv": os.path.join(HERE, "n16_openloop_mse.csv"),
        "color": "#1f77b4",
    },
    "N1.7-150k": {
        "trainer_state": (f"{ROOT}/openarm_linkerhando6_multitask_N17_150k_dataset_0403_no_tune_visual"
                          "/checkpoint-150000/trainer_state.json"),
        "openloop_csv": os.path.join(HERE, "n17_openloop_mse.csv"),
        "color": "#d62728",
    },
    "N1.7-300k": {
        "trainer_state": (f"{ROOT}/N1_7_fft_0607_300k_no_tune_visual"
                          "/checkpoint-300000/trainer_state.json"),
        "openloop_csv": os.path.join(HERE, "n17_300k_openloop_mse.csv"),
        "color": "#2ca02c",
    },
}


def load_trainer(path):
    h = json.load(open(path))["log_history"]
    steps = [e["step"] for e in h if "loss" in e]
    loss = [e["loss"] for e in h if "loss" in e]
    grad = [e.get("grad_norm", 0.0) for e in h if "loss" in e]
    return steps, loss, grad


def load_openloop(path):
    """Return (steps, mse) from a `step,mse,mae` CSV, or (None, None) if absent."""
    if not os.path.exists(path):
        return None, None
    steps, mse = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                steps.append(int(float(row["step"])))
                mse.append(float(row["mse"]))
            except (KeyError, ValueError):
                continue
    if not steps:
        return None, None
    order = sorted(range(len(steps)), key=lambda i: steps[i])
    return [steps[i] for i in order], [mse[i] for i in order]


def smooth(xs, w):
    out, acc = [], []
    for x in xs:
        acc.append(x)
        if len(acc) > w:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def downsample(steps, ys, n=600):
    if len(steps) <= n:
        return steps, ys
    stride = len(steps) / n
    idx = [int(i * stride) for i in range(n)]
    return [steps[i] for i in idx], [ys[i] for i in idx]


# ---- geometry ----
W, H = 1000, 1040
ML, MR = 90, 200
plot_w = W - ML - MR
P1_T, P1_H = 60, 260       # train-loss panel
P2_T, P2_H = 400, 260      # open-loop MSE panel
P3_T, P3_H = 740, 200      # grad-norm panel
STEP_MAX = 300000

LOSS_LO, LOSS_HI = 0.002, 2.0
MSE_LO, MSE_HI = 0.0005, 0.5
GRAD_LO, GRAD_HI = 0.0, 2.5


def sx(step):
    return ML + plot_w * step / STEP_MAX


def _logy(v, lo, hi, top, height):
    v = max(lo, min(hi, v))
    f = (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    return top + height * (1 - f)


def sy_loss(v):
    return _logy(v, LOSS_LO, LOSS_HI, P1_T, P1_H)


def sy_mse(v):
    return _logy(v, MSE_LO, MSE_HI, P2_T, P2_H)


def sy_grad(v):
    f = (v - GRAD_LO) / (GRAD_HI - GRAD_LO)
    return P3_T + P3_H * (1 - max(0.0, min(1.0, f)))


def polyline(steps, ys, syfun, color, width=2.0, opacity=1.0, dash=None):
    pts = " ".join(f"{sx(s):.1f},{syfun(y):.1f}" for s, y in zip(steps, ys))
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polyline fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-opacity="{opacity}"{d} points="{pts}"/>')


def marker(s, y, syfun, color):
    return f'<circle cx="{sx(s):.1f}" cy="{syfun(y):.1f}" r="3.2" fill="{color}"/>'


svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'font-family="DejaVu Sans, Arial, sans-serif" font-size="13">']
svg.append(f'<rect width="{W}" height="{H}" fill="white"/>')
svg.append(f'<text x="{W/2}" y="28" text-anchor="middle" font-size="19" font-weight="bold">'
           'GR00T N1.6 vs N1.7(150k) vs N1.7(300k retrain) — Can-Sorting, dataset 0403</text>')

# Panel 1: train loss (log)
svg.append(f'<text x="{ML}" y="{P1_T-14}" font-size="15" font-weight="bold">'
           '1. Training loss (velocity-field MSE on train data, log scale)</text>')
svg.append(f'<rect x="{ML}" y="{P1_T}" width="{plot_w}" height="{P1_H}" fill="#fafafa" stroke="#ccc"/>')
for dec in (1.0, 0.1, 0.01, 0.003):
    y = sy_loss(dec)
    svg.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+plot_w}" y2="{y:.1f}" stroke="#e3e3e3"/>')
    svg.append(f'<text x="{ML-8}" y="{y+4:.1f}" text-anchor="end" fill="#555">{dec:g}</text>')

# Panel 2: held-out open-loop MSE (log)
svg.append(f'<text x="{ML}" y="{P2_T-14}" font-size="15" font-weight="bold">'
           '2. Held-out open-loop action MSE (overfitting signal, log scale)</text>')
svg.append(f'<rect x="{ML}" y="{P2_T}" width="{plot_w}" height="{P2_H}" fill="#fafafa" stroke="#ccc"/>')
for dec in (0.5, 0.1, 0.01, 0.001):
    y = sy_mse(dec)
    svg.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+plot_w}" y2="{y:.1f}" stroke="#e3e3e3"/>')
    svg.append(f'<text x="{ML-8}" y="{y+4:.1f}" text-anchor="end" fill="#555">{dec:g}</text>')

# Panel 3: grad_norm (linear)
svg.append(f'<text x="{ML}" y="{P3_T-14}" font-size="15" font-weight="bold">3. Gradient norm</text>')
svg.append(f'<rect x="{ML}" y="{P3_T}" width="{plot_w}" height="{P3_H}" fill="#fafafa" stroke="#ccc"/>')
for gv in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
    y = sy_grad(gv)
    svg.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+plot_w}" y2="{y:.1f}" stroke="#e3e3e3"/>')
    svg.append(f'<text x="{ML-8}" y="{y+4:.1f}" text-anchor="end" fill="#555">{gv:g}</text>')

# X ticks (shared across all three panels)
for sxk in range(0, STEP_MAX + 1, 50000):
    x = sx(sxk)
    for (t, h) in ((P1_T, P1_H), (P2_T, P2_H), (P3_T, P3_H)):
        svg.append(f'<line x1="{x:.1f}" y1="{t}" x2="{x:.1f}" y2="{t+h}" stroke="#eee"/>')
    svg.append(f'<text x="{x:.1f}" y="{P3_T+P3_H+22:.1f}" text-anchor="middle" fill="#555">'
               f'{sxk//1000}k</text>')
svg.append(f'<text x="{ML+plot_w/2}" y="{P3_T+P3_H+44}" text-anchor="middle" font-size="14">'
           'training step</text>')

legend_y = P1_T + 14
finals = {}
have_openloop = False
best_lines = []
for i, (name, cfg) in enumerate(RUNS.items()):
    color = cfg["color"]
    steps, loss, grad = load_trainer(cfg["trainer_state"])
    ls = smooth(loss, 51)
    gs = smooth(grad, 51)
    sds, lds = downsample(steps, ls)
    _, gds = downsample(steps, gs)
    rds_s, rds_l = downsample(steps, loss)
    svg.append(polyline(rds_s, rds_l, sy_loss, color, width=1.0, opacity=0.20))
    svg.append(polyline(sds, lds, sy_loss, color, width=2.4))
    svg.append(polyline(sds, gds, sy_grad, color, width=2.0))
    tail = [l for s, l in zip(steps, loss) if s >= max(steps) - 10000]
    finals[name] = sum(tail) / len(tail)

    # Panel 2: open-loop MSE, plotted as points + connecting line (sparse sweep)
    o_steps, o_mse = load_openloop(cfg["openloop_csv"])
    if o_steps:
        have_openloop = True
        svg.append(polyline(o_steps, o_mse, sy_mse, color, width=2.0))
        for s, m in zip(o_steps, o_mse):
            svg.append(marker(s, m, sy_mse, color))
        bi = min(range(len(o_mse)), key=lambda k: o_mse[k])
        bx_, by_ = sx(o_steps[bi]), sy_mse(o_mse[bi])
        svg.append(f'<circle cx="{bx_:.1f}" cy="{by_:.1f}" r="6" fill="none" '
                   f'stroke="{color}" stroke-width="2"/>')
        svg.append(f'<text x="{bx_:.1f}" y="{by_-10:.1f}" text-anchor="middle" '
                   f'fill="{color}" font-size="11">best {o_steps[bi]//1000}k</text>')
        best_lines.append(f"  {name} min MSE = {o_mse[bi]:.4f} @ {o_steps[bi]//1000}k")

    ly = legend_y + i * 22
    svg.append(f'<line x1="{ML+plot_w+18}" y1="{ly}" x2="{ML+plot_w+48}" y2="{ly}" '
               f'stroke="{color}" stroke-width="3"/>')
    svg.append(f'<text x="{ML+plot_w+54}" y="{ly+4}" font-weight="bold">{name}</text>')
    yend = sy_loss(finals[name])
    svg.append(f'<text x="{ML+plot_w+6}" y="{yend+4:.1f}" fill="{color}" font-size="12">'
               f'{finals[name]:.4f}</text>')

# Placeholder note inside panel 2 when no CSV is available yet
if not have_openloop:
    cx, cy = ML + plot_w / 2, P2_T + P2_H / 2
    svg.append(f'<text x="{cx}" y="{cy-10}" text-anchor="middle" fill="#999" font-size="15">'
               'no open-loop MSE yet</text>')
    svg.append(f'<text x="{cx}" y="{cy+14}" text-anchor="middle" fill="#999" font-size="12">'
               'run openloop_mse_sweep.sh -> n16_openloop_mse.csv / n17_openloop_mse.csv</text>')
    svg.append(f'<text x="{cx}" y="{cy+32}" text-anchor="middle" fill="#999" font-size="12">'
               '(CSV columns: step,mse,mae) then re-run this script</text>')

# Summary box
ratio = finals["N1.7-300k"] / finals["N1.6-150k"]
bx, by = ML + plot_w + 16, P1_T + 70
box_lines = [
    "train loss (mean",
    " of last 10k):",
    f"  N1.6-150k = {finals['N1.6-150k']:.4f}",
    f"  N1.7-150k = {finals['N1.7-150k']:.4f}",
    f"  N1.7-300k = {finals['N1.7-300k']:.4f}",
    f"  N1.7/N1.6 = {ratio:.2f}x",
]
if best_lines:
    box_lines += ["", "open-loop min MSE:"] + best_lines
else:
    box_lines += ["", "success 18->53%", "with loss ~flat:", "loss != success."]
bh = 22 + 17 * len(box_lines)
svg.append(f'<rect x="{bx}" y="{by}" width="{MR-26}" height="{bh}" fill="#f4f4f4" stroke="#ccc"/>')
for j, t in enumerate(box_lines):
    svg.append(f'<text x="{bx+10}" y="{by+22+j*17}" font-size="12" fill="#222">{t}</text>')

svg.append('</svg>')
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "n16_vs_n17_compare.svg")
open(out, "w").write("\n".join(svg))
msg = (f"wrote {out}  (train loss N1.6 {finals['N1.6-150k']:.4f} / N1.7-150k "
       f"{finals['N1.7-150k']:.4f} / N1.7-300k {finals['N1.7-300k']:.4f}, {ratio:.2f}x")
msg += "; open-loop MSE overlaid)" if have_openloop else "; open-loop panel = placeholder)"
print(msg)
