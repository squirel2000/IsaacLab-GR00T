#!/usr/bin/env python3
"""Render N1.6 vs N1.7(150k) vs N1.7(300k retrain) comparison as a 3-panel SVG.

Reads each checkpoint's trainer_state.json log_history, applies rolling-mean
smoothing, and draws three stacked panels on a shared 0..300k step axis:
  - train loss            (log Y)   <- velocity-field MSE on train data
  - grad_norm             (linear)
  - closed-loop success   (bars)    <- 100-episode IsaacSim eval, per run

The point of the figure: N1.7-300k's train loss barely moves vs N1.7-150k
(still ~2.5x N1.6) yet closed-loop success jumps 18% -> 53%, i.e. train loss
is decoupled from task success. Pure-stdlib; no matplotlib dependency.

Usage: make_loss_svg.py <out.svg>
"""
import json
import math
import sys

ROOT = "/home/asus/Gits/IsaacLab-GR00T/artifacts/checkpoints/gr00t"
# name -> (trainer_state.json, color, closed-loop success % over 100 eps)
RUNS = {
    "N1.6-150k": (f"{ROOT}/openarm_linkerhando6_multitask_N16_150k_dataset_0403_no_tune_visual"
                  "/checkpoint-150000/trainer_state.json", "#1f77b4", 94),
    "N1.7-150k": (f"{ROOT}/openarm_linkerhando6_multitask_N17_150k_dataset_0403_no_tune_visual"
                  "/checkpoint-150000/trainer_state.json", "#d62728", 75),
    "N1.7-300k": (f"{ROOT}/N1_7_fft_0607_300k_no_tune_visual"
                  "/checkpoint-300000/trainer_state.json", "#2ca02c", 98),
}


def load(path):
    h = json.load(open(path))["log_history"]
    steps = [e["step"] for e in h if "loss" in e]
    loss = [e["loss"] for e in h if "loss" in e]
    grad = [e.get("grad_norm", 0.0) for e in h if "loss" in e]
    return steps, loss, grad


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
W, H = 1120, 940
ML, MR = 90, 210
plot_w = W - ML - MR
P1_T, P1_H = 70, 300       # loss panel
P2_T, P2_H = 440, 170      # grad panel
P3_T, P3_H = 700, 170      # success-rate bar panel
STEP_MAX = 300000

LOSS_LO, LOSS_HI = 0.002, 2.0    # log-scale bounds
GRAD_LO, GRAD_HI = 0.0, 2.5
SR_MAX = 100.0                   # success-rate axis (%)


def sx(step):
    return ML + plot_w * step / STEP_MAX


def sy_loss(v):
    v = max(LOSS_LO, min(LOSS_HI, v))
    f = (math.log10(v) - math.log10(LOSS_LO)) / (math.log10(LOSS_HI) - math.log10(LOSS_LO))
    return P1_T + P1_H * (1 - f)


def sy_grad(v):
    f = (v - GRAD_LO) / (GRAD_HI - GRAD_LO)
    return P2_T + P2_H * (1 - max(0.0, min(1.0, f)))


def sy_sr(v):
    return P3_T + P3_H * (1 - max(0.0, min(1.0, v / SR_MAX)))


def polyline(steps, ys, syfun, color, width=2.0, opacity=1.0):
    pts = " ".join(f"{sx(s):.1f},{syfun(y):.1f}" for s, y in zip(steps, ys))
    return (f'<polyline fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-opacity="{opacity}" points="{pts}"/>')


svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'font-family="DejaVu Sans, Arial, sans-serif" font-size="13">']
svg.append(f'<rect width="{W}" height="{H}" fill="white"/>')
svg.append(f'<text x="{W/2}" y="28" text-anchor="middle" font-size="19" font-weight="bold">'
           'GR00T N1.6 vs N1.7(150k) vs N1.7(300k retrain) — Can-Sorting, dataset 0403</text>')
svg.append(f'<text x="{W/2}" y="48" text-anchor="middle" font-size="13" fill="#666">'
           'train loss barely changes, yet closed-loop success goes 75% -> 98% '
           '(loss is decoupled from task success)</text>')

# Panel 1: loss (log)
svg.append(f'<text x="{ML}" y="{P1_T-14}" font-size="15" font-weight="bold">'
           '1. Training loss (velocity-field MSE on train data, log scale)</text>')
svg.append(f'<rect x="{ML}" y="{P1_T}" width="{plot_w}" height="{P1_H}" fill="#fafafa" stroke="#ccc"/>')
for dec in (1.0, 0.1, 0.01, 0.003):
    y = sy_loss(dec)
    svg.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+plot_w}" y2="{y:.1f}" stroke="#e3e3e3"/>')
    svg.append(f'<text x="{ML-8}" y="{y+4:.1f}" text-anchor="end" fill="#555">{dec:g}</text>')

# Panel 2: grad_norm (linear)
svg.append(f'<text x="{ML}" y="{P2_T-14}" font-size="15" font-weight="bold">2. Gradient norm</text>')
svg.append(f'<rect x="{ML}" y="{P2_T}" width="{plot_w}" height="{P2_H}" fill="#fafafa" stroke="#ccc"/>')
for gv in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
    y = sy_grad(gv)
    svg.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+plot_w}" y2="{y:.1f}" stroke="#e3e3e3"/>')
    svg.append(f'<text x="{ML-8}" y="{y+4:.1f}" text-anchor="end" fill="#555">{gv:g}</text>')

# X ticks (shared by panels 1 & 2)
for sxk in range(0, STEP_MAX + 1, 50000):
    x = sx(sxk)
    svg.append(f'<line x1="{x:.1f}" y1="{P1_T}" x2="{x:.1f}" y2="{P1_T+P1_H}" stroke="#eee"/>')
    svg.append(f'<line x1="{x:.1f}" y1="{P2_T}" x2="{x:.1f}" y2="{P2_T+P2_H}" stroke="#eee"/>')
    svg.append(f'<text x="{x:.1f}" y="{P2_T+P2_H+22:.1f}" text-anchor="middle" fill="#555">'
               f'{sxk//1000}k</text>')
svg.append(f'<text x="{ML+plot_w/2}" y="{P2_T+P2_H+44}" text-anchor="middle" font-size="14">'
           'training step</text>')

legend_y = P1_T + 14
finals = {}
for i, (name, (path, color, _sr)) in enumerate(RUNS.items()):
    steps, loss, grad = load(path)
    ls = smooth(loss, 51)
    gs = smooth(grad, 51)
    sds, lds = downsample(steps, ls)
    _, gds = downsample(steps, gs)
    rds_s, rds_l = downsample(steps, loss)
    svg.append(polyline(rds_s, rds_l, sy_loss, color, width=1.0, opacity=0.18))
    svg.append(polyline(sds, lds, sy_loss, color, width=2.4))
    svg.append(polyline(sds, gds, sy_grad, color, width=2.0))
    tail = [l for s, l in zip(steps, loss) if s >= max(steps) - 10000]
    finals[name] = sum(tail) / len(tail)
    # final-loss annotation at the curve's own right end
    xend = sx(max(steps))
    yend = sy_loss(finals[name])
    svg.append(f'<text x="{xend+5:.1f}" y="{yend+4:.1f}" fill="{color}" font-size="12">'
               f'{finals[name]:.4f}</text>')
    # legend
    ly = legend_y + i * 22
    svg.append(f'<line x1="{ML+plot_w+20}" y1="{ly}" x2="{ML+plot_w+50}" y2="{ly}" '
               f'stroke="{color}" stroke-width="3"/>')
    svg.append(f'<text x="{ML+plot_w+56}" y="{ly+4}" font-weight="bold" font-size="12">{name}</text>')

# Panel 3: closed-loop success-rate bars
svg.append(f'<text x="{ML}" y="{P3_T-14}" font-size="15" font-weight="bold">'
           '3. Closed-loop success rate (IsaacSim, 100 episodes each)</text>')
svg.append(f'<rect x="{ML}" y="{P3_T}" width="{plot_w}" height="{P3_H}" fill="#fafafa" stroke="#ccc"/>')
for sv in (0, 20, 40, 60, 80, 100):
    y = sy_sr(sv)
    svg.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+plot_w}" y2="{y:.1f}" stroke="#e3e3e3"/>')
    svg.append(f'<text x="{ML-8}" y="{y+4:.1f}" text-anchor="end" fill="#555">{sv}%</text>')
names = list(RUNS.keys())
n = len(names)
slot = plot_w / n
bar_w = slot * 0.42
for i, name in enumerate(names):
    color = RUNS[name][1]
    sr = RUNS[name][2]
    cx = ML + slot * (i + 0.5)
    x0 = cx - bar_w / 2
    y0 = sy_sr(sr)
    svg.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bar_w:.1f}" '
               f'height="{(P3_T+P3_H)-y0:.1f}" fill="{color}" fill-opacity="0.85"/>')
    svg.append(f'<text x="{cx:.1f}" y="{y0-8:.1f}" text-anchor="middle" '
               f'font-size="15" font-weight="bold" fill="{color}">{sr}%</text>')
    svg.append(f'<text x="{cx:.1f}" y="{P3_T+P3_H+22:.1f}" text-anchor="middle" '
               f'font-size="13" fill="#333">{name}</text>')

# summary box (right margin)
r17_150 = finals["N1.7-150k"] / finals["N1.6-150k"]
r17_300 = finals["N1.7-300k"] / finals["N1.6-150k"]
bx, by = ML + plot_w + 16, P1_T + 90
lines = [
    "final train loss",
    " (mean last 10k):",
    f"  N1.6-150k = {finals['N1.6-150k']:.4f}",
    f"  N1.7-150k = {finals['N1.7-150k']:.4f}",
    f"  N1.7-300k = {finals['N1.7-300k']:.4f}",
    f"  N1.7/N1.6 = {r17_300:.2f}x",
    "",
    "loss ~flat across",
    "the 150k->300k",
    "retrain, but",
    "success 75->98%.",
    "=> loss decoupled",
    "   from success.",
]
bh = 18 + 16 * len(lines)
svg.append(f'<rect x="{bx}" y="{by}" width="{MR-26}" height="{bh}" fill="#f4f4f4" stroke="#ccc"/>')
for j, t in enumerate(lines):
    svg.append(f'<text x="{bx+10}" y="{by+20+j*16}" font-size="11.5" fill="#222">{t}</text>')

svg.append('</svg>')
out = sys.argv[1] if len(sys.argv) > 1 else "n16_vs_n17_loss.svg"
open(out, "w").write("\n".join(svg))
print(f"wrote {out}  (N1.6 {finals['N1.6-150k']:.4f} / N1.7-150k {finals['N1.7-150k']:.4f} / "
      f"N1.7-300k {finals['N1.7-300k']:.4f}; success 94/75/98%)")
