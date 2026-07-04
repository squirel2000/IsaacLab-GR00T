"""Render the eval comparison figure (white background, data-driven):

  1. Training loss   — log scale, one curve per run
  2. Gradient norm   — linear, one curve per run
  3. Success rate    — stacked success / timeout / unsafe bars per run, with %

Driven entirely by the aggregated rows + per-run training curves (no hardcoded
numbers). Called by analysis.compare_runs.report; render() is pure (takes data, writes SVG).
"""

from __future__ import annotations

import math
from pathlib import Path

_SEG = {"success": "#2ca02c", "timeout": "#ff9800", "unsafe": "#d62728"}
_L, _R = 90, 910          # plot x-range (curve panels)
_W, _H = 1120, 940


def _smooth(ys, win=9):
    if win <= 1 or len(ys) < win:
        return ys
    out = []
    for i in range(len(ys)):
        a, b = max(0, i - win // 2), min(len(ys), i + win // 2 + 1)
        out.append(sum(ys[a:b]) / (b - a))
    return out


def _thin(seq, cap=500):
    """Stride-sample to at most `cap` points (keep the last), to keep the SVG light."""
    if len(seq) <= cap:
        return seq
    k = len(seq) // cap + 1
    return seq[::k] + ([seq[-1]] if (len(seq) - 1) % k else [])


def render(rows, curves, out_path, title="Closed-loop success vs training curves"):
    maxstep = max((c[0][-1] for c in curves.values() if c[0]), default=1) or 1
    sx = lambda step: _L + (_R - _L) * step / maxstep
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{_H}" '
         f'font-family="DejaVu Sans, Arial, sans-serif" font-size="13">',
         f'<rect width="{_W}" height="{_H}" fill="white"/>',
         f'<text x="{_W / 2}" y="28" text-anchor="middle" font-size="19" font-weight="bold">{title}</text>']

    # ---- Panel 1: training loss (log) ----
    y0, y1 = 70, 370
    losses = [v for c in curves.values() for v in c[1] if v and v > 0]
    lo, hi = (min(losses), max(losses)) if losses else (1e-3, 1.0)
    llo, lhi = math.log10(lo * 0.8), math.log10(hi * 1.2)
    ly = lambda v: y1 - (y1 - y0) * (math.log10(max(v, 1e-12)) - llo) / ((lhi - llo) or 1)
    s.append(f'<text x="{_L}" y="56" font-size="15" font-weight="bold">1. Training loss (log scale)</text>')
    s.append(f'<rect x="{_L}" y="{y0}" width="{_R - _L}" height="{y1 - y0}" fill="#fafafa" stroke="#ccc"/>')
    dec = math.floor(llo)
    while dec <= math.ceil(lhi):
        v = 10.0 ** dec
        if llo <= math.log10(v) <= lhi:
            y = ly(v)
            s.append(f'<line x1="{_L}" y1="{y:.1f}" x2="{_R}" y2="{y:.1f}" stroke="#e3e3e3"/>')
            s.append(f'<text x="{_L - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#555">{v:g}</text>')
        dec += 1

    # ---- Panel 2: gradient norm (linear) ----
    g0, g1 = 440, 610
    grads = sorted(v for c in curves.values() for v in c[2] if v is not None)
    # 95th-percentile top so early-training grad spikes don't squash the whole curve flat.
    gmax = (grads[min(len(grads) - 1, int(0.95 * len(grads)))] * 1.25) if grads else 1.0
    gmax = gmax or 1.0
    gy = lambda v: max(g0, g1 - (g1 - g0) * (v / gmax))  # clamp spikes to the panel top
    s.append(f'<text x="{_L}" y="426" font-size="15" font-weight="bold">2. Gradient norm</text>')
    s.append(f'<rect x="{_L}" y="{g0}" width="{_R - _L}" height="{g1 - g0}" fill="#fafafa" stroke="#ccc"/>')
    for k in range(6):
        gv = gmax * k / 5
        y = gy(gv)
        s.append(f'<line x1="{_L}" y1="{y:.1f}" x2="{_R}" y2="{y:.1f}" stroke="#e3e3e3"/>')
        s.append(f'<text x="{_L - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#555">{gv:.2g}</text>')

    # ---- x ticks (both curve panels) ----
    step = 0
    while step <= maxstep:
        x = sx(step)
        s.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="#eee"/>')
        s.append(f'<line x1="{x:.1f}" y1="{g0}" x2="{x:.1f}" y2="{g1}" stroke="#eee"/>')
        s.append(f'<text x="{x:.1f}" y="{g1 + 22}" text-anchor="middle" fill="#555">{step // 1000}k</text>')
        step += 50000

    # ---- curves ----
    for r in rows:
        steps, loss, grad = curves.get(r["tag"], ([], [], []))
        if not steps:
            continue
        lpts = _thin([(x, v) for x, v in zip(steps, _smooth(loss)) if v and v > 0])
        s.append(f'<polyline points="{" ".join(f"{sx(x):.1f},{ly(v):.1f}" for x, v in lpts)}" '
                 f'fill="none" stroke="{r["color"]}" stroke-width="1.8"/>')
        gg = [(x, v) for x, v in zip(steps, grad) if v is not None]
        if gg:
            gs = _thin(list(zip([x for x, _ in gg], _smooth([v for _, v in gg]))))
            s.append(f'<polyline points="{" ".join(f"{sx(x):.1f},{gy(v):.1f}" for x, v in gs)}" '
                     f'fill="none" stroke="{r["color"]}" stroke-width="1.8"/>')

    # ---- run legend (right margin; one block per run, two lines each) ----
    s.append(f'<text x="{_R + 18}" y="{y0 + 2}" font-size="12" font-weight="bold" fill="#444">runs</text>')
    for i, r in enumerate(rows):
        ly_ = y0 + 24 + i * 34
        s.append(f'<rect x="{_R + 18}" y="{ly_ - 10}" width="12" height="12" fill="{r["color"]}"/>')
        loss = f"{r['loss']:.4f}" if r["loss"] is not None else "—"
        s.append(f'<text x="{_R + 36}" y="{ly_}" font-size="12">{r["tag"]}</text>')
        s.append(f'<text x="{_R + 36}" y="{ly_ + 14}" font-size="10" fill="#777">loss {loss}</text>')

    # ---- Panel 3: success-rate stacked bars ----
    base_y, plot_h = 900, 180
    s.append(f'<text x="{_L}" y="676" font-size="15" font-weight="bold">3. Closed-loop success rate (success / timeout / unsafe)</text>')
    s.append(f'<rect x="{_L}" y="{base_y - plot_h}" width="{_R - _L}" height="{plot_h}" fill="#fafafa" stroke="#ccc"/>')
    for pct in (0, 50, 100):
        y = base_y - plot_h * pct / 100
        s.append(f'<line x1="{_L}" y1="{y:.1f}" x2="{_R}" y2="{y:.1f}" stroke="#e3e3e3"/>')
        s.append(f'<text x="{_L - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#555">{pct}%</text>')
    n_runs = max(len(rows), 1)
    slot = (_R - _L) / n_runs
    bw = min(90, slot * 0.5)
    for i, r in enumerate(rows):
        cx = _L + slot * (i + 0.5)
        x = cx - bw / 2
        n = r["n"] or 1
        y = base_y
        for name, val in (("success", r["success"]), ("timeout", r["truncated"]), ("unsafe", r["terminated"])):
            h = plot_h * val / n
            y -= h
            if h > 0.5:
                s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{_SEG[name]}"/>')
                if h > 16:
                    s.append(f'<text x="{cx:.1f}" y="{y + h / 2 + 4:.1f}" text-anchor="middle" '
                             f'font-size="11" fill="white">{100 * val / n:.0f}%</text>')
        s.append(f'<text x="{cx:.1f}" y="{base_y - plot_h - 6:.1f}" text-anchor="middle" '
                 f'font-size="13" font-weight="bold">{100 * r["rate"]:.0f}%</text>')
        s.append(f'<text x="{cx:.1f}" y="{base_y + 18:.1f}" text-anchor="middle" font-size="11">{r["tag"]} (n={r["n"]})</text>')

    lx = _L
    for name, c in _SEG.items():
        s.append(f'<rect x="{lx}" y="{base_y + 28}" width="12" height="12" fill="{c}"/>')
        s.append(f'<text x="{lx + 17}" y="{base_y + 38}" font-size="12">{name}</text>')
        lx += 110

    s.append("</svg>")
    Path(out_path).write_text("\n".join(s))
