#!/usr/bin/env python3
"""Plot one checkpoint's training curve (train loss log + grad norm) from its
trainer_state.json — no wandb needed. Generic standalone util.

  python scripts/eval/analysis/make_curve_svg.py <checkpoint>/trainer_state.json [out.svg]
"""

import json
import math
import sys
from pathlib import Path

W, H, L, R = 900, 520, 80, 860


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    state = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else state.parent / "train_curve.svg"
    hist = [h for h in json.loads(state.read_text()).get("log_history", []) if "loss" in h]
    steps = [h["step"] for h in hist]
    loss = [h["loss"] for h in hist]
    grad = [h.get("grad_norm") for h in hist]
    if not steps:
        sys.exit("no loss entries in trainer_state.json")
    smax = steps[-1] or 1
    sx = lambda v: L + (R - L) * v / smax
    pos = [v for v in loss if v and v > 0]
    llo, lhi = math.log10(min(pos) * 0.8), math.log10(max(pos) * 1.2)
    ly = lambda v: 60 + 200 * (lhi - math.log10(max(v, 1e-12))) / ((lhi - llo) or 1)
    gmax = (max(v for v in grad if v is not None) * 1.1) if any(grad) else 1.0
    gy = lambda v: 480 - 160 * (v / (gmax or 1))

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Arial, sans-serif" font-size="13">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{W/2}" y="26" text-anchor="middle" font-size="16" font-weight="bold">{state.parent.name} — training curve</text>',
         f'<text x="{L}" y="50" font-size="13" font-weight="bold">train loss (log)</text>',
         f'<text x="{L}" y="330" font-size="13" font-weight="bold">grad norm</text>']
    dec = math.floor(llo)
    while dec <= math.ceil(lhi):
        v = 10.0 ** dec
        if llo <= math.log10(v) <= lhi:
            s.append(f'<line x1="{L}" y1="{ly(v):.1f}" x2="{R}" y2="{ly(v):.1f}" stroke="#eee"/>')
            s.append(f'<text x="{L-6}" y="{ly(v)+4:.1f}" text-anchor="end" fill="#666">{v:g}</text>')
        dec += 1
    s.append(f'<polyline points="{" ".join(f"{sx(x):.1f},{ly(v):.1f}" for x, v in zip(steps, loss) if v and v > 0)}" fill="none" stroke="#1f77b4" stroke-width="1.8"/>')
    gg = [(x, v) for x, v in zip(steps, grad) if v is not None]
    if gg:
        s.append(f'<polyline points="{" ".join(f"{sx(x):.1f},{gy(v):.1f}" for x, v in gg)}" fill="none" stroke="#d62728" stroke-width="1.8"/>')
    for k in range(5):
        x = L + (R - L) * k / 4
        s.append(f'<text x="{x:.0f}" y="505" text-anchor="middle" fill="#666">{int(smax*k/4)//1000}k</text>')
    s.append("</svg>")
    out.write_text("\n".join(s))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
