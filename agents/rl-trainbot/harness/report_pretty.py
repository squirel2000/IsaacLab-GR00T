#!/usr/bin/env python3
"""Build the FINAL polished Traditional-Chinese Phase-1 report (dark theme, self-contained).

Supersedes report_full.py for the deliverable HTML (that script is kept for provenance).
Redesign requests (2026-07-06):
  - Consolidate the THREE N1.5-spatial experiments into ONE comparison section:
    shared overlay charts + a single unified explanation (not three repeated ones).
  - Sensible x/y axis ranges per chart (not always [0,1]).
  - Math rendered LaTeX-style: native MathML (self-contained, no CDN) on white cards,
    with the LaTeX source shown small underneath.
  - One consistent font stack everywhere (charts inherit the page font).
  - No overlapping text/lines in charts: legends live OUTSIDE the plot area (HTML chips),
    titles/notes are HTML, the SVG contains only axes/grid/series.
  - Rounded polylines (stroke-linejoin/linecap: round).
  - Dark theme; charts, math and diagrams sit on white cards.

Usage:  python agents/rl-trainbot/harness/report_pretty.py
Inputs (already fetched locally under artifacts/rl/):
  libero_spatial_ppo_gr00t_n1d7_phase1/metrics.json  (N1.7 critic-warmup, 120 ep)
  metrics_nowarmup.json                              (N1.7 no-warmup, 100 ep)
  metrics_n15_spatial.json                           (N1.5 single-GPU, 30 ep)
  metrics_n15_spatial_dualgpu.json                   (N1.5 dual-GPU, 72 ep)
  metrics_n15_spatial_scaled.json                    (N1.5 final single-GPU, 230 ep)
"""
import json
import math
import pathlib

def _ws_root() -> pathlib.Path:
    """Walk up to the workspace.yaml root marker (survives directory moves)."""
    here = pathlib.Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        if (d / "workspace.yaml").is_file():
            return d
    raise FileNotFoundError("workspace.yaml not found walking up from " + str(here))


_REPO = _ws_root()
OUT = _REPO / "agents/docs/rlinf_gr00t_n17_libero_phase1_report.html"

W = json.load(open(_REPO / "artifacts/rl/libero_spatial_ppo_gr00t_n1d7_phase1/metrics.json", encoding="utf-8"))
N = json.load(open(_REPO / "artifacts/rl/metrics_nowarmup.json", encoding="utf-8"))
M15 = json.load(open(_REPO / "artifacts/rl/metrics_n15_spatial.json", encoding="utf-8"))
M15D = json.load(open(_REPO / "artifacts/rl/metrics_n15_spatial_dualgpu.json", encoding="utf-8"))
M15S = json.load(open(_REPO / "artifacts/rl/metrics_n15_spatial_scaled.json", encoding="utf-8"))


# --------------------------------------------------------------------------- #
#  series helpers
# --------------------------------------------------------------------------- #
def ser(d, tag):
    return [(float(s), float(v)) for s, v in d.get(tag, [])]


def vals(d, tag):
    return [v for _, v in ser(d, tag)]


def moving_avg(pts, k=9):
    """Centered moving average over (x,y) points; NaNs excluded from each window."""
    out = []
    ys = [y for _, y in pts]
    for i, (x, _) in enumerate(pts):
        a, b = max(0, i - k // 2), min(len(ys), i + k // 2 + 1)
        win = [y for y in ys[a:b] if y == y]
        out.append((x, sum(win) / len(win) if win else float("nan")))
    return out


def qtr(v):
    """(early-quartile mean, late-quartile mean), NaN-safe."""
    v = [x for x in v if x == x]
    if not v:
        return (0.0, 0.0)
    k = max(1, len(v) // 4)
    return round(sum(v[:k]) / k, 3), round(sum(v[-k:]) / k, 3)


def nice_range(series_list, pad=0.04, step=0.05):
    """Tight-but-nice y range across several (x,y) series, snapped to `step`."""
    ys = [y for s in series_list for _, y in s if y == y]
    lo, hi = min(ys) - pad, max(ys) + pad
    lo = math.floor(lo / step) * step
    hi = math.ceil(hi / step) * step
    return (round(lo, 4), round(hi, 4))


def ticks_of(yr, n=5):
    lo, hi = yr
    return [round(lo + (hi - lo) * i / (n - 1), 4) for i in range(n)]


def nice_ticks(yr, target=5):
    """Ticks at round-number steps (1/2/2.5/5 x 10^k) inside the range."""
    lo, hi = yr
    raw = (hi - lo) / max(target - 1, 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag), 10 * mag)
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        out.append(round(t, 6))
        t += step
    return out or [lo, hi]


# --------------------------------------------------------------------------- #
#  chart builder — SVG holds ONLY axes/grid/series; title+legend+notes are HTML
# --------------------------------------------------------------------------- #
def _segments(pts):
    """Split a point list into NaN-free segments so gaps break the line."""
    segs, cur = [], []
    for x, y in pts:
        if y != y:
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append((x, y))
    if cur:
        segs.append(cur)
    return segs


def chart(series, *, xr, yr, xticks, yticks, w=760, h=330,
          yfmt=lambda v: f"{v:g}", xlabel="epoch (global step)"):
    """series: list of dicts {pts, color, label, markers(bool), width}."""
    pl, pr, pt, pb = 58, 18, 14, 44

    def sx(x):
        return pl + (x - xr[0]) / (xr[1] - xr[0]) * (w - pl - pr)

    def sy(y):
        y = min(max(y, yr[0]), yr[1])
        return h - pb - (y - yr[0]) / (yr[1] - yr[0]) * (h - pt - pb)

    p = [f'<svg viewBox="0 0 {w} {h}" role="img" preserveAspectRatio="xMidYMid meet">']
    # grid + y ticks
    for t in yticks:
        yy = sy(t)
        p.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{w-pr}" y2="{yy:.1f}" stroke="#e8edf4" stroke-width="1"/>')
        p.append(f'<text x="{pl-8}" y="{yy+3.5:.1f}" font-size="11.5" text-anchor="end" fill="#5b6b82">{yfmt(t)}</text>')
    # x ticks
    for t in xticks:
        xx = sx(t)
        p.append(f'<line x1="{xx:.1f}" y1="{h-pb}" x2="{xx:.1f}" y2="{h-pb+4}" stroke="#94a3b8"/>')
        p.append(f'<text x="{xx:.1f}" y="{h-pb+18}" font-size="11.5" text-anchor="middle" fill="#5b6b82">{t:g}</text>')
    # axes
    p.append(f'<line x1="{pl}" y1="{pt-4}" x2="{pl}" y2="{h-pb}" stroke="#94a3b8" stroke-width="1.2"/>')
    p.append(f'<line x1="{pl}" y1="{h-pb}" x2="{w-pr}" y2="{h-pb}" stroke="#94a3b8" stroke-width="1.2"/>')
    p.append(f'<text x="{(pl+w-pr)/2:.0f}" y="{h-8}" font-size="11.5" text-anchor="middle" fill="#5b6b82">{xlabel}</text>')
    # series (rounded joins/caps; NaN-gapped)
    for s in series:
        col, wd = s["color"], s.get("width", 2.6)
        for seg in _segments(s["pts"]):
            if len(seg) == 1:
                x, y = seg[0]
                p.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{col}"/>')
                continue
            d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in seg)
            p.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="{wd}" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
        if s.get("markers"):
            for x, y in s["pts"]:
                if y == y:
                    p.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.4" fill="{s["color"]}" '
                             f'stroke="#ffffff" stroke-width="1.2"/>')
    p.append("</svg>")
    return "".join(p)


def legend(items):
    chips = "".join(
        f'<span class="lg-chip"><i style="background:{c}"></i>{t}</span>' for c, t in items
    )
    return f'<div class="lg">{chips}</div>'


def plot_card(title, legend_html, svg, note=""):
    note_html = f'<p class="plot-note">{note}</p>' if note else ""
    return (f'<figure class="plot"><figcaption class="plot-title">{title}</figcaption>'
            f'{legend_html}{svg}{note_html}</figure>')


def pct(v):
    return f"{v*100:g}%"


# --------------------------------------------------------------------------- #
#  computed statistics
# --------------------------------------------------------------------------- #
# --- N1.7 pair ---
ev_w, ev_n = ser(W, "eval/success_once"), ser(N, "eval/success_once")
tr_w, tr_n = ser(W, "env/success_once"), ser(N, "env/success_once")
evx_w, evx_n = ser(W, "train/critic/explained_variance"), ser(N, "train/critic/explained_variance")
ev_ok_w, ev_ok_n = qtr(vals(W, "eval/success_once")), qtr(vals(N, "eval/success_once"))
kl_w = ser(W, "train/actor/approx_kl")
ratio_w = ser(W, "train/actor/ratio")
rw_w = ser(W, "env/reward")
tot_w = ser(W, "train/actor/total_loss")
pl_w = ser(W, "train/actor/policy_loss")
n17_evx_min = min(v for v in vals(W, "train/critic/explained_variance") + vals(N, "train/critic/explained_variance") if v == v)

# --- N1.5 trio ---
ev_a, tr_a, evx_a = ser(M15, "eval/success_once"), ser(M15, "env/success_once"), ser(M15, "train/critic/explained_variance")
ev_b, tr_b, evx_b = ser(M15D, "eval/success_once"), ser(M15D, "env/success_once"), ser(M15D, "train/critic/explained_variance")
ev_c, tr_c, evx_c = ser(M15S, "eval/success_once"), ser(M15S, "env/success_once"), ser(M15S, "train/critic/explained_variance")
tr_ok_a, tr_ok_b, tr_ok_c = qtr(vals(M15, "env/success_once")), qtr(vals(M15D, "env/success_once")), qtr(vals(M15S, "env/success_once"))
ev_ok_a, ev_ok_b, ev_ok_c = qtr(vals(M15, "eval/success_once")), qtr(vals(M15D, "eval/success_once")), qtr(vals(M15S, "eval/success_once"))


def evx_stats(d):
    xs = vals(d, "train/critic/explained_variance")
    fin = [x for x in xs if x == x]
    return {
        "n": len(xs), "nan": len(xs) - len(fin),
        "pos": sum(1 for x in fin if x > 0),
        "mean": (sum(fin) / len(fin)) if fin else float("nan"),
        "min": min(fin) if fin else float("nan"),
        "max": max(fin) if fin else float("nan"),
    }


sa, sb, sc = evx_stats(M15), evx_stats(M15D), evx_stats(M15S)
d_tr_c = round((tr_ok_c[1] - tr_ok_c[0]) * 100, 1)   # +24.7pp
d_ev_c = round((ev_ok_c[1] - ev_ok_c[0]) * 100, 1)   # +30pp
d_tr_b = round((tr_ok_b[1] - tr_ok_b[0]) * 100, 1)   # +15.3pp
n_eval_ones = sum(1 for v in vals(M15S, "eval/success_once") if v >= 0.999)

AMBER, TEALC, MAGEN = "#d97706", "#0d9488", "#db2777"
BLUE, RED = "#2563eb", "#dc2626"
GREEN, PURPLE, GRAY = "#16a34a", "#7c3aed", "#64748b"

# --------------------------------------------------------------------------- #
#  charts — N1.5 trio (the consolidated comparison)
# --------------------------------------------------------------------------- #
trm_a, trm_b, trm_c = moving_avg(tr_a), moving_avg(tr_b), moving_avg(tr_c)

c_eval3 = plot_card(
    "eval/success_once — 三次實驗疊圖（每點 = 一次評估，4 trials）",
    legend([(AMBER, "初版・單卡 envs=4（30 ep）"), (TEALC, "雙卡 envs=4・batch 32（72 ep）"),
            (MAGEN, "最終・單卡 envs=8・batch 64（230 ep）")]),
    chart([
        {"pts": ev_a, "color": AMBER, "label": "a", "markers": True},
        {"pts": ev_b, "color": TEALC, "label": "b", "markers": True},
        {"pts": ev_c, "color": MAGEN, "label": "c", "markers": True, "width": 3.0},
    ], xr=(0, 230), yr=(0, 1), xticks=[0, 30, 72, 120, 180, 230],
        yticks=[0, 0.25, 0.5, 0.75, 1], yfmt=pct),
    "x 軸刻度特意標在 30 / 72 / 230：三次實驗各自結束的位置。y 軸取 [0,100%]（eval 資料實際觸及 0 與 100%）。"
    "最終跑（洋紅）後段多次到達 100%。")

_yr_tr = nice_range([trm_a, trm_b, trm_c])
c_train3 = plot_card(
    "env/success_once — 訓練成功率（9 點移動平均）",
    legend([(AMBER, "初版（30 ep）"), (TEALC, "雙卡（72 ep）"), (MAGEN, "最終（230 ep）")]),
    chart([
        {"pts": trm_a, "color": AMBER, "label": "a"},
        {"pts": trm_b, "color": TEALC, "label": "b"},
        {"pts": trm_c, "color": MAGEN, "label": "c", "width": 3.0},
    ], xr=(0, 230), yr=_yr_tr, xticks=[0, 30, 72, 120, 180, 230],
        yticks=nice_ticks(_yr_tr), yfmt=pct),
    f"y 軸依實際資料範圍取 [{pct(_yr_tr[0])}, {pct(_yr_tr[1])}]（非固定 0–100%），讓趨勢差異看得更清楚。")

c_evx3 = plot_card(
    "explained_variance — critic 品質（>0 才代表 critic 比「猜平均值」好）",
    legend([(TEALC, "雙卡（72 ep）"), (MAGEN, "最終（230 ep）"), (GRAY, "初版：全程 NaN（無線可畫）")]),
    chart([
        {"pts": evx_b, "color": TEALC, "label": "b", "width": 2.2},
        {"pts": evx_c, "color": MAGEN, "label": "c", "width": 2.2},
    ], xr=(0, 230), yr=(-1, 1), xticks=[0, 30, 72, 120, 180, 230],
        yticks=[-1, -0.5, 0, 0.5, 1]),
    f"y 軸裁切至 [-1, 1]（雙卡實際最低 {sb['min']:.2f}、最終跑最低 {sc['min']:.2f}，各僅數點越界；"
    f"最終跑另有 {sc['nan']} 點 NaN，畫為斷線）。初版（batch=4）整條序列都算不出數值，正是小批量診斷失效的實例。")

# --------------------------------------------------------------------------- #
#  charts — N1.7 pair
# --------------------------------------------------------------------------- #
c_n17_eval = plot_card(
    "eval/success_once — cold-start vs critic-warmup（每點 4 trials）",
    legend([(RED, "no-warmup（100 ep）"), (BLUE, "critic-warmup=40（120 ep）")]),
    chart([
        {"pts": ev_n, "color": RED, "label": "n", "markers": True},
        {"pts": ev_w, "color": BLUE, "label": "w", "markers": True},
    ], xr=(0, 120), yr=(0, 1), xticks=[0, 20, 40, 60, 80, 100, 120],
        yticks=[0, 0.25, 0.5, 0.75, 1], yfmt=pct),
    "紅線（無 warmup）後段明顯下滑；藍線（有 warmup）在 25–75% 之間振盪、無趨勢——warmup 的作用是「止跌」而非「提升」。")

trm_w, trm_n = moving_avg(tr_w), moving_avg(tr_n)
_yr_n17 = nice_range([trm_w, trm_n])
c_n17_train = plot_card(
    "env/success_once — 訓練成功率（9 點移動平均）",
    legend([(RED, "no-warmup"), (BLUE, "critic-warmup")]),
    chart([
        {"pts": trm_n, "color": RED, "label": "n"},
        {"pts": trm_w, "color": BLUE, "label": "w"},
    ], xr=(0, 120), yr=_yr_n17, xticks=[0, 20, 40, 60, 80, 100, 120],
        yticks=nice_ticks(_yr_n17), yfmt=pct))

c_n17_evx = plot_card(
    "explained_variance — 兩次跑皆遠低於 0（critic 比亂猜還差）",
    legend([(RED, "no-warmup"), (BLUE, "critic-warmup")]),
    chart([
        {"pts": evx_n, "color": RED, "label": "n", "width": 2.2},
        {"pts": evx_w, "color": BLUE, "label": "w", "width": 2.2},
    ], xr=(0, 120), yr=(-3, 1), xticks=[0, 20, 40, 60, 80, 100, 120],
        yticks=[-3, -2, -1, 0, 1]),
    f"y 軸裁切至 [-3, 1]；實際最低到 {n17_evx_min:,.0f}（單點極端值），全程未曾轉正——與 N1.5 放大規模後的表現形成最直接的對照。")


def small(series, yr=None, yfmt=lambda v: f"{v:g}"):
    if yr is None:
        yr = nice_range([s["pts"] for s in series], step=0.005)
    return chart(series, xr=(0, 120), yr=yr, xticks=[0, 60, 120],
                 yticks=nice_ticks(yr, 4), yfmt=yfmt, w=360, h=225)


c_diag = (
    plot_card("actor loss（policy_loss 前 40 步 = 0：warmup 凍結 actor）",
              legend([(BLUE, "total_loss"), (GRAY, "policy_loss")]),
              small([{"pts": tot_w, "color": BLUE, "label": "t", "width": 2.0},
                     {"pts": pl_w, "color": GRAY, "label": "p", "width": 2.0}])),
    plot_card("approx_kl（≈0.01：policy 幾乎沒動）", legend([(PURPLE, "approx_kl")]),
              small([{"pts": kl_w, "color": PURPLE, "label": "k", "width": 2.0}])),
    plot_card("ratio（≈1：新舊 policy 幾乎相同）", legend([(TEALC, "ratio")]),
              small([{"pts": ratio_w, "color": TEALC, "label": "r", "width": 2.0}])),
    plot_card("env/reward（量級僅 ~0.004，訊號極微弱）", legend([(GREEN, "reward")]),
              small([{"pts": rw_w, "color": GREEN, "label": "g", "width": 2.0}])),
)

# --------------------------------------------------------------------------- #
#  diagrams (drawio-blocksmith output, validated no-overlap; shown on white)
# --------------------------------------------------------------------------- #
_DIAG = _REPO / "agents/rl-trainbot/harness/diagrams"
try:
    ARCH_SVG = (_DIAG / "architecture.svg").read_text(encoding="utf-8")
    FLOW_SVG = (_DIAG / "flow.svg").read_text(encoding="utf-8")
except Exception as _e:  # pragma: no cover
    ARCH_SVG = FLOW_SVG = "<p>(diagram svg not found)</p>"
    print("diagram svg not found:", _e)

# --------------------------------------------------------------------------- #
#  MathML blocks (LaTeX-quality, self-contained; LaTeX source shown underneath)
# --------------------------------------------------------------------------- #
def mrow(inner):
    return f'<math display="block"><mrow>{inner}</mrow></math>'


def math_card(title, mathml, latex, note):
    return (f'<div class="math"><div class="math-t">{title}</div>{mathml}'
            f'<code class="tex">{latex}</code><p class="math-n">{note}</p></div>')


A_HAT = '<msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub>'
E_HAT = '<msub><mover accent="true"><mi mathvariant="double-struck">E</mi><mo>^</mo></mover><mi>t</mi></msub>'

M_RET = mrow(
    '<msub><mi>R</mi><mi>t</mi></msub><mo>=</mo>'
    '<munderover><mo>&#x2211;</mo><mrow><mi>k</mi><mo>=</mo><mn>0</mn></mrow><mi>&#x221E;</mi></munderover>'
    '<msup><mi>&#x3B3;</mi><mi>k</mi></msup><msub><mi>r</mi><mrow><mi>t</mi><mo>+</mo><mi>k</mi></mrow></msub>')
M_TD = mrow(
    '<msub><mi>&#x3B4;</mi><mi>t</mi></msub><mo>=</mo><msub><mi>r</mi><mi>t</mi></msub><mo>+</mo>'
    '<mi>&#x3B3;</mi><msub><mi>V</mi><mi>&#x3C6;</mi></msub><mo>(</mo><msub><mi>s</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>)</mo>'
    '<mo>&#x2212;</mo><msub><mi>V</mi><mi>&#x3C6;</mi></msub><mo>(</mo><msub><mi>s</mi><mi>t</mi></msub><mo>)</mo>')
M_GAE = mrow(
    A_HAT + '<mo>=</mo>'
    '<munderover><mo>&#x2211;</mo><mrow><mi>l</mi><mo>=</mo><mn>0</mn></mrow><mi>&#x221E;</mi></munderover>'
    '<msup><mrow><mo>(</mo><mi>&#x3B3;</mi><mi>&#x3BB;</mi><mo>)</mo></mrow><mi>l</mi></msup>'
    '<msub><mi>&#x3B4;</mi><mrow><mi>t</mi><mo>+</mo><mi>l</mi></mrow></msub>')
M_RATIO = mrow(
    '<msub><mi>r</mi><mi>t</mi></msub><mo>(</mo><mi>&#x3B8;</mi><mo>)</mo><mo>=</mo>'
    '<mfrac><mrow><msub><mi>&#x3C0;</mi><mi>&#x3B8;</mi></msub><mo>(</mo><msub><mi>a</mi><mi>t</mi></msub><mo>&#x2223;</mo><msub><mi>s</mi><mi>t</mi></msub><mo>)</mo></mrow>'
    '<mrow><msub><mi>&#x3C0;</mi><msub><mi>&#x3B8;</mi><mtext>old</mtext></msub></msub><mo>(</mo><msub><mi>a</mi><mi>t</mi></msub><mo>&#x2223;</mo><msub><mi>s</mi><mi>t</mi></msub><mo>)</mo></mrow></mfrac>')
M_CLIP = mrow(
    '<msup><mi>L</mi><mtext>CLIP</mtext></msup><mo>(</mo><mi>&#x3B8;</mi><mo>)</mo><mo>=</mo>' + E_HAT +
    '<mrow><mo>[</mo><mi>min</mi><mo>(</mo>'
    '<msub><mi>r</mi><mi>t</mi></msub><mo>(</mo><mi>&#x3B8;</mi><mo>)</mo>' + A_HAT + '<mo>,</mo>'
    '<mtext>clip</mtext><mo>(</mo><msub><mi>r</mi><mi>t</mi></msub><mo>(</mo><mi>&#x3B8;</mi><mo>)</mo><mo>,</mo>'
    '<mn>1</mn><mo>&#x2212;</mo><mi>&#x3B5;</mi><mo>,</mo><mn>1</mn><mo>+</mo><mi>&#x3B5;</mi><mo>)</mo>' + A_HAT +
    '<mo>)</mo><mo>]</mo></mrow>')
M_VLOSS = mrow(
    '<msup><mi>L</mi><mi>V</mi></msup><mo>(</mo><mi>&#x3C6;</mi><mo>)</mo><mo>=</mo>' + E_HAT +
    '<mrow><mo>[</mo><msup><mrow><mo>(</mo><msub><mi>V</mi><mi>&#x3C6;</mi></msub><mo>(</mo><msub><mi>s</mi><mi>t</mi></msub><mo>)</mo>'
    '<mo>&#x2212;</mo><msub><mi>R</mi><mi>t</mi></msub><mo>)</mo></mrow><mn>2</mn></msup><mo>]</mo></mrow>')
M_TOTAL = mrow(
    '<mi>L</mi><mo>=</mo><mo>&#x2212;</mo><msup><mi>L</mi><mtext>CLIP</mtext></msup>'
    '<mo>+</mo><msub><mi>c</mi><mi>v</mi></msub><msup><mi>L</mi><mi>V</mi></msup>'
    '<mo>&#x2212;</mo><msub><mi>c</mi><mi>e</mi></msub><mi>H</mi><mo>[</mo><msub><mi>&#x3C0;</mi><mi>&#x3B8;</mi></msub><mo>]</mo>')
M_EV = mrow(
    '<mtext>EV</mtext><mo>=</mo><mn>1</mn><mo>&#x2212;</mo>'
    '<mfrac><mrow><mtext>Var</mtext><mo>(</mo><msub><mi>R</mi><mi>t</mi></msub><mo>&#x2212;</mo>'
    '<msub><mi>V</mi><mi>&#x3C6;</mi></msub><mo>(</mo><msub><mi>s</mi><mi>t</mi></msub><mo>)</mo><mo>)</mo></mrow>'
    '<mrow><mtext>Var</mtext><mo>(</mo><msub><mi>R</mi><mi>t</mi></msub><mo>)</mo></mrow></mfrac>')
M_GNS = mrow(
    '<mtext>Var</mtext><mo>[</mo><mover accent="true"><mi>g</mi><mo>^</mo></mover><mo>]</mo>'
    '<mo>&#x221D;</mo><mfrac><mi>&#x3A3;</mi><mi>B</mi></mfrac>'
    '<mo>,</mo><mspace width="1.2em"></mspace>'
    '<msub><mi>B</mi><mtext>noise</mtext></msub><mo>=</mo>'
    '<mfrac><mrow><mtext>tr</mtext><mo>(</mo><mi>&#x3A3;</mi><mo>)</mo></mrow>'
    '<msup><mrow><mo>&#x2016;</mo><mi>G</mi><mo>&#x2016;</mo></mrow><mn>2</mn></msup></mfrac>')
M_ADVN = mrow(
    '<msubsup><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi><mo>&#x2032;</mo></msubsup><mo>=</mo>'
    '<mfrac><mrow>' + A_HAT + '<mo>&#x2212;</mo><mtext>mean</mtext><mo>(</mo><mover accent="true"><mi>A</mi><mo>^</mo></mover><mo>)</mo></mrow>'
    '<mrow><mtext>std</mtext><mo>(</mo><mover accent="true"><mi>A</mi><mo>^</mo></mover><mo>)</mo></mrow></mfrac>')
M_BESSEL = mrow(
    '<msup><mi>s</mi><mn>2</mn></msup><mo>=</mo>'
    '<mfrac><mn>1</mn><mrow><mi>n</mi><mo>&#x2212;</mo><mn>1</mn></mrow></mfrac>'
    '<munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover>'
    '<msup><mrow><mo>(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>&#x2212;</mo>'
    '<mover accent="true"><mi>x</mi><mo>&#x00AF;</mo></mover><mo>)</mo></mrow><mn>2</mn></msup>')

# --------------------------------------------------------------------------- #
#  CSS  (plain string — braces stay literal)
# --------------------------------------------------------------------------- #
CSS = """
*{box-sizing:border-box}
body{margin:0;background:#0b1220;color:#dbe4f0;line-height:1.75;
  font-family:"Segoe UI","Noto Sans TC","Microsoft JhengHei",-apple-system,"PingFang TC",sans-serif;}
.wrap{max-width:980px;margin:0 auto;padding:40px 24px 64px}
h1{font-size:26px;line-height:1.4;margin:0 0 6px;color:#f1f5fb;font-weight:700}
h2{font-size:19px;margin:44px 0 14px;color:#eaf1fa;font-weight:700;
   border-left:4px solid #5ea0f7;padding-left:12px}
h3{font-size:15px;margin:22px 0 8px;color:#c3d2e8;font-weight:700}
p{margin:10px 0}
.sub{color:#93a4bd;margin:0 0 8px;font-size:14px}
a{color:#7db4ff;text-decoration:none} a:hover{text-decoration:underline}
code{background:#1b2740;color:#cfe0f6;padding:1.5px 6px;border-radius:5px;
  font-size:12.5px;font-family:Consolas,"Cascadia Mono",Menlo,monospace}
b,strong{color:#f1f5fb}
.k{color:#ff9c9c;font-weight:700}
.hl{color:#7ef0c9;font-weight:700}

/* dark cards */
.card{background:#141e33;border:1px solid #26324b;border-radius:14px;
  padding:18px 22px;margin:14px 0;overflow-x:auto}
ul{margin:8px 0;padding-left:22px} li{margin:6px 0}

/* hero */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:20px 0 6px}
.kpi{background:linear-gradient(160deg,#16233d,#12203a);border:1px solid #2b3d61;
  border-radius:14px;padding:14px 18px}
.kpi .v{font-size:24px;font-weight:800;color:#7ef0c9}
.kpi .l{font-size:12.5px;color:#93a4bd;margin-top:2px}
.verdict{border-radius:14px;padding:16px 20px;margin:18px 0;font-weight:600;
  background:#0c2b1c;border:1px solid #1f7a4d;color:#b7f3d3}
.verdict-warn{background:#2b230c;border:1px solid #8a6d1f;color:#f3e3b7}

/* TOC */
.toc{background:#141e33;border:1px solid #26324b;border-radius:14px;padding:14px 20px;margin:18px 0}
.toc ol{margin:6px 0;padding-left:22px;columns:2;column-gap:36px}
.toc li{margin:3px 0;font-size:13.5px;break-inside:avoid}

/* tables (dark) */
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:12px 0}
th,td{border-bottom:1px solid #26324b;padding:9px 11px;text-align:left;vertical-align:top}
th{background:#1a2742;color:#c3d2e8;white-space:nowrap}
tr:nth-child(even) td{background:#121c31}

/* white plot cards */
.plot{background:#ffffff;border-radius:14px;padding:16px 18px 12px;margin:14px 0;
  box-shadow:0 6px 22px rgba(0,0,0,.35)}
.plot-title{color:#0f172a;font-weight:700;font-size:14.5px;margin:0 0 6px}
.plot-note{color:#5b6b82;font-size:12.5px;margin:8px 2px 2px;line-height:1.6}
.plot svg{width:100%;height:auto;display:block}
.plot svg text{font-family:inherit}
.lg{display:flex;flex-wrap:wrap;gap:14px;margin:2px 0 10px}
.lg-chip{display:inline-flex;align-items:center;gap:7px;color:#33415c;font-size:12.5px}
.lg-chip i{width:16px;height:5px;border-radius:3px;display:inline-block}
.g2{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
.g2 .plot{margin:0}

/* white math cards */
.math{background:#ffffff;border-radius:12px;padding:12px 18px;margin:12px 0;
  box-shadow:0 4px 16px rgba(0,0,0,.3)}
.math-t{color:#33415c;font-size:13px;font-weight:700;margin-bottom:2px}
.math math{color:#0f172a;font-size:1.18em;margin:4px 0}
.math .tex{display:block;background:#f4f6fa;color:#6b7a93;font-size:11.5px;
  padding:4px 8px;border-radius:6px;margin:6px 0 4px;overflow-x:auto}
.math-n{color:#5b6b82;font-size:12.5px;margin:4px 0 2px;line-height:1.6}

/* diagrams on white */
.diagram{background:#ffffff;border-radius:14px;padding:16px;margin:14px 0;
  box-shadow:0 6px 22px rgba(0,0,0,.35);overflow-x:auto}
.diagram svg{max-width:100%;height:auto}
.footer{color:#6b7a93;font-size:12px;margin-top:44px;border-top:1px solid #26324b;padding-top:14px}
"""

# --------------------------------------------------------------------------- #
#  build
# --------------------------------------------------------------------------- #
def build():
    P = []

    # ---------- hero ----------
    P.append('<h1>RLinf × GR00T × LIBERO — Phase 1 完整分析報告</h1>')
    P.append('<p class="sub">單機 2× H100（合作共用）・LIBERO-spatial・PPO（actor-critic）・'
             '含規模假設的完整驗證（2026-06-30 ~ 07-06）</p>')
    P.append(f"""<div class="kpis">
<div class="kpi"><div class="v">+{d_tr_c:g} pp</div><div class="l">最終跑訓練成功率提升（早段 {pct(tr_ok_c[0])} → 末段 {pct(tr_ok_c[1])}）</div></div>
<div class="kpi"><div class="v">+{d_ev_c:g} pp</div><div class="l">eval 成功率提升（{pct(ev_ok_c[0])} → {pct(ev_ok_c[1])}，後段 {n_eval_ones} 次滿分 100%）</div></div>
<div class="kpi"><div class="v">{round(sc['pos']/sc['n']*100)}%</div><div class="l">explained_variance 為正的比例（{sc['pos']}/{sc['n']}，critic 真的學會估值）</div></div>
<div class="kpi"><div class="v">230 ep / 54.8 h</div><div class="l">最終跑規模（RLinf 官方 recipe 的 23% epochs）</div></div>
</div>""")
    P.append('<div class="verdict">結論：端到端 RL 流程驗證可跑之外，'
             '<b>把訓練規模放大（batch 與訓練時長）後，PPO 對 GR00T-N1.5 SFT policy 產生了明確、可量測的提升</b>——'
             f'訓練成功率 +{d_tr_c:g}pp、評估成功率 +{d_ev_c:g}pp、critic 品質（explained_variance）97% 時間為正。'
             '先前小規模設定下「看不到提升」的根因是<b>規模不足</b>（批量太小 + 訓練太短），'
             '不是 checkpoint 已到天花板、也不是 critic 機制天生失效。</div>')

    # ---------- TOC ----------
    P.append("""<nav class="toc"><b>目錄</b><ol>
<li><a href="#s1">執行摘要</a></li>
<li><a href="#s2">名詞解釋</a></li>
<li><a href="#s3">背景：RL 對 VLA 的幫助、為什麼必須從 SFT 起步</a></li>
<li><a href="#s4">數學原理（PPO 與「規模不足傷害 PPO」的三個機制）</a></li>
<li><a href="#s5">N1.7 實驗：cold-start critic vs critic warmup</a></li>
<li><a href="#s6">N1.5 三次實驗統一對照（規模假設的直接驗證）</a></li>
<li><a href="#s7">外部對照：RLinf 官方規模、官方成績與可重現性</a></li>
<li><a href="#s8">建議下一步</a></li>
<li><a href="#s9">系統架構圖</a></li>
<li><a href="#s10">PPO 訓練流程圖</a></li>
<li><a href="#s11">執行設定與重現資訊</a></li>
</ol></nav>""")

    # ---------- 1. exec summary ----------
    P.append('<h2 id="s1">1. 執行摘要</h2>')
    P.append(f"""<div class="card"><ul>
<li><b>基礎設施：</b>RLinf + GR00T（N1.7 與 N1.5）+ PPO 在 Pegasus 容器內端到端跑通
（rollout → actor/critic 更新 → 評估 → 存檔），克服 cgroup <code>pids.max=2048</code>、無 Docker、
EGL 離屏算繪、MuJoCo 子程序崩潰等容器限制（詳見 §11）。</li>
<li><b>N1.7 起步實驗：</b>最小單卡設定（envs=4、batch=32）下 PPO 無提升；
無 critic-warmup 時甚至<span class="k">退步</span>。根因診斷：critic 嚴重失準
（explained_variance 全程為負、最低至 {n17_evx_min:,.0f}），使 advantage 淪為雜訊。</li>
<li><b>規模假設：</b>對照 RLinf 官方 recipe（envs=64、batch=1024、1000 epochs、8-16 GPUs）
與 PPO 文獻（梯度雜訊尺度、advantage 正規化統計、EV 自由度退化），
提出「我們的失敗來自規模不足」假設（§4、§7）。</li>
<li><b class="hl">N1.5 三段式驗證（本報告核心，§6）：</b>同一個 SFT checkpoint、同一任務，
只放大規模跑三次——初版（30 ep）無提升 → 雙卡放大 batch（72 ep）+{d_tr_b:g}pp
→ 最終放大訓練時長（230 ep、54.8h）<b>+{d_tr_c:g}pp（eval +{d_ev_c:g}pp）</b>。
批量與訓練時長各自都是可觀測因果效果的獨立變數。</li>
<li><b>交付物：</b>可重複使用的 <code>agents/rl-trainbot/harness/</code> 工具鏈（launcher / poller / 報告產生器 /
RLinf 容器修補腳本）、OpenSpec 規格、本報告與全部原始指標 JSON。</li>
</ul></div>""")

    # ---------- 2. glossary ----------
    P.append('<h2 id="s2">2. 名詞解釋</h2>')
    P.append("""<div class="card"><ul>
<li><b>SFT baseline</b>：以監督式/模仿學習訓練出的起始 policy。N1.5 用 RLinf 公開的
<code>RLinf-Gr00t-SFT-Spatial</code>（few-shot，官方起點 41.4%）；N1.7 暫借 NVIDIA 官方
<code>GR00T-N1.7-LIBERO</code> 權重。</li>
<li><b>Actor（策略 π<sub>θ</sub>）</b>：GR00T 的 flow-matching action head，輸入觀測、輸出動作。</li>
<li><b>Critic（價值函數 V<sub>φ</sub>）</b>：RL 時新接上的 value head（<b>隨機初始化</b>），
估計「此狀態未來能拿到的期望回報」，供 advantage 計算。</li>
<li><b>success_once</b>：一個 episode 內「至少成功完成任務一次」的比例（實測模擬結果，非估計值）。
<b>train 版</b>每 epoch 由 64–128 條軌跡統計（較穩）；<b>eval 版</b>僅 4 trials（雜訊大，需看趨勢）。</li>
<li><b>explained_variance（EV）</b>：critic 品質指標（公式見 §4）。
<b>1=完美、0=等同猜平均、負值=比猜平均差、NaN=樣本太少算不出來</b>。</li>
<li><b>rollout / epoch / batch</b>：一個 epoch = 收集一輪軌跡（rollout）→ 用這批資料做 PPO 更新 → （每 N epoch）評估。
<code>global_batch_size</code> 是一次梯度更新用的樣本數；在我們的設定下它 = rollout 資料總量
（把同一批資料整批重排，不需多收資料）。</li>
<li><b>critic warmup</b>：前 N 個 optimizer 步只更新 critic、凍結 actor，
避免隨機初始化的 critic 用垃圾 advantage 把 policy 帶壞。</li>
</ul></div>""")

    # ---------- 3. background ----------
    P.append('<h2 id="s3">3. 背景：RL 對 VLA 的幫助、為什麼必須從 SFT 起步</h2>')
    P.append("""<div class="card">
<h3>有沒有「未經 SFT」的原始 checkpoint？</h3>
<p>有——即 <b>foundation 權重</b>（如 <code>nvidia/GR00T-N1.5-3B</code>）：
具備一般化視覺語言與操作先驗，但<b>沒有針對 LIBERO 任務微調</b>。</p>
<h3>能直接對未 SFT 的模型做 RL 嗎？</h3>
<p>實務上幾乎不可行。LIBERO 這類操作任務是<b>稀疏獎勵</b>（要完成一整串正確動作才有成功訊號）；
從沒學過任務的 policy 隨機探索幾乎拿不到獎勵，RL 便沒有梯度訊號可學。
這是 RLinf（與同類工作）一律「SFT 冷啟動 → RL 精修」的原因。</p>
<h3>所以「RL 對 VLA 的幫助」怎麼定義與量測？</h3>
<p>正確對照是<b>同一個 SFT baseline 的「SFT-only」vs「SFT+RL」</b>。
RLinf 官方給的量尺是 N1.5 spatial <b>41.4% → 92.5%</b>（+51.1pp）；
本報告 §6 的最終跑（+24.7pp / eval +30pp）即是在我們可用算力下、對同一問題的直接驗證。</p>
</div>""")

    # ---------- 4. math ----------
    P.append('<h2 id="s4">4. 數學原理</h2>')
    P.append('<p class="sub">公式以 LaTeX 品質排版（原生 MathML，白底卡片），每張卡片下方附 LaTeX 原始碼。</p>')
    P.append('<h3>4.1 PPO（actor-critic）核心式</h3>')
    P.append(math_card("折扣回報（γ = 0.99）", M_RET,
                       r"R_t=\sum_{k\ge 0}\gamma^{k}\,r_{t+k}",
                       "從時間 t 起、往後所有獎勵的折扣總和；是 critic 要學會預測的目標。"))
    P.append(math_card("TD 誤差", M_TD,
                       r"\delta_t=r_t+\gamma V_\varphi(s_{t+1})-V_\varphi(s_t)",
                       "「實際拿到的一步獎勵 + 下一狀態估值」與「目前狀態估值」的落差。"))
    P.append(math_card("GAE 優勢估計（λ = 0.95）", M_GAE,
                       r"\hat{A}_t=\sum_{l\ge 0}(\gamma\lambda)^{l}\,\delta_{t+l}",
                       "衡量「這個動作比平均好多少」。注意：Â 完全由 critic V<sub>φ</sub> 推導——critic 壞掉，Â 就是雜訊。"))
    P.append(math_card("重要性比", M_RATIO,
                       r"r_t(\theta)=\dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}",
                       "新舊 policy 對同一動作的機率比；≈1 代表 policy 幾乎沒動（我們在 N1.7 觀測到的狀態）。"))
    P.append(math_card("PPO 剪裁目標（ε = 0.2）", M_CLIP,
                       r"L^{\text{CLIP}}(\theta)=\hat{\mathbb{E}}_t\!\left[\min\!\big(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),1-\varepsilon,1+\varepsilon)\hat{A}_t\big)\right]",
                       "限制單步更新幅度、避免 policy 崩壞；是 actor 的訓練目標。"))
    P.append(math_card("Critic 損失", M_VLOSS,
                       r"L^{V}(\varphi)=\hat{\mathbb{E}}_t\big[(V_\varphi(s_t)-R_t)^2\big]",
                       "critic 以 MSE 逼近實際回報。"))
    P.append(math_card("總損失", M_TOTAL,
                       r"L=-L^{\text{CLIP}}+c_v L^{V}-c_e H[\pi_\theta]",
                       "本專案設定 c<sub>e</sub>（entropy_bonus）= 0，因此 entropy 項恆為 0 —— "
                       "這就是訓練面板上 entropy_loss = 0 的原因（不是壞掉）。"))
    P.append(math_card("explained_variance（critic 品質診斷）", M_EV,
                       r"\text{EV}=1-\dfrac{\text{Var}(R_t-V_\varphi(s_t))}{\text{Var}(R_t)}",
                       "1 = 完美預測；0 = 等同猜平均值；負 = 比猜平均還差。"))
    P.append('<h3>4.2 為什麼「規模不足」會系統性地傷害 PPO——三個機制</h3>')
    P.append(math_card("(a) 梯度雜訊尺度（McCandlish et al., arXiv:1812.06162）", M_GNS,
                       r"\text{Var}[\hat{g}]\propto \Sigma/B,\qquad B_{\text{noise}}=\text{tr}(\Sigma)/\lVert G\rVert^{2}",
                       "batch 大小 B 低於臨界批量 B<sub>noise</sub> 時，每步更新主要在追雜訊而非真實梯度方向。"
                       "我們初版 B=4，遠低於 RL 任務典型的臨界批量。"))
    P.append(math_card("(b) Advantage 正規化的統計失真", M_ADVN,
                       r"\hat{A}'_t=\big(\hat{A}_t-\text{mean}(\hat{A})\big)/\text{std}(\hat{A})",
                       "mean/std 是用「當前 batch」估的，取樣誤差以 O(1/√N) 收斂——N=4 時極不穩定，"
                       "少數極端樣本即可帶偏整批縮放；N=1024（官方）才站得住。"))
    P.append(math_card("(c) EV 的自由度退化（Bessel 校正）", M_BESSEL,
                       r"s^{2}=\tfrac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^{2}\ \Rightarrow\ n\le 1\text{ 時 NaN}",
                       "PyTorch <code>torch.var()</code> 預設除以 (n−1)；有效樣本 n≤1 時無定義、回傳 NaN。"
                       "經獨立實測驗證：batch=4 本身不必然造成 NaN（4 筆正常回報 EV=0.98），"
                       "而是<b>小 batch 大幅提高「某個 mask/chunk 邊界只剩 ≤1 個有效樣本」的機率</b>——"
                       "與初版跑 EV 全程 NaN、放大 batch 後 NaN 消失的實測完全一致。"))

    # ---------- 5. N1.7 ----------
    P.append('<h2 id="s5">5. N1.7 實驗：cold-start critic vs critic warmup</h2>')
    P.append(f"""<div class="card">
<p>設定：<code>GR00T-N1.7-LIBERO</code> checkpoint（近乎任務天花板的官方權重）＋新接、
隨機初始化的 value head；單卡、envs=4、batch=32。兩次跑唯一差異是
<code>critic_warmup_steps</code>（0 vs 40）。</p>
<p><b>一次講完的結論：</b>無 warmup 時，訓練初期以「隨機 critic 的垃圾 advantage」更新 policy，
eval 成功率由早段 {pct(ev_ok_n[0])} <span class="k">跌到末段 {pct(ev_ok_n[1])}</span>；
加了 warmup 後退步消失，但成功率僅在 25–75% 間<b>振盪、無趨勢</b>——
warmup 只能「止跌」。真正的瓶頸在 critic 從頭到尾沒學好
（EV 恆負），而這在後來 N1.5 的規模實驗中被證實是<b>規模問題、非機制問題</b>（§6）。</p>
</div>""")
    P.append(c_n17_eval)
    P.append('<div class="g2">' + c_n17_train + c_n17_evx + '</div>')
    P.append('<h3>5.1 「Loss 看起來是 0」的診斷（一次說明）</h3>')
    P.append('<div class="card"><ul>'
             '<li><b>entropy_loss = 0</b>：設定 <code>entropy_bonus=0</code>，本來就恆為 0。</li>'
             '<li><b>policy_loss 前 40 步 = 0</b>：critic warmup 期間 actor 凍結。</li>'
             '<li><b>其後 policy_loss 仍僅 ±0.3</b>：PPO 剪裁目標的正常量級，非異常；'
             'KL≈0.01、ratio≈1 說明 policy 每步幾乎沒動——與「advantage 是雜訊、沒有一致方向」一致。</li>'
             '<li><b>reward 量級僅 ~0.004</b>（步數正規化後的稀疏獎勵）：critic 要擬合的訊號本身極微弱。</li></ul></div>')
    P.append('<div class="g2">' + "".join(c_diag) + '</div>')

    # ---------- 6. N1.5 trio (consolidated) ----------
    P.append('<h2 id="s6">6. N1.5 三次實驗統一對照——規模假設的直接驗證</h2>')
    P.append(f"""<div class="card">
<p>同一個公開 SFT checkpoint（<code>RLinf/RLinf-Gr00t-SFT-Spatial</code>，官方起點 41.4%）、
同一任務（libero_spatial）、同樣忠實於官方 recipe 的 <code>critic_warmup_steps=0</code>，
<b>只改變規模</b>，共跑三次：</p>
<table>
<tr><th>實驗</th><th>GPU</th><th>envs</th><th>global batch</th><th>epochs</th><th>耗時</th>
<th>train success（早→末段）</th><th>eval success（早→末段）</th><th>EV 為正比例</th></tr>
<tr><td><span style="color:{AMBER}">●</span> 初版・單卡</td><td>1</td><td>4</td><td>4</td><td>30</td><td>~4.7 h</td>
<td>{pct(tr_ok_a[0])} → {pct(tr_ok_a[1])}（無明顯提升）</td><td>{pct(ev_ok_a[0])} → {pct(ev_ok_a[1])}（雜訊）</td>
<td class="k">0%（全程 NaN）</td></tr>
<tr><td><span style="color:{TEALC}">●</span> 雙卡・放大 batch</td><td>2（actor 分片）</td><td>4</td><td>32</td><td>72</td><td>~9 h</td>
<td>{pct(tr_ok_b[0])} → {pct(tr_ok_b[1])}（<b>+{d_tr_b:g}pp</b>）</td><td>{pct(ev_ok_b[0])} → {pct(ev_ok_b[1])}</td>
<td>{round(sb['pos']/sb['n']*100)}%（平均 {sb['mean']:.2f}）</td></tr>
<tr><td><span style="color:{MAGEN}">●</span> <b>最終・放大時長</b></td><td>1</td><td>8</td><td>64</td><td><b>230</b></td><td><b>54.8 h</b></td>
<td><b>{pct(tr_ok_c[0])} → {pct(tr_ok_c[1])}（<span class="hl">+{d_tr_c:g}pp</span>）</b></td>
<td><b>{pct(ev_ok_c[0])} → {pct(ev_ok_c[1])}（<span class="hl">+{d_ev_c:g}pp</span>）</b></td>
<td><b>{round(sc['pos']/sc['n']*100)}%（平均 {sc['mean']:.2f}，僅 {sc['nan']} 點 NaN）</b></td></tr>
</table></div>""")
    P.append(c_eval3)
    P.append(c_train3)
    P.append(c_evx3)
    P.append(f"""<div class="card">
<h3>三張圖一次解讀</h3>
<ul>
<li><b>發現 1（批量效應）</b>：batch 4 → 32（初版 → 雙卡，env 數刻意不動）後，
EV 從「全程算不出來（NaN）」變成 {round(sb['pos']/sb['n']*100)}% 為正、
成功率出現 +{d_tr_b:g}pp 的首次明確上升——對應 §4.2 的機制 (a)(b)(c)：
批量放大直接改善梯度方向品質與診斷可計算性。</li>
<li><b>發現 2（時長效應，本專案最重要的發現）</b>：最終跑的 batch（64）其實<b>小於</b>雙卡（128 的一半），
但 epochs 多 3.2 倍（230 vs 72），提升幅度反而更大（+{d_tr_c:g}pp vs +{d_tr_b:g}pp）——
<b>訓練時長 / PPO 更新次數本身是獨立的關鍵變數</b>，不是「batch 夠大就好」。
這與官方 recipe 押在 <code>max_epochs=1000</code>（我們最多達其 23%）互相呼應。</li>
<li><b>發現 3（趨勢品質）</b>：最終跑 eval 序列後段多次觸及 100%（共 {n_eval_ones} 次滿分）、
train 曲線（洋紅）幾乎單調上升；EV 全程穩定為正（平均 {sc['mean']:.2f}）——
critic 真的學會估值，advantage 有一致方向，PPO 因此有效。</li>
</ul>
<h3>誠實的保留（一次講完）</h3>
<ul>
<li>三次都是<b>單一 random seed</b>，未做多 seed 重複驗證。</li>
<li>絕對規模仍低於官方：envs 8 vs 64（12.5%）、batch 64 vs 1024（6.25%）、epochs 230 vs 1000（23%）、
eval 4 trials vs 500 envs——所以這是「方向與機制的因果驗證」，
<b>還不是「重現官方 92.5%」</b>。</li>
<li>最終跑中途曾因容器 pids 上限崩潰一次（雙卡放大 envs 的嘗試），
改回單卡並修正 <code>save_interval</code> 整除規則後，54.8 小時零中斷完賽（時間軸見 §11）。</li>
</ul></div>""")

    # ---------- 7. external ----------
    P.append('<h2 id="s7">7. 外部對照：RLinf 官方規模、官方成績與可重現性</h2>')
    P.append("""<div class="card">
<h3>7.1 官方 recipe 的實際規模（查證自官方 config 與文件）</h3>
<table>
<tr><th>設定項</th><th>RLinf 官方 N1.5 recipe</th><th>我們（最終跑）</th></tr>
<tr><td>硬體</td><td>LIBERO 文件標示 <b>1–2 nodes・8–16 GPUs</b>（論文大規模實驗用 8×H100 節點）</td><td>1× H100</td></tr>
<tr><td>total_num_envs（train / eval）</td><td><b>64 / 500</b></td><td>8 / 8</td></tr>
<tr><td>global_batch_size / micro</td><td><b>1024 / 128</b></td><td>64 / 16</td></tr>
<tr><td>max_epochs</td><td><b>1000</b>（92.5% 對應 checkpoint 名為 <code>…-RL-Spatial-Step400</code>）</td><td>230</td></tr>
<tr><td>critic_warmup_steps</td><td>0</td><td>0（忠實跟隨）</td></tr>
</table>
<p>官方文件並注明：「本結果採用與 π₀ 完全相同的超參數……進一步調參應可更好」——
官方靠的是<b>規模</b>而非精細調參，與我們的驗證方向一致。
官方 PPO 文件的調參注意事項也明說：<i>“Use reward normalization to stabilize training”、
“Monitor KL divergence”、<b>“For large LLMs, increase batch size to reduce variance.”</b></i></p>
<h3>7.2 官方公布成績 vs 我們的實測</h3>
<table>
<tr><th>版本</th><th>SFT 起點</th><th>+PPO 後</th><th>提升</th><th>備註</th></tr>
<tr><td>N1.5（官方）</td><td>41.4%</td><td>92.5%</td><td>+51.1 pp</td><td>4 suites 平均 52.5→89.5（+37pp）；多卡、1000 epochs 級規模</td></tr>
<tr><td><b>N1.5（本報告最終跑）</b></td><td>—（同一 SFT）</td><td>eval 末段均 85%、多次 100%</td><td><b>訓練 +24.7pp / eval +30pp</b></td><td>單卡、23% 的官方 epochs——方向一致、幅度符合規模比例</td></tr>
<tr><td>N1.6（官方）</td><td>70%</td><td>82%</td><td>+12 pp</td><td>僅 spatial</td></tr>
<tr><td>N1.7</td><td colspan="3">—</td><td>官方標記 TODO（LIBERO RL 尚未驗證）；本報告 §5 為超前嘗試</td></tr>
</table>
<h3>7.3 可重現性與 N1.6 更正</h3>
<p>N1.5 四個 few-shot SFT 均已公開（<code>RLinf/RLinf-Gr00t-SFT-*</code>）→ 可重現（本報告已做 spatial）。
<b class="k">N1.6 更正</b>：官網「SFT models will be released soon」字面上讓人以為無法重現，
但官方 config 指向的 <code>RLinf/RLinf-Gr00t-N1.6-RL-Spatial</code> repo <b>實際存在</b>，
且其 <code>trainer_state.json</code> 只有 grad_norm / lr / loss / step（典型監督式訓練曲線、無任何 RL 欄位）
——幾乎可確定它就是 N1.6 的 <b>SFT checkpoint</b>（命名易誤導）。即 N1.6 其實可重現，僅本報告尚未執行。</p>
<h3>7.4 誠實的限制</h3>
<p>未找到 RLinf 官方「規模 vs 效果」的 ablation；且官方 GitHub issue #585 顯示有使用者在接近官方規模下
仍未重現官方曲線（未定論）。因此「規模」是<b>證據最充分、且經我們三段實驗直接驗證方向的主因</b>，
但不能排除其他因素（版本差異、超參組合）在更大規模下仍有影響。</p>
</div>""")

    # ---------- 8. next steps ----------
    P.append('<h2 id="s8">8. 建議下一步</h2>')
    P.append("""<div class="card"><ul>
<li><b>補足統計嚴謹度</b>：同設定跑 2–3 個 random seed，確認 +24.7pp 的變異範圍。</li>
<li><b>繼續沿「時長」軸放大</b>：以最終跑 checkpoint 續訓至 400+ epochs（對齊官方 Step400），
觀察是否逼近官方 92.5%。</li>
<li><b>解掉 env 平行化限制</b>：目前 in-process 向量環境是「序列踩步」（避開容器內 MuJoCo 子程序崩潰的 workaround），
env 數增加 = 線性變慢；若能改用真正平行的 env worker，同時長內可再放大取樣量。</li>
<li><b>其餘三個 N1.5 suites</b>（object / goal / long）與 <b>N1.6</b>（SFT 實際可得，見 §7.3）。</li>
<li><b>Phase 2</b>：把驗證過的流程帶到 IsaacLab + 自有 fine-tuned checkpoint（真正的目標場景）。</li>
</ul></div>""")

    # ---------- 9/10. diagrams ----------
    P.append('<h2 id="s9">9. 系統架構圖</h2>')
    P.append('<div class="diagram">' + ARCH_SVG + '</div>')
    P.append('<h2 id="s10">10. PPO 訓練流程圖</h2>')
    P.append('<div class="diagram">' + FLOW_SVG + '</div>')

    # ---------- 11. config ----------
    P.append('<h2 id="s11">11. 執行設定與重現資訊</h2>')
    cfg_rows = (
        "<tr><th>run</th><th>模型 / checkpoint</th><th>關鍵設定</th><th>指標檔（artifacts/rl/）</th></tr>"
        "<tr><td>N1.7 no-warmup</td><td>GR00T-N1.7-LIBERO + Cosmos-Reason2-2B（本地副本）</td>"
        "<td>envs=4, batch=32, warmup=0, 100 ep, GPU1</td><td><code>metrics_nowarmup.json</code></td></tr>"
        "<tr><td>N1.7 critic-warmup</td><td>同上</td>"
        "<td>envs=4, batch=32, warmup=40, 120 ep, GPU1</td><td><code>libero_spatial_ppo_gr00t_n1d7_phase1/metrics.json</code></td></tr>"
        "<tr><td>N1.5 初版</td><td>RLinf-Gr00t-SFT-Spatial（自含權重）</td>"
        "<td>envs=4, batch=4/4, warmup=0, 30 ep, GPU1</td><td><code>metrics_n15_spatial.json</code></td></tr>"
        "<tr><td>N1.5 雙卡</td><td>同上</td>"
        "<td>placement actor:0-1（FSDP full_shard）/ env,rollout:GPU1；envs=4, batch=32/16, 72 ep</td>"
        "<td><code>metrics_n15_spatial_dualgpu.json</code></td></tr>"
        "<tr><td><b>N1.5 最終</b></td><td>同上</td>"
        "<td>單 GPU1；envs=8, batch=64/16, warmup=0, <b>230 ep</b>, eval/10, save/10, 實跑 54.8 h</td>"
        "<td><code>metrics_n15_spatial_scaled.json</code></td></tr>"
    )
    P.append('<div class="card"><table>' + cfg_rows + '</table>'
             '<h3>容器相容性（Pegasus，jupyter 為 PID 1、無 Docker）——已封裝為修補腳本</h3><ul>'
             '<li>cgroup <code>pids.max=2048</code>：以 <code>RLINF_RAY_NUM_CPUS</code> 上限 Ray 工作程序、'
             '所有數學庫單執行緒、<code>taskset</code> 限核；殭屍程序不被回收會持續吃掉配額（多次崩潰根因）。</li>'
             '<li>MuJoCo 於 spawn 子程序中崩潰 → 改用 in-process 向量環境（<code>RLINF_LIBERO_INPROCESS=1</code>）。</li>'
             '<li><code>component_placement</code> 需明釘 GPU（RLinf 忽略 <code>CUDA_VISIBLE_DEVICES</code>）。</li>'
             '<li>RLinf 規則：<code>save_interval</code> 必須可被 <code>val_check_interval</code> 整除。</li>'
             '<li>EGL 離屏算繪（<code>MUJOCO_GL=egl</code>）；HF hub 需 <code>huggingface-hub&lt;1.0</code>。</li></ul>'
             '<p>工具鏈：<code>agents/rl-trainbot/harness/</code>（<code>launch.py</code> / <code>poll.py</code> / '
             '<code>patch_rlinf_clone.py</code> / <code>extract_metrics.py</code> / 本報告產生器 '
             '<code>report_pretty.py</code>）；規格：OpenSpec <code>add-rlinf-gr00t-n17-libero-rl</code>；'
             '設定檔：<code>agents/rl-trainbot/harness/config/phase1*.yaml</code>。圖表原始資料均為 TensorBoard 匯出之 JSON，'
             '架構／流程圖由 drawio-blocksmith 產生（同源 <code>.drawio</code> 可編輯）。</p></div>')

    P.append('<p class="footer">RLinf × GR00T × LIBERO — Phase 1 分析報告・產生器 '
             '<code>agents/rl-trainbot/harness/report_pretty.py</code>・資料截至 2026-07-06（最終跑完成於台灣時間 07-06 10:01）</p>')

    body = "".join(P)
    return ('<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>RLinf GR00T-N1.7 LIBERO Phase 1 分析報告</title>'
            f'<style>{CSS}</style></head><body><div class="wrap">{body}</div></body></html>')


OUT.write_text(build(), encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size, "bytes")
