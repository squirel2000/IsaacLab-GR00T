#!/usr/bin/env python3
"""Render the training monitor page: one tabbed template, any model.

Nothing here knows about RLDX-1 or GR00T specifically — every label, number and note comes
from the RunSpec registry in runs.py, so adding a backend needs no changes to this file.

Emits a single self-contained HTML file (no external requests) with an overview tab plus one
tab per run, each showing live scalars, a loss sparkline drawn from the parsed metrics, and
the filtered log tail.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timedelta, timezone

TPE = timezone(timedelta(hours=8))

CSS = """
:root{
  --bg:#0a0c10; --panel:#12171f; --panel2:#161c26; --line:#222b38; --line2:#2e3a4b;
  --ink:#e8edf4; --ink2:#9aa8bb; --ink3:#63718a;
  --amber:#ffb020; --sig:#5fd68a; --cyan:#48c4dc; --red:#ff5f5f; --violet:#9d8cff;
  --mono:'Cascadia Mono',Consolas,'SF Mono',ui-monospace,monospace;
  --disp:'Saira Condensed','Oswald','Arial Narrow',system-ui,sans-serif;
  --sans:'IBM Plex Sans','Segoe UI',system-ui,sans-serif;
}
*{box-sizing:border-box} html,body{margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);
 background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),
                  linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);
 background-size:44px 44px,44px 44px;}
.wrap{max-width:1280px;margin:0 auto;padding:22px 22px 60px}
header{border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:16px;
 display:flex;align-items:flex-end;gap:18px;flex-wrap:wrap}
.eyebrow{font-family:var(--mono);font-size:10.5px;letter-spacing:.24em;text-transform:uppercase;color:var(--amber)}
h1{font-family:var(--disp);font-weight:700;font-size:clamp(26px,3.6vw,40px);line-height:.96;
 margin:2px 0 0;text-transform:uppercase}
.sub{font-family:var(--mono);font-size:11px;color:var(--ink3);margin-top:6px}
.spacer{margin-left:auto}
.pill{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;
 padding:4px 9px;border:1px solid var(--line2);border-radius:2px;color:var(--ink2);white-space:nowrap}
.pill.running{color:var(--amber);border-color:var(--amber)}
.pill.finished{color:var(--sig);border-color:var(--sig)}
.pill.pending{color:var(--ink3)}
.pill.error{color:var(--red);border-color:var(--red)}
/* tabs */
nav.tabs{display:flex;gap:2px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin-bottom:16px}
nav.tabs button{font-family:var(--mono);font-size:11px;letter-spacing:.06em;background:transparent;
 color:var(--ink3);border:1px solid transparent;border-bottom:none;padding:9px 14px;cursor:pointer;
 border-radius:2px 2px 0 0}
nav.tabs button:hover{color:var(--ink2)}
nav.tabs button[aria-selected=true]{color:var(--ink);background:var(--panel);
 border-color:var(--line);border-bottom:1px solid var(--panel);margin-bottom:-1px}
nav.tabs .fam{font-family:var(--mono);font-size:9px;color:var(--ink3);align-self:center;
 padding:0 6px 0 12px;letter-spacing:.14em;text-transform:uppercase}
section[role=tabpanel][hidden]{display:none}
.card{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);
 border-radius:3px;padding:16px 18px;margin-bottom:12px}
.card h2{font-family:var(--disp);font-size:21px;font-weight:500;text-transform:uppercase;margin:0 0 4px}
.meta{font-family:var(--mono);font-size:11px;color:var(--ink3);line-height:1.7;margin-bottom:12px}
.note{font-size:13px;color:var(--ink2);line-height:1.65;border-left:2px solid var(--line2);
 padding-left:11px;margin:10px 0 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(108px,1fr));gap:14px;
 font-family:var(--mono);font-variant-numeric:tabular-nums}
.k{font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3)}
.v{font-size:21px;font-weight:600;margin-top:3px;line-height:1.1}
.v small{font-size:11px;color:var(--ink3);font-weight:400}
.bar{height:6px;background:#0a0c10;border:1px solid var(--line);margin-top:13px;overflow:hidden}
.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--amber),var(--sig));transition:width .6s}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12.5px;
 font-variant-numeric:tabular-nums}
th{text-align:right;padding:0 10px 8px;font-size:9px;letter-spacing:.13em;text-transform:uppercase;
 color:var(--ink3);border-bottom:1px solid var(--line2);font-weight:600}
th:first-child{text-align:left}
td{text-align:right;padding:9px 10px;border-bottom:1px solid var(--line)}
td:first-child{text-align:left;font-family:var(--sans);font-size:13px}
tr:last-child td{border-bottom:0}
pre{margin:0;font-family:var(--mono);font-size:10.5px;line-height:1.6;color:var(--ink2);
 background:#0a0c10;border:1px solid var(--line);padding:10px 12px;max-height:260px;
 overflow:auto;white-space:pre-wrap;word-break:break-word}
svg.spark{display:block;width:100%;margin-top:12px}
.metric-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
 gap:16px;margin-top:14px;padding-top:14px;border-top:1px solid var(--line)}
footer{border-top:1px solid var(--line);margin-top:20px;padding-top:12px;
 font-family:var(--mono);font-size:10.5px;color:var(--ink3);line-height:1.8}
.stale{color:var(--red)}
"""

JS = """
const tabs=[...document.querySelectorAll('nav.tabs button')];
const panels=[...document.querySelectorAll('section[role=tabpanel]')];
function show(id){
  tabs.forEach(t=>t.setAttribute('aria-selected', String(t.dataset.tab===id)));
  panels.forEach(p=>{p.hidden = p.id!=='panel-'+id});
  try{localStorage.setItem('rldx-monitor-tab',id)}catch(e){}
}
tabs.forEach(t=>t.addEventListener('click',()=>show(t.dataset.tab)));
let want=null; try{want=localStorage.getItem('rldx-monitor-tab')}catch(e){}
show(tabs.some(t=>t.dataset.tab===want)?want:tabs[0].dataset.tab);
"""


def _fmt_hms(total_sec: int | float | None) -> str | None:
    """Compact duration for display, e.g. 34532 -> '9h35m'. None in, None out."""
    if total_sec is None:
        return None
    h, rem = divmod(int(total_sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _fmt(v, spec="", dash="—"):
    if v is None or v == "":
        return dash
    try:
        return format(v, spec) if spec else str(v)
    except (TypeError, ValueError):
        return str(v)


def _downsample(pts: list[tuple[int, float]], max_points: int = 400) -> list[tuple[int, float]]:
    """Shrink a long (step, value) series for SVG rendering without hiding spikes.

    A run logging every 10 steps over 30k steps is ~3000 points — fine for wandb's own charting
    library, too many for a hand-rolled polyline to stay both fast and legible. Uniform-stride
    sampling would skip straight over a single-step spike in grad_norm; bucketing and keeping
    each bucket's min AND max VALUE (with its own step, so the x-axis stays truthful) instead
    keeps the outlier visible at the cost of one extra point per bucket.
    """
    n = len(pts)
    if n <= max_points:
        return pts
    buckets = max(1, max_points // 2)
    size = n / buckets
    out: list[tuple[int, float]] = []
    for i in range(buckets):
        chunk = pts[int(i * size):int((i + 1) * size)] or [pts[min(int(i * size), n - 1)]]
        lo_pt = min(chunk, key=lambda p: p[1])
        hi_pt = max(chunk, key=lambda p: p[1])
        out.extend([lo_pt, hi_pt] if chunk.index(lo_pt) <= chunk.index(hi_pt) else [hi_pt, lo_pt])
    return out


def _fmt_axis(v: float, log_scale: bool) -> str:
    """Compact numeric label for an axis tick — real units even when the chart itself is log-scaled."""
    if log_scale:
        v = 10 ** v
    if v != 0 and abs(v) < 1e-3:
        return f"{v:.1e}"
    return f"{v:.4g}"


def _sparkline(points: list[tuple[int, float]], colour: str = "#5fd68a", *, log_scale: bool = False,
              label: str = "", height: int = 120) -> str:
    """A metric series as inline SVG with real axis labels — small-multiples like wandb's run
    panels, but wandb's own axes are the whole reason those are readable, so this draws its own:
    3 horizontal gridlines with the metric's actual value (unlogged even when the chart is log-
    scaled) on the y-axis, and the step number at the start/middle/end on the x-axis.

    ``points`` is ``[(step, value), ...]``, chronological. Log scale suits loss (spans orders of
    magnitude); grad_norm/lr read better linear, where a spike or a scheduler step is visible as
    shape rather than compressed away. The full history is passed in (poller.py no longer
    truncates it) and downsampled here for rendering, so the chart covers the whole run the way
    wandb's would, not just a recent window.
    """
    pts = [(s, v) for s, v in points if isinstance(v, (int, float)) and (not log_scale or v > 0)]
    total_pts = len(pts)
    if total_pts < 2:
        return ""
    true_first, true_last = pts[0][1], pts[-1][1]  # before downsampling reorders bucket contents
    true_first_step, true_last_step = pts[0][0], pts[-1][0]
    pts = _downsample(pts)
    import math
    steps = [p[0] for p in pts]
    ys = [math.log10(v) for _, v in pts] if log_scale else [v for _, v in pts]
    lo, hi = min(ys), max(ys)
    rng = (hi - lo) or 1.0
    n = len(ys)

    # Canvas has real margins for axis text, unlike a bare 0-100 box — text drawn at that scale
    # would either be illegibly tiny or distort under preserveAspectRatio='none'.
    W, H, PAD_L, PAD_R, PAD_T, PAD_B = 320, 130, 50, 10, 10, 22
    plot_w, plot_h = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def x_of(i: int) -> float:
        return PAD_L + i / (n - 1) * plot_w

    def y_of(y: float) -> float:
        return PAD_T + (1 - (y - lo) / rng) * plot_h

    coords = " ".join(f"{x_of(i):.1f},{y_of(y):.1f}" for i, y in enumerate(ys))

    y_ticks = sorted({hi, (hi + lo) / 2, lo})
    y_axis = "".join(
        f"<line x1='{PAD_L}' y1='{y_of(t):.1f}' x2='{W - PAD_R}' y2='{y_of(t):.1f}' "
        f"stroke='#222b38' stroke-width='1'/>"
        f"<text x='{PAD_L - 6}' y='{y_of(t) + 3:.1f}' text-anchor='end' font-size='11' "
        f"fill='#9aa8bb'>{_fmt_axis(t, log_scale)}</text>"
        for t in y_ticks)
    # The two end labels use the TRUE first/last step, not steps[0]/steps[-1] — downsampling
    # keeps each bucket's min/max VALUE and orders the pair by which occurs first, so the point
    # that lands at index 0 or n-1 is not guaranteed to be the chronologically first/last one.
    # An earlier version showed e.g. 29,850 instead of 30,000 for a finished run, making the
    # chart look like it covered less of the run than it actually did.
    x_idx = sorted({0, n // 2, n - 1})
    x_labels = {0: f"{true_first_step:,}", n - 1: f"{true_last_step:,}"}
    x_axis = "".join(
        f"<text x='{x_of(i):.1f}' y='{H - 5}' "
        f"text-anchor='{'start' if i == 0 else 'end' if i == n - 1 else 'middle'}' "
        f"font-size='11' fill='#9aa8bb'>{x_labels.get(i, f'{steps[i]:,}')}</text>"
        for i in x_idx)

    scale_note = "log scale" if log_scale else "linear"
    return (
        f"<svg class='spark' viewBox='0 0 {W} {H}' preserveAspectRatio='xMidYMid meet' "
        f"role='img' style='height:{height}px' "
        f"aria-label='{html.escape(label)} vs step, {scale_note}, {true_first:.4g} to {true_last:.4g}'>"
        f"{y_axis}{x_axis}"
        f"<polyline points='{coords}' fill='none' stroke='{colour}' stroke-width='1.6' "
        f"vector-effect='non-scaling-stroke'/></svg>"
        f"<div class='k' style='margin-top:4px'>{html.escape(label)} vs step, {scale_note} "
        f"&middot; {true_first:.4g} &rarr; {true_last:.4g} over {total_pts} pts"
        + (f" ({n} rendered)" if total_pts != n else "") + "</div>")


def _metric_row(st: dict) -> str:
    """Three small-multiple charts side by side: loss, grad_norm, lr — the wandb-panel layout."""
    cells = [
        ("loss", st.get("loss_series") or [], "#5fd68a", True),
        ("grad norm", st.get("grad_norm_series") or [], "#48c4dc", False),
        ("learning rate", st.get("lr_series") or [], "#9d8cff", False),
    ]
    parts = []
    for label, series, colour, log_scale in cells:
        spark = _sparkline(series, colour, log_scale=log_scale, label=label, height=120)
        if spark:
            parts.append(f"<div>{spark}</div>")
    if not parts:
        return ""
    return f"<div class='metric-row'>{''.join(parts)}</div>"


def _state_class(state: str) -> str:
    s = state.lower()
    if "running" in s:
        return "running"
    if "finish" in s:
        return "finished"
    if "error" in s or "unknown" in s:
        return "error"
    return "pending"


def _overview(statuses: list[dict]) -> str:
    rows = []
    for st in statuses:
        pct = st.get("pct") or 0
        finish = html.escape(st.get("eta_wall") or ("complete" if pct >= 100 else "—"))
        rows.append(
            f"<tr><td>{html.escape(st['label'])}</td>"
            f"<td><span class='pill {_state_class(st['state'])}'>{html.escape(st['state'])}</span></td>"
            f"<td>{_fmt(st.get('step'), ',')}<small style='color:#63718a'>/{st['max_steps']:,}</small></td>"
            f"<td>{pct}%</td>"
            f"<td>{_fmt(st.get('loss'), '.4f')}</td>"
            f"<td>{finish}</td>"
            f"<td>{st['gpus']}</td>"
            f"<td>{st['effective_batch']}</td>"
            f"<td style='text-align:left;font-family:var(--mono);font-size:11px'>"
            f"{html.escape(st['action_space'])}</td></tr>")
    return (
        "<div class='card'><h2>All runs</h2>"
        "<div class='meta'>Effective batch is held equal across runs so the comparison is on "
        "equal footing. Loss is <b>not</b> comparable between different action-space sizes — "
        "judge those on evaluation success rate instead. Finish time is tqdm's own remaining-"
        "time estimate (accounts for the whole run, not just this page's sampling window).</div>"
        "<table><thead><tr><th>Run</th><th>State</th><th>Step</th><th>Progress</th><th>Loss</th>"
        "<th>Est. finish</th><th>GPU</th><th>Batch</th><th>Action space</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>")


def _panel(st: dict) -> str:
    pct = st.get("pct") or 0
    metrics = _metric_row(st)
    log = html.escape((st.get("log_tail") or "").strip() or "— no log yet —")
    elapsed = _fmt_hms(st.get("elapsed_sec"))
    remaining = "complete" if st["state"].lower() == "finished" else _fmt_hms(st.get("remaining_sec"))
    total_est = _fmt_hms(st.get("total_estimate_sec"))
    finish = st.get("eta_wall") or ("complete" if st["state"].lower() == "finished" else "—")
    return (
        f"<div class='card'><h2>{html.escape(st['label'])} "
        f"<span class='pill {_state_class(st['state'])}'>{html.escape(st['state'])}</span></h2>"
        f"<div class='meta'>{html.escape(st['model'])}<br>"
        f"{st['gpus']} GPU &middot; effective batch {st['effective_batch']} &middot; "
        f"{html.escape(st['tuning'])}<br>action {html.escape(st['action_space'])}</div>"
        f"<div class='grid'>"
        f"<div><div class='k'>step</div><div class='v'>{_fmt(st.get('step'), ',')}"
        f"<small>/{st['max_steps']:,}</small></div></div>"
        f"<div><div class='k'>loss</div><div class='v'>{_fmt(st.get('loss'), '.4f')}</div></div>"
        f"<div><div class='k'>grad norm</div><div class='v'>{_fmt(st.get('grad_norm'), '.3f')}</div></div>"
        f"<div><div class='k'>lr</div><div class='v'>{_fmt(st.get('lr'), '.2e')}</div></div>"
        f"<div><div class='k'>progress</div><div class='v'>{pct}%</div></div>"
        f"<div><div class='k'>rate</div><div class='v'><small>{html.escape(st.get('rate_desc') or '—')}</small></div></div>"
        f"<div><div class='k'>elapsed</div><div class='v'><small>{html.escape(elapsed or '—')}</small></div></div>"
        f"<div><div class='k'>remaining</div><div class='v'><small>{html.escape(remaining or '—')}</small></div></div>"
        f"<div><div class='k'>est. total</div><div class='v'><small>{html.escape(total_est or '—')}</small></div></div>"
        f"<div><div class='k'>est. finish</div><div class='v'><small>{html.escape(finish)}</small></div></div>"
        f"</div><div class='bar'><i style='width:{min(pct, 100)}%'></i></div>"
        f"{metrics}"
        + (f"<p class='note'>{html.escape(st['notes'])}</p>" if st.get("notes") else "")
        + f"<div class='k' style='margin:14px 0 6px'>log tail</div><pre>{log}</pre>"
        f"<div class='meta' style='margin:10px 0 0'>output: {html.escape(st['output'])}</div>"
        f"</div>")


def _host_panel(host_state: dict | None) -> str:
    """GPU occupancy for the whole box, ours and other people's — answers "what's on GPU N"
    without a fresh probe. Rides along on the same call the logs come from (see poller.py)."""
    gpus = (host_state or {}).get("gpus") or []
    if not gpus:
        return ""
    rows = []
    for g in sorted(gpus, key=lambda g: g["index"]):
        apps = g.get("apps") or []
        if apps:
            app_lines = "".join(
                f"<div>pid {a['pid']} &middot; {a['mem_gb']}GB &middot; up {a['elapsed']}<br>"
                f"<span style='color:var(--ink3)'>{html.escape(a['cmd'])}</span></div>"
                for a in apps)
        else:
            app_lines = "<div style='color:var(--ink3)'>(idle — no compute process)</div>"
        rows.append(
            f"<div><div class='k'>GPU{g['index']}</div>"
            f"<div class='v' style='font-size:15px'>{g['mem_gb']:.1f} / {g['total_gb']:.1f} GB "
            f"<small>&middot; util {g['util']}%</small></div>"
            f"<pre style='margin-top:6px;max-height:none'>{app_lines}</pre></div>")
    return (
        "<div class='card'><h2>Box state</h2>"
        "<div class='meta'>What is actually on GPU0/GPU1 right now, including processes that "
        "are not ours — this box is shared.</div>"
        f"<div class='grid' style='grid-template-columns:repeat(auto-fit,minmax(280px,1fr))'>"
        + "".join(rows) + "</div></div>")


def render(statuses: list[dict], *, host: str = "pegasus pa-jp-v1",
           poll_seconds: int = 90, host_state: dict | None = None) -> str:
    now = datetime.now(TPE).strftime("%Y-%m-%d %H:%M:%S")
    running = sum(1 for s in statuses if _state_class(s["state"]) == "running")
    done = sum(1 for s in statuses if _state_class(s["state"]) == "finished")

    tabs, panels = ["<button data-tab='overview' aria-selected='true'>Overview</button>"], []
    last_family = None
    for st in statuses:
        if st["family"] != last_family:
            # A grouping label, not a tab: it has no panel of its own, so it is deliberately
            # not a <button> and not clickable. The runs listed after it (below) are the tabs.
            tabs.append(f"<span class='fam' role='presentation' "
                        f"title='Model family — groups the run tabs after it; not clickable'>"
                        f"{html.escape(st['family'])}</span>")
            last_family = st["family"]
        tabs.append(f"<button data-tab='{st['id']}' aria-selected='false'>"
                    f"{html.escape(st['label'].split('·')[-1].strip())}</button>")
        panels.append(f"<section role='tabpanel' id='panel-{st['id']}' hidden>{_panel(st)}</section>")

    return (
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Training Monitor</title>"
        f"<meta http-equiv='refresh' content='{poll_seconds}'>"
        f"<style>{CSS}</style></head><body><div class='wrap'>"
        f"<header><div><div class='eyebrow'>Training monitor</div>"
        f"<h1>OpenArm O6 &middot; Can Sorting</h1>"
        f"<div class='sub'>{html.escape(host)} &middot; {len(statuses)} runs &middot; "
        f"{running} running, {done} finished &middot; metrics parsed by vla-trainbot&rsquo;s extractor</div></div>"
        f"<span class='pill spacer'>refreshed {now} TPE</span></header>"
        f"<nav class='tabs' role='tablist'>{''.join(tabs)}</nav>"
        f"<section role='tabpanel' id='panel-overview'>{_overview(statuses)}{_host_panel(host_state)}</section>"
        f"{''.join(panels)}"
        f"<footer>Training runs are detached on the training host via <code>setsid nohup</code>; "
        f"this page is a read-only view and stopping the poller does not affect them. "
        f"Judge liveness by this page&rsquo;s refresh stamp, not by the poller&rsquo;s console output. "
        f"W&amp;B reporting is disabled for these runs (<code>WANDB_MODE=disabled</code>, set to "
        f"avoid needing an API key / network egress on the headless box) — this page is the only "
        f"live view of them; metrics come from parsing the training log, same regexes as "
        f"vla-trainbot&rsquo;s own dashboard.</footer>"
        f"</div><script>{JS}</script></body></html>")


def write(statuses: list[dict], path, **kw) -> None:
    path.write_text(render(statuses, **kw), encoding="utf-8")
    (path.parent / "runs.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
