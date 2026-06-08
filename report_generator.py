"""Generate a self-contained, offline HTML report for a pipeline run.

Pulls together pipeline_state.json + logs/metrics.jsonl + config and renders
``report_<timestamp>.html`` with: a training summary, a loss-vs-step curve (Chart.js
inlined for offline use, with an inline-SVG fallback), deploy status, a manual-sim note,
the per-stage pipeline log, and environment info. Dark/light toggle + responsive layout;
all data and the chart library are embedded so the file can be shared and opened offline.

Pure builders (``build_context`` / ``build_html`` / ``_svg_line_chart``) take plain data
so they can be unit-tested without the filesystem.
"""
from __future__ import annotations

import datetime
import html
import json
from pathlib import Path

from pipeline_state import Stage, ORDER, PipelineState
from pipeline_logging import LOGS_DIR, get_logger

log = get_logger("report")

HERE = Path(__file__).resolve().parent
METRICS_PATH = LOGS_DIR / "metrics.jsonl"
CHARTJS_PATH = HERE / "vendor" / "chart.umd.min.js"


# --------------------------------------------------------------------------- #
#  Data assembly (pure)
# --------------------------------------------------------------------------- #
def load_metrics(path: Path | None = None) -> list[dict]:
    """Read logs/metrics.jsonl into a list of {step,total,loss,lr} dicts."""
    path = path or METRICS_PATH                  # resolve the global at call time
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _duration(rec: dict) -> str:
    """Human 'Hh Mm' between a stage record's started_at and finished_at."""
    a, b = rec.get("started_at"), rec.get("finished_at")
    if not (a and b):
        return "—"
    try:
        d = datetime.datetime.fromisoformat(b) - datetime.datetime.fromisoformat(a)
    except ValueError:
        return "—"
    secs = int(d.total_seconds())
    h, m = divmod(secs // 60, 60)
    return f"{h}h {m}m" if h else f"{m}m"


def build_context(state_dict: dict, metrics: list[dict], config: dict) -> dict:
    """Assemble the values the HTML template needs (pure)."""
    data = state_dict.get("data", {})
    stages = state_dict.get("stages", {})
    profile_name = state_dict.get("profile") or config.get("active_profile") or "?"
    profile = (config.get("profiles") or {}).get(profile_name, {})
    final_loss = metrics[-1]["loss"] if metrics else None
    zip_size = data.get("zip_size")
    return {
        "run_name": data.get("output_dir", profile.get("output_dir", "")).rstrip("/").split("/")[-1] or "(run)",
        "profile": profile_name,
        "current": state_dict.get("current", "?"),
        "gpu_id": data.get("gpu_id"),
        "final_loss": final_loss,
        "num_points": len(metrics),
        "train_duration": _duration(stages.get(Stage.TRAINING.value, {})),
        "output_dir": data.get("output_dir", profile.get("output_dir", "—")),
        "dataset": profile.get("dataset_path", "—"),
        "zip_size_mb": (zip_size / 1e6) if isinstance(zip_size, (int, float)) else None,
        "zip_local": data.get("zip_local", "—"),
        "deploy_host": data.get("deploy_host", "—"),
        "deploy_remote": data.get("deploy_remote", "—"),
        "stages": [(s.value, stages.get(s.value, {})) for s in ORDER],
    }


def _svg_line_chart(steps: list, losses: list, width: int = 820, height: int = 300) -> str:
    """Render a minimal inline-SVG loss curve (zero-dependency fallback)."""
    if not steps or not losses:
        return '<p class="muted">No training metrics were captured.</p>'
    pad = 40
    xmin, xmax = min(steps), max(steps) or 1
    ymin, ymax = min(losses), max(losses)
    yspan = (ymax - ymin) or 1.0
    xspan = (xmax - xmin) or 1.0

    def px(x):
        return pad + (x - xmin) / xspan * (width - 2 * pad)

    def py(y):
        return height - pad - (y - ymin) / yspan * (height - 2 * pad)

    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(steps, losses))
    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
        f'aria-label="loss curve">'
        f'<polyline fill="none" stroke="#6ea8fe" stroke-width="2" points="{pts}"/>'
        f'<text x="{pad}" y="{height-10}" class="svgtxt">{xmin}</text>'
        f'<text x="{width-pad}" y="{height-10}" class="svgtxt" text-anchor="end">{xmax}</text>'
        f'<text x="6" y="{pad}" class="svgtxt">{ymax:.3f}</text>'
        f'<text x="6" y="{height-pad}" class="svgtxt">{ymin:.3f}</text>'
        f'</svg>'
    )


# --------------------------------------------------------------------------- #
#  HTML rendering (pure)
# --------------------------------------------------------------------------- #
_CSS = """
:root{--bg:#0f1419;--panel:#171d26;--text:#e6edf3;--muted:#8b98a5;--accent:#6ea8fe;
--border:#2a3441;--ok:#3fb950;--bad:#f85149;--warn:#d29922;}
[data-theme=light]{--bg:#f6f8fa;--panel:#fff;--text:#1f2328;--muted:#656d76;
--accent:#0969da;--border:#d0d7de;--ok:#1a7f37;--bad:#cf222e;--warn:#9a6700;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:24px}
header{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
h1{font-size:22px;margin:0}h2{font-size:16px;margin:28px 0 12px;color:var(--accent)}
.muted{color:var(--muted)}.sub{color:var(--muted);font-size:13px;margin-top:4px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:8px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px}
.card .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:20px;margin-top:6px;word-break:break-word}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--border);vertical-align:top}
th{color:var(--muted);font-weight:600}
.pill{padding:2px 8px;border-radius:20px;font-size:12px;border:1px solid var(--border)}
.s-done{color:var(--ok)}.s-failed{color:var(--bad)}.s-in_progress{color:var(--warn)}
.chart{width:100%;height:auto;background:var(--panel);border:1px solid var(--border);border-radius:10px}
.svgtxt{fill:var(--muted);font-size:11px}
button.toggle{background:var(--panel);color:var(--text);border:1px solid var(--border);
border-radius:8px;padding:8px 12px;cursor:pointer}
code{background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:2px 6px}
""".strip()

_THEME_JS = """
(function(){var k='gr00t-report-theme';var t=localStorage.getItem(k)||'dark';
document.documentElement.setAttribute('data-theme',t);
window.__toggle=function(){t=(t==='dark')?'light':'dark';
document.documentElement.setAttribute('data-theme',t);localStorage.setItem(k,t);};})();
""".strip()

_CHART_INIT = """
(function(){if(!window.Chart)return;var D=__LOSS__;
var ctx=document.getElementById('lossChart').getContext('2d');
new Chart(ctx,{type:'line',data:{labels:D.steps,datasets:[{label:'training loss',
data:D.loss,borderColor:'#6ea8fe',backgroundColor:'rgba(110,168,254,.15)',
borderWidth:2,pointRadius:0,tension:.25,fill:true}]},
options:{responsive:true,plugins:{legend:{display:true}},
scales:{x:{title:{display:true,text:'step'},ticks:{maxTicksLimit:10}},
y:{title:{display:true,text:'loss'}}}}});})();
""".strip()


def _esc(v) -> str:
    return html.escape("—" if v is None else str(v))


def _status_class(status: str) -> str:
    return {"done": "s-done", "failed": "s-failed", "in_progress": "s-in_progress"}.get(status, "")


def build_html(ctx: dict, metrics: list[dict], chart_js_src: str | None,
               sim_cmd: str = "python launch_isaac_policy.py") -> str:
    """Render the full report HTML (pure). ``chart_js_src`` None -> inline-SVG fallback."""
    steps = [m.get("step") for m in metrics]
    losses = [m.get("loss") for m in metrics]
    final_loss = f"{ctx['final_loss']:.4f}" if ctx.get("final_loss") is not None else "—"
    zip_mb = f"{ctx['zip_size_mb']:.1f} MB" if ctx.get("zip_size_mb") is not None else "—"

    cards = "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v in [
            ("Profile", _esc(ctx["profile"])),
            ("GPU (Pegasus 2×H100)", "index " + _esc(ctx["gpu_id"])),
            ("Training time", _esc(ctx["train_duration"])),
            ("Final loss", final_loss),
            ("Checkpoint size", zip_mb),
            ("Pipeline state", _esc(ctx["current"])),
        ])

    stage_rows = "".join(
        f'<tr><td>{_esc(name)}</td>'
        f'<td class="{_status_class(rec.get("status",""))}">{_esc(rec.get("status","pending"))}</td>'
        f'<td>{_esc(rec.get("attempts",0))}</td>'
        f'<td>{_esc(rec.get("started_at"))}</td>'
        f'<td>{_esc(rec.get("finished_at") or rec.get("failed_at"))}</td>'
        f'<td>{_esc(rec.get("error"))}</td></tr>'
        for name, rec in ctx["stages"])

    env_rows = "".join(
        f'<tr><th>{k}</th><td>{_esc(v)}</td></tr>' for k, v in [
            ("Checkpoint (remote output_dir)", ctx["output_dir"]),
            ("Dataset", ctx["dataset"]),
            ("Local zip", ctx["zip_local"]),
            ("Deployed to", f'{ctx["deploy_host"]}:{ctx["deploy_remote"]}'),
            ("Loss points captured", ctx["num_points"]),
        ])

    if chart_js_src:
        chart_block = '<canvas id="lossChart" height="120"></canvas>'
        chart_script = (f'<script>{chart_js_src}</script>\n'
                        f'<script>{_CHART_INIT.replace("__LOSS__", json.dumps({"steps": steps, "loss": losses}))}</script>')
    else:
        chart_block = _svg_line_chart(steps, losses)
        chart_script = ""

    generated = datetime.datetime.now().isoformat(timespec="seconds")
    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GR00T fine-tune report — {_esc(ctx['run_name'])}</title>
<style>{_CSS}</style><script>{_THEME_JS}</script></head>
<body><div class="wrap">
<header>
  <div><h1>GR00T fine-tune report</h1>
  <div class="sub">{_esc(ctx['run_name'])} · generated {generated}</div></div>
  <button class="toggle" onclick="window.__toggle()">◐ theme</button>
</header>

<h2>Summary</h2>
<div class="cards">{cards}</div>

<h2>Training loss</h2>
<div class="panel">{chart_block}</div>

<h2>Deployment</h2>
<div class="panel"><table>
  <tr><th>Target</th><td>{_esc(ctx['deploy_host'])}</td></tr>
  <tr><th>Remote path</th><td>{_esc(ctx['deploy_remote'])}</td></tr>
</table></div>

<h2>Simulation</h2>
<div class="panel"><p class="muted">Sim validation is run manually on asus-4090:</p>
<p><code>{_esc(sim_cmd)}</code></p>
<p class="sub">The robot repeats the pick-and-place task ~50–100× and accumulates a success rate.</p></div>

<h2>Pipeline log</h2>
<div class="panel"><table>
<tr><th>Stage</th><th>Status</th><th>Attempts</th><th>Started</th><th>Ended</th><th>Error</th></tr>
{stage_rows}</table></div>

<h2>Environment</h2>
<div class="panel"><table>{env_rows}</table></div>

</div>{chart_script}</body></html>"""


# --------------------------------------------------------------------------- #
#  File entry point
# --------------------------------------------------------------------------- #
def generate(config: dict, state, out_dir: Path = HERE) -> Path:
    """REPORTING stage entry point: write report_<timestamp>.html and return its path."""
    metrics = load_metrics()
    state_dict = json.loads(Path(state.path).read_text(encoding="utf-8")) if Path(state.path).exists() else {}
    ctx = build_context(state_dict, metrics, config)
    chart_src = CHARTJS_PATH.read_text(encoding="utf-8") if CHARTJS_PATH.exists() else None
    if not chart_src:
        log.warning("vendor/chart.umd.min.js missing — using inline-SVG fallback chart.")
    html_text = build_html(ctx, metrics, chart_src)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(out_dir) / f"report_{ts}.html"
    out.write_text(html_text, encoding="utf-8")
    state.record_output("report_path", str(out))
    log.info("Report written: %s", out)
    return out
