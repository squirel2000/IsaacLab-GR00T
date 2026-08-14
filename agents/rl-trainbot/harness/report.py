"""Generate a self-contained HTML analysis report for a Phase-1 RL run.

Flow:
  1. find the run's remote log dir (logs/<ts>-<config>/)
  2. upload + run extract_metrics.py in the RLinf venv -> scalars JSON on Pegasus
  3. fetch the JSON locally
  4. render one standalone HTML file (inline CSS + inline SVG charts; no external
     network deps, opens offline) with config, before/after success, training
     curves, and an honest pass/fail verdict.

Usage:
  python agents/rl-trainbot/harness/report.py --config libero_spatial_ppo_gr00t_n1d7_phase1 [--baseline 0.42]
"""
from __future__ import annotations

import argparse
import html
import json
import pathlib
import time

import yaml

import remote
import pegasus as pg  # via remote's sys.path injection

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = remote.ROOT  # workspace root (resolved via the workspace.yaml marker)
_CFG = _HERE / "config" / "phase1.yaml"


def load_cfg():
    with open(_CFG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def latest_log_dir(s, rlinf_dir, config_name):
    out, _ = remote.sh(s, f"ls -dt {rlinf_dir}/logs/*-{config_name} 2>/dev/null | head -1")
    out = out.strip()
    return out.splitlines()[0].strip() if out else None


def pull_metrics(s, cfg, config_name, logdir=None):
    """Run extract_metrics.py remotely in the venv; return (logdir, metrics_dict)."""
    rlinf = cfg["remote"]["rlinf_dir"]
    logdir = logdir or latest_log_dir(s, rlinf, config_name)
    if not logdir:
        raise SystemExit(f"no log dir found for {config_name}")
    remote_script = f"{rlinf}/scripts_rl_extract_metrics.py"
    remote_json = f"{logdir}/metrics.json"
    pg.put_file(s, str(_HERE / "extract_metrics.py"), remote_script)
    cmd = (
        f"source {cfg['remote']['venv_activate']} && "
        f"python {remote_script} {logdir} {remote_json}"
    )
    out, rc = remote.sh(s, cmd, timeout=600)
    print(out)
    if rc != 0:
        raise SystemExit(f"remote metric extraction failed (rc={rc})")
    local_json = _REPO / "artifacts" / "rl" / config_name / "metrics.json"
    local_json.parent.mkdir(parents=True, exist_ok=True)
    if local_json.exists():
        local_json.unlink()  # avoid resumable-download appending to a stale file
    remote.fetch(s, remote_json, local_json)
    with open(local_json, encoding="utf-8") as f:
        return logdir, json.load(f)


def pick_tag(data, *needles, exact=None):
    if exact and exact in data:
        return exact
    for t in data:
        if all(n.lower() in t.lower() for n in needles):
            return t
    return None


def _svg_line(points, color, w=560, h=160, pad=28):
    """Inline SVG line chart from [[x,y],...]; returns an <svg> string."""
    if not points:
        return '<svg width="%d" height="%d"></svg>' % (w, h)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs) or 1
    ymin, ymax = min(ys), max(ys)
    xr = (xmax - xmin) or 1
    yr = (ymax - ymin) or 1

    def sx(x):
        return pad + (x - xmin) / xr * (w - 2 * pad)

    def sy(y):
        return h - pad - (y - ymin) / yr * (h - 2 * pad)

    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
    grid = (
        f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#ccc"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#ccc"/>'
    )
    labels = (
        f'<text x="{pad}" y="{pad-8}" font-size="10" fill="#666">{ymax:.3g}</text>'
        f'<text x="{pad}" y="{h-pad+14}" font-size="10" fill="#666">{ymin:.3g}</text>'
        f'<text x="{w-pad-30}" y="{h-pad+14}" font-size="10" fill="#666">step {int(xmax)}</text>'
    )
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f"{grid}"
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'
        f"{labels}</svg>"
    )


def _chart_block(title, points, color):
    last = f"{points[-1][1]:.4g}" if points else "—"
    return (
        f'<div class="card"><h3>{html.escape(title)} '
        f'<span class="muted">(latest: {last})</span></h3>{_svg_line(points, color)}</div>'
    )


def build_html(cfg, config_name, logdir, data, baseline):
    succ_tag = pick_tag(data, exact=cfg["metrics"]["success_key"]) or pick_tag(data, "success")
    succ = data.get(succ_tag, []) if succ_tag else []
    final_succ = succ[-1][1] if succ else None

    # Verdict
    margin = 0.02
    if final_succ is None:
        verdict, vclass = "INCONCLUSIVE — no success-rate scalar found yet", "warn"
    elif baseline is None:
        verdict, vclass = (
            f"Post-RL success = {final_succ:.3f} (no baseline provided — supply --baseline to judge lift)",
            "warn",
        )
    elif final_succ > baseline + margin:
        verdict, vclass = (
            f"MET — success rose {baseline:.3f} → {final_succ:.3f} (+{final_succ-baseline:.3f})",
            "ok",
        )
    else:
        verdict, vclass = (
            f"NOT MET — success {baseline:.3f} → {final_succ:.3f} (no clear lift beyond noise)",
            "bad",
        )

    reward_tag = pick_tag(data, "reward")
    loss_tag = pick_tag(data, "loss")
    kl_tag = pick_tag(data, "kl")

    charts = [_chart_block(f"Success rate — {succ_tag or 'n/a'}", succ, "#2563eb")]
    if reward_tag:
        charts.append(_chart_block(f"Reward — {reward_tag}", data[reward_tag], "#16a34a"))
    if loss_tag:
        charts.append(_chart_block(f"Loss — {loss_tag}", data[loss_tag], "#dc2626"))
    if kl_tag:
        charts.append(_chart_block(f"KL — {kl_tag}", data[kl_tag], "#9333ea"))

    meta = {
        "Suite": cfg["suite"],
        "Config": config_name,
        "Remote log dir": logdir,
        "Base checkpoint": f"{cfg['checkpoints']['task_repo']}/{cfg['checkpoints']['task_subdir']}",
        "Backbone": cfg["checkpoints"]["backbone_repo"],
        "GPU policy": f"single GPU, prefer index {cfg['gpu']['prefer_index']}",
        "Baseline success": "—" if baseline is None else f"{baseline:.3f}",
        "Post-RL success": "—" if final_succ is None else f"{final_succ:.3f}",
        "Scalar tags found": ", ".join(data.keys()) or "(none)",
        "Generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in meta.items()
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>RLinf GR00T-N1.7 LIBERO — Phase 1 Report</title>
<style>
  body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#f7f8fa;color:#1f2937}}
  .wrap{{max-width:880px;margin:0 auto;padding:32px 20px}}
  h1{{font-size:22px;margin:0 0 4px}} .sub{{color:#6b7280;margin:0 0 24px}}
  .verdict{{padding:14px 18px;border-radius:10px;font-weight:600;margin:0 0 24px}}
  .ok{{background:#dcfce7;color:#166534}} .bad{{background:#fee2e2;color:#991b1b}} .warn{{background:#fef9c3;color:#854d0e}}
  table{{border-collapse:collapse;width:100%;background:#fff;border-radius:10px;overflow:hidden;margin:0 0 24px;box-shadow:0 1px 2px rgba(0,0,0,.06)}}
  th,td{{text-align:left;padding:9px 14px;border-bottom:1px solid #eef0f3;font-size:13px;vertical-align:top}}
  th{{width:200px;color:#374151;background:#fafbfc}}
  .card{{background:#fff;border-radius:10px;padding:14px 16px;margin:0 0 18px;box-shadow:0 1px 2px rgba(0,0,0,.06);overflow-x:auto}}
  .card h3{{font-size:14px;margin:0 0 8px}} .muted{{color:#9ca3af;font-weight:400}}
  details{{font-size:12px;color:#6b7280}} pre{{white-space:pre-wrap;word-break:break-all}}
</style></head><body><div class="wrap">
  <h1>RLinf GR00T-N1.7 — LIBERO Phase 1</h1>
  <p class="sub">RL validation on one LIBERO suite, single Pegasus H100. Spec: add-rlinf-gr00t-n17-libero-rl</p>
  <div class="verdict {vclass}">{html.escape(verdict)}</div>
  <table>{rows}</table>
  {''.join(charts)}
  <details><summary>Raw metrics JSON</summary><pre>{html.escape(json.dumps(data)[:200000])}</pre></details>
  <p class="sub" style="margin-top:24px">Generated by agents/rl-trainbot/harness/report.py — self-contained, no external dependencies.</p>
</div></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="Build the Phase-1 HTML report")
    ap.add_argument("--config", required=True)
    ap.add_argument("--baseline", type=float, default=None, help="baseline success rate (0-1)")
    ap.add_argument("--logdir", default=None, help="remote log dir override")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_cfg()
    s = remote.connect()
    logdir, data = pull_metrics(s, cfg, args.config, logdir=args.logdir)
    out = pathlib.Path(args.out) if args.out else (_REPO / cfg["report"]["out_html"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(cfg, args.config, logdir, data, args.baseline), encoding="utf-8")
    print(f"\n✓ wrote report -> {out}")


if __name__ == "__main__":
    main()
