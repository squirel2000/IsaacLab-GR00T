#!/usr/bin/env python3
"""Mirror Pegasus training logs locally, parse them, and write a runs index page.

Local and disposable by design: the training itself runs detached on Pegasus under
`setsid nohup`, so killing this poller loses nothing but the live view. Restart it any time.

Metric parsing is delegated to vla-trainbot's own `training_monitor.extract_metrics` via
`training_bridge`, so the two never drift apart.

Writes into --outdir:
    index.html            runs index with live status
    <run-id>.metrics.jsonl
    <run-id>.log          mirrored tail of the training log
    runs.json             machine-readable status
"""
from __future__ import annotations

import argparse
import html
import json
import socket
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

socket.setdefaulttimeout(180)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import training_bridge as bridge  # noqa: E402

sys.path.insert(0, str(bridge.COMMON_TOOLS))
import pegasus  # noqa: E402

TPE = timezone(timedelta(hours=8))

# Filter on Pegasus, not locally. A raw `tail -c 400000` of a tqdm log is mostly carriage
# returns and ANSI, and shipping it over the Jupyter websocket exceeded the call timeout —
# which returns *partial* output rather than failing, so the poller silently saw a 4 KB
# fragment with no progress lines in it. Extracting just the progress and metric fragments
# server-side cuts the payload by ~100x, and training_monitor's regexes still parse them
# because it stamps each metric record with the most recent step it has seen.
# Remote side does nothing but `tail` — no quoting, no regex, nothing to get wrong. Filtering
# happens locally in _keep_lines(). Two earlier attempts to filter server-side both failed for
# shell-quoting reasons (one silently returned the unfiltered tail), and the transfer was never
# the bottleneck: ~400 KB comes back fine at timeout=180.
TAIL_BYTES = 250_000

# Lines worth keeping: tqdm progress, HF metric dicts, and the final summary. Everything else
# is shard-cache chatter that would only bloat the mirrored log.
_KEEP = ("it/s]", "s/it]", "'loss'", "'eval_loss'", "train_runtime")


def _keep_lines(text: str, limit: int = 800) -> str:
    lines = text.replace("\r", "\n").splitlines()
    kept = [ln for ln in lines if any(tok in ln for tok in _KEEP)]
    return "\n".join(kept[-limit:])


def mirror_and_parse(s, run: dict, outdir: Path) -> dict:
    """Pull a run's log tail, parse it, and summarise. Never raises."""
    st = {**{k: run[k] for k in ("id", "label", "model", "max_steps", "gpus",
                                 "effective_batch", "action_space")},
          "state": "unknown", "step": None, "loss": None, "lr": None,
          "grad_norm": None, "records": 0, "pct": 0.0}
    try:
        raw, _ = pegasus.sh(s, f"tail -c {TAIL_BYTES} {run['log']} 2>/dev/null", timeout=180)
        text = _keep_lines(raw)
        (outdir / f"{run['id']}.log").write_text(text, encoding="utf-8")
        if not text.strip():
            st["state"] = "not started"
            return st

        records = bridge.extract(text, run.get("max_steps"))
        (outdir / f"{run['id']}.metrics.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8")
        st["records"] = len(records)
        for r in reversed(records):
            if r.get("step"):
                st.update(step=r.get("step"), loss=r.get("loss"),
                          lr=r.get("lr"), grad_norm=r.get("grad_norm"))
                break

        alive, _ = pegasus.sh(
            s, f"ps -eo cmd | grep -c '{run['log'].split('/')[-2]}' || true", timeout=60)
        done = "train_runtime" in text
        if done:
            st["state"] = "finished"
        elif st["step"]:
            st["state"] = "running"
        else:
            st["state"] = "starting"
        if st["step"] and run.get("max_steps"):
            st["pct"] = round(100.0 * st["step"] / run["max_steps"], 1)
    except Exception as e:  # noqa: BLE001 - a view must not take down the loop
        st["state"] = f"probe error: {type(e).__name__}"
    return st


CSS = """
:root{--bg:#0a0c10;--card:#12171f;--line:#222b38;--ink:#e8edf4;--ink2:#9aa8bb;--ink3:#63718a;
--amber:#ffb020;--sig:#5fd68a;--red:#ff5f5f;--mono:'Cascadia Mono',Consolas,monospace;
--disp:'Saira Condensed','Arial Narrow',sans-serif}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:'IBM Plex Sans',system-ui,sans-serif;padding:24px}
h1{font-family:var(--disp);text-transform:uppercase;font-size:34px;margin:0 0 4px;letter-spacing:.01em}
.sub{font-family:var(--mono);font-size:11.5px;color:var(--ink3);margin-bottom:22px}
.run{background:var(--card);border:1px solid var(--line);border-radius:3px;padding:16px 18px;margin-bottom:12px}
.run h2{font-family:var(--disp);font-size:20px;margin:0 0 8px;font-weight:500;text-transform:uppercase}
.meta{font-family:var(--mono);font-size:11px;color:var(--ink3);margin-bottom:10px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:12px;
font-family:var(--mono);font-variant-numeric:tabular-nums}
.k{font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3)}
.v{font-size:19px;font-weight:600;margin-top:2px}
.bar{height:6px;background:#0a0c10;border:1px solid var(--line);margin-top:12px;overflow:hidden}
.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--amber),var(--sig))}
.pill{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;
padding:3px 8px;border:1px solid var(--line);border-radius:2px;color:var(--ink2)}
.pill.running{color:var(--amber);border-color:var(--amber)}
.pill.finished{color:var(--sig);border-color:var(--sig)}
.pill.error,.pill.unknown{color:var(--red);border-color:var(--red)}
a{color:var(--amber);font-family:var(--mono);font-size:11px}
footer{font-family:var(--mono);font-size:10.5px;color:var(--ink3);margin-top:18px;
border-top:1px solid var(--line);padding-top:12px}
"""


def render_index(runs: list[dict], outdir: Path) -> None:
    now = datetime.now(TPE).strftime("%Y-%m-%d %H:%M:%S")
    parts = [f"<!doctype html><meta charset='utf-8'><title>RLDX-1 Runs</title>"
             f"<meta http-equiv='refresh' content='30'><style>{CSS}</style>",
             "<h1>RLDX-1 &middot; Training Runs</h1>",
             f"<div class='sub'>Pegasus pa-jp-v1 &middot; refreshed {now} TPE "
             f"&middot; metrics parsed by vla-trainbot&rsquo;s own extractor</div>"]
    for r in runs:
        state = r["state"].split(":")[0]
        cls = state if state in ("running", "finished") else "error" if "error" in r["state"] else "unknown"
        step = f"{r['step']:,}" if r["step"] else "—"
        loss = f"{r['loss']:.4f}" if isinstance(r["loss"], (int, float)) else "—"
        gn = f"{r['grad_norm']:.3f}" if isinstance(r["grad_norm"], (int, float)) else "—"
        lr = f"{r['lr']:.2e}" if isinstance(r["lr"], (int, float)) else "—"
        parts.append(
            f"<div class='run'><h2>{html.escape(r['label'])} "
            f"<span class='pill {cls}'>{html.escape(r['state'])}</span></h2>"
            f"<div class='meta'>{html.escape(r['model'])} &middot; {r['gpus']} GPU &middot; "
            f"effective batch {r['effective_batch']} &middot; action {html.escape(r['action_space'])}</div>"
            f"<div class='grid'>"
            f"<div><div class='k'>step</div><div class='v'>{step}<span style='font-size:11px;color:#63718a'>"
            f"/{r['max_steps']:,}</span></div></div>"
            f"<div><div class='k'>loss</div><div class='v'>{loss}</div></div>"
            f"<div><div class='k'>grad norm</div><div class='v'>{gn}</div></div>"
            f"<div><div class='k'>lr</div><div class='v'>{lr}</div></div>"
            f"<div><div class='k'>progress</div><div class='v'>{r['pct']}%</div></div>"
            f"</div><div class='bar'><i style='width:{min(r['pct'],100)}%'></i></div>"
            f"<div style='margin-top:10px'><a href='{r['id']}.metrics.jsonl'>metrics.jsonl</a> "
            f"&nbsp; <a href='{r['id']}.log'>raw log</a></div></div>")
    parts.append("<footer>Training runs detached on Pegasus via setsid nohup — this page is a "
                 "read-only view and stopping the poller does not affect them.</footer>")
    (outdir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path("tmp/dashboard"))
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    s = pegasus.connect()
    while True:
        runs = [mirror_and_parse(s, r, args.outdir) for r in bridge.KNOWN_RUNS]
        (args.outdir / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
        render_index(runs, args.outdir)
        stamp = datetime.now(TPE).strftime("%H:%M:%S")
        print(f"[{stamp} TPE] " + " | ".join(
            f"{r['id']}={r['state']}@{r['step'] or '-'}" for r in runs), flush=True)
        if args.once or all(r["state"] == "finished" for r in runs):
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
