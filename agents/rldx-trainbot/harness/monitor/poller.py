#!/usr/bin/env python3
"""Poll training runs on the training host and write the monitor page.

Model-agnostic: it iterates runs.RUNS and delegates metric parsing to vla-trainbot's own
extractor, so RLDX-1, GR00T N1.7 and anything else that trains through HF Trainer are handled
by the same code path.

Hardened against the two failure modes seen in practice:
  * a Pegasus call blocking past its timeout in a way `socket.setdefaulttimeout` does not cover
    -> every probe runs in a worker thread with a hard `future.result(timeout=...)`, so one
       hung call degrades a single run's tile instead of wedging the loop;
  * the session going stale -> reconnect on repeated failure.

Judge liveness by the mtime of index.html, not by this process's stdout: the harness can reap
the captured output while the process keeps working (observed).
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timedelta, timezone
from pathlib import Path

socket.setdefaulttimeout(180)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render  # noqa: E402
import runs as runs_mod  # noqa: E402
import training_bridge as bridge  # noqa: E402

sys.path.insert(0, str(bridge.COMMON_TOOLS))
import pegasus  # noqa: E402

TPE = timezone(timedelta(hours=8))
TAIL_BYTES = 60_000          # 250 KB x 3 runs in one response exceeded the call timeout and
                             # came back truncated; 60 KB per run still carries plenty of history
PROBE_TIMEOUT = 150          # hard ceiling per run, below the loop interval
LOG_TAIL_LINES = 40

# Keep only lines the extractor needs plus the final summary; the rest is shard-cache chatter.
_KEEP = ("it/s]", "s/it]", "'loss'", "'eval_loss'", "train_runtime")


def _keep_lines(text: str, limit: int = 800) -> str:
    lines = text.replace("\r", "\n").splitlines()
    return "\n".join([ln for ln in lines if any(t in ln for t in _KEEP)][-limit:])


# tqdm's own bar: "  7%|▋ | 1954/30000 [41:17<9:35:32,  1.23s/it]" — elapsed<remaining, rate.
# Preferred over a session-baseline rate because it is computed by the trainer itself over the
# WHOLE run (warmup, eval pauses, checkpoint writes and all), not just what we've sampled.
_TQDM_RE = re.compile(r"\[(?P<elapsed>[\d:]+)<(?P<remaining>[\d:]+),\s*"
                      r"(?P<rate>[\d.]+)\s*(?P<unit>it/s|s/it)\]")


def _dur_to_sec(s: str) -> int:
    parts = [int(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return h * 3600 + m * 60 + sec


REMOTE_HELPER = "/data/VLA/tingying/pegasus_runs/_scripts/remote_tail.py"
REMOTE_HELPER_LOCAL = HERE / "remote_tail.py"


def _fetch_all(s, specs: list[runs_mod.RunSpec]) -> tuple[dict[str, dict], dict]:
    """Ask the training host to filter each log, and return (per-run logs, host GPU state).

    The call layer returns an EMPTY string for large responses — shipping raw tqdm tails
    (~180 KB) silently produced nothing, and the monitor read that as "not started" for runs
    that were training or already finished. Doing the filtering on the host keeps each response
    to a few KB, which transfers reliably. The host returns only the lines the extractor needs,
    so parsing still happens locally with vla-trainbot's own extract_metrics.

    GPU state rides along in the same call: it answers "what is on GPU0/1 right now" (including
    other users' jobs) without a separate round trip, which matters on a box we should not be
    logging into back-to-back.
    """
    paths = " ".join(sp.log for sp in specs)
    raw, _ = pegasus.sh(s, f"python3 {REMOTE_HELPER} {paths}", timeout=180)
    data = json.loads(raw.strip().splitlines()[-1])
    logs = {sp.id: data.get(sp.log, {"exists": False, "lines": [], "errors": []})
            for sp in specs}
    return logs, data.get("__host__", {"gpus": []})


def _probe(spec: runs_mod.RunSpec, summary: dict) -> dict:
    st = spec.as_dict()
    st.update(state="pending", step=None, loss=None, lr=None, grad_norm=None,
              pct=0.0, records=0, loss_series=[], grad_norm_series=[], lr_series=[],
              log_tail="", eta=None, elapsed_sec=None, remaining_sec=None,
              total_estimate_sec=None, eta_wall=None, rate_desc=None)

    if not summary.get("exists"):
        st["state"] = "not started"
        return st
    text = "\n".join(summary.get("lines") or [])
    errors = summary.get("errors") or []
    if not text.strip():
        # A log that exists but has no metric lines is either just starting or it died.
        st["state"] = "FAILED" if errors else "starting"
        st["log_tail"] = "\n".join(errors)
        return st

    records = bridge.extract(text, spec.max_steps)
    st["records"] = len(records)
    # No cap here: remote_tail.py already bounds what arrives (KEEP_LINES), and render.py
    # downsamples for display — capping again here would silently drop early-run history,
    # which is exactly what "full training history, like wandb" means not to do.
    # Each point carries its step so the chart can label a real x-axis, not just point index.
    st["loss_series"] = [(r["step"], r["loss"]) for r in records if r.get("loss") is not None]
    st["grad_norm_series"] = [(r["step"], r["grad_norm"]) for r in records
                              if r.get("grad_norm") is not None]
    st["lr_series"] = [(r["step"], r["lr"]) for r in records if r.get("lr") is not None]
    for r in reversed(records):
        if r.get("step"):
            st.update(step=r.get("step"), loss=r.get("loss"),
                      lr=r.get("lr"), grad_norm=r.get("grad_norm"))
            break

    if "train_runtime" in text:
        st["state"] = "finished"
    elif errors:
        st["state"] = "FAILED"          # crashed mid-run; must not read as merely stalled
    else:
        st["state"] = "running" if st["step"] else "starting"
    if st["step"]:
        st["pct"] = round(100.0 * st["step"] / spec.max_steps, 1)

    for line in reversed(text.splitlines()):
        m = _TQDM_RE.search(line)
        if m:
            st["elapsed_sec"] = _dur_to_sec(m["elapsed"])
            st["remaining_sec"] = _dur_to_sec(m["remaining"])
            st["total_estimate_sec"] = st["elapsed_sec"] + st["remaining_sec"]
            st["eta"] = m["remaining"].strip()
            st["rate_desc"] = f"{m['rate']} {m['unit']}"
            break
    if st["state"] == "finished":
        st["eta"] = "complete"
        st["remaining_sec"] = 0
    if st.get("remaining_sec") is not None:
        finish = datetime.now(TPE) + timedelta(seconds=st["remaining_sec"])
        st["eta_wall"] = finish.strftime("%m-%d %H:%M") + " TPE"

    st["log_tail"] = "\n".join(text.splitlines()[-LOG_TAIL_LINES:])
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path("tmp/dashboard"))
    ap.add_argument("--interval", type=int, default=90)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    index = args.outdir / "index.html"

    s = pegasus.connect()
    # Push the current remote_tail.py every start-up so the host copy cannot silently drift from
    # this one (it grew a host-GPU-state field after the copy on Pegasus was first uploaded).
    pegasus.put_file(s, str(REMOTE_HELPER_LOCAL), REMOTE_HELPER)
    fails = 0
    # A finished run's log never changes again, so re-fetching and re-parsing its full history
    # every cycle is pure waste — it was the single biggest driver of poll bandwidth once every
    # run in the registry had finished. Cache its last computed status and stop asking for it.
    finished_cache: dict[str, dict] = {}

    while True:
        statuses = []
        cycle_failed = 0
        pending = [sp for sp in runs_mod.RUNS if sp.id not in finished_cache]

        # One worker, and a fresh pool each cycle. Probes MUST be serialised: they share a
        # single Pegasus kernel session, and running them concurrently interleaved the
        # websocket traffic so responses came back empty — which this code then misread as
        # "not started" for a run that had in fact completed. A fresh pool per cycle also
        # means a hung probe leaks one thread instead of blocking every later cycle behind it.
        # A fresh single-worker pool per cycle gives the fetch a hard timeout; abandoning it on
        # hang leaks one thread instead of blocking every later cycle behind it.
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            logs, host_state = pool.submit(_fetch_all, s, pending).result(timeout=PROBE_TIMEOUT)
        except (FutureTimeout, Exception) as e:  # noqa: BLE001 - the view must survive anything
            reason = "fetch timeout" if isinstance(e, FutureTimeout) else f"fetch error: {type(e).__name__}"
            cycle_failed = len(pending)
            logs, host_state = {}, {"gpus": []}
            for spec in pending:
                st = spec.as_dict()
                st.update(state=reason, step=None, loss=None, lr=None, grad_norm=None,
                          pct=0.0, records=0, loss_series=[], log_tail="", eta=None)
                statuses.append(st)

        for spec in pending:
            if spec.id not in logs:
                continue
            try:
                st = _probe(spec, logs[spec.id])
                statuses.append(st)
                if st["state"] == "finished":
                    finished_cache[spec.id] = st       # never fetched again this process
            except Exception as e:  # noqa: BLE001
                cycle_failed += 1
                st = spec.as_dict()
                st.update(state=f"parse error: {type(e).__name__}", step=None, loss=None,
                          lr=None, grad_norm=None, pct=0.0, records=0,
                          loss_series=[], log_tail="", eta=None)
                statuses.append(st)

        pool.shutdown(wait=False)          # abandon a hung worker rather than block on it
        # Re-assemble in registry order — cached-finished runs interleave with freshly-probed
        # ones exactly as runs.py lists them, so tab order never depends on fetch order.
        by_id = {st["id"]: st for st in statuses}
        statuses = [finished_cache.get(sp.id, by_id.get(sp.id)) for sp in runs_mod.RUNS]
        statuses = [st for st in statuses if st is not None]
        render.write(statuses, index, poll_seconds=args.interval, host_state=host_state)
        stamp = datetime.now(TPE).strftime("%H:%M:%S")
        print(f"[{stamp} TPE] " + " | ".join(
            f"{s_['id'].split('_30k')[0]}={s_['state']}@{s_['step'] or '-'}"
            for s_ in statuses), flush=True)

        # Compare against len(pending), not len(runs_mod.RUNS): once any run is cached-finished,
        # pending shrinks, and comparing against the original full count would mean this never
        # trips again even if every REMAINING probe fails every cycle. An empty `pending` (every
        # run cached) trivially has cycle_failed == 0, which must read as "nothing failed," not
        # "everything failed" — the `and pending` guard keeps that case out of the count.
        fails = fails + 1 if pending and cycle_failed == len(pending) else 0
        if fails >= 2:
            print("  all probes failing — reconnecting", flush=True)
            try:
                s = pegasus.connect()
                fails = 0
            except Exception as e:  # noqa: BLE001
                print(f"  reconnect failed: {e}", flush=True)

        if args.once or all(st["state"] == "finished" for st in statuses):
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
