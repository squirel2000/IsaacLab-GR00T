#!/usr/bin/env python3
"""Runs ON the training host: emit a compact, filtered tail of each training log as JSON.

Exists because the Pegasus call layer returns an EMPTY string for large responses — shipping
~180 KB of raw tqdm output silently produced nothing, which the monitor then read as "run not
started" for runs that were in fact training or already finished. Filtering here keeps each
response to a few KB, which transfers reliably.

Only the lines the metric extractor needs are returned, so the local side can still parse them
with vla-trainbot's own `extract_metrics` rather than duplicating that logic here.

Also reports the box's GPU state and who is training, in the same round trip: the monitor needs
those to show real VRAM occupancy, and asking separately would double the number of calls (and
the login pressure) for no benefit.

Usage:  python3 remote_tail.py <log_path> [<log_path> ...]
Output: {"<log_path>": {"exists": bool, "lines": [...], "size": int}, ..., "__host__": {...}}
"""
import json
import re
import subprocess
import sys
from pathlib import Path

TAIL_BYTES = 6_000_000    # read locally on the host — cheap, never crosses the wire. A finished
                          # RLDX-1 30k-step run measured ~3.0 MB; this covers a whole run, not
                          # just its tail — the charts want full history, same as W&B would show.
KEEP_LINES = 6000         # RLDX-1 logs a metric line every logging_steps=10 -> measured 4847
                          # matched lines over a finished 30k-step run. 6000 gives headroom.
KEEP = ("it/s]", "s/it]", "'loss'", "'eval_loss'", "train_runtime")
ERR = ("OutOfMemoryError", "Traceback", "CUDA out of memory")


def summarise(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"exists": False, "lines": [], "size": 0}
    size = p.stat().st_size
    with p.open("rb") as f:
        if size > TAIL_BYTES:
            f.seek(-TAIL_BYTES, 2)
        blob = f.read().decode("utf-8", "replace")

    lines = blob.replace("\r", "\n").splitlines()
    kept = [ln.strip() for ln in lines if any(t in ln for t in KEEP)][-KEEP_LINES:]
    # Surface failures too: a crashed run must not look like a run that never started.
    errs = [ln.strip()[:300] for ln in lines if any(t in ln for t in ERR)][-4:]
    return {"exists": True, "size": size, "lines": kept, "errors": errs}


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except Exception:                                    # a probe must never break the tail
        return ""


def host_state() -> dict:
    """GPU occupancy and, per GPU, which processes hold it — ours and other people's.

    Attribution comes from `--query-compute-apps`, not from the launcher's environment:
    `CUDA_VISIBLE_DEVICES=N` is an env prefix, so it never appears in argv, and guessing from the
    command line would silently mislabel every job. compute-apps also covers processes owned by
    other users, which is the case that matters on a shared box.
    """
    uuid_to_idx, gpus = {}, []
    for line in _run(["nvidia-smi",
                      "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
                      "--format=csv,noheader,nounits"]).splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 5 and f[0].isdigit():
            used, total = float(f[3]) / 1024.0, float(f[4]) / 1024.0
            uuid_to_idx[f[1]] = int(f[0])
            gpus.append({"index": int(f[0]), "util": int(float(f[2])),
                         "mem_gb": round(used, 1), "total_gb": round(total, 1),
                         "free_gb": round(total - used, 1), "apps": []})

    # pid -> (elapsed, cmdline) for labelling; ps sees every user's processes.
    ps = {}
    for line in _run(["ps", "-eo", "pid,etime,args", "--no-headers"]).splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit():
            ps[int(parts[0])] = (parts[1], re.sub(r"\s+", " ", parts[2]))

    by_idx = {g["index"]: g for g in gpus}
    for line in _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                      "--format=csv,noheader,nounits"]).splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 3 or not f[1].isdigit():
            continue
        idx = uuid_to_idx.get(f[0])
        if idx is None or idx not in by_idx:
            continue
        etime, cmd = ps.get(int(f[1]), ("?", "(process not visible)"))
        try:
            mem_gb = round(float(f[2]) / 1024.0, 1)
        except ValueError:
            mem_gb = None
        by_idx[idx]["apps"].append({"pid": int(f[1]), "mem_gb": mem_gb,
                                    "elapsed": etime, "cmd": cmd[:200]})
    return {"gpus": gpus}


if __name__ == "__main__":
    out = {p: summarise(p) for p in sys.argv[1:]}
    out["__host__"] = host_state()
    print(json.dumps(out))
