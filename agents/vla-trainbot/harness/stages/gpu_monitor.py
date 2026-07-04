"""GPU detection & wait module for the Pegasus (2x H100) training box.

Pegasus is reached over the Jupyter HTTP/WebSocket client (``pegasus.py``), NOT SSH, so
we query ``nvidia-smi`` by running it through :func:`pegasus.sh`. A GPU counts as idle
when utilization < threshold AND memory used < threshold. If both H100s are busy we log
the wait and re-check every ``gpu_poll_interval_min`` minutes until one frees up, then
return its index for the training stage to pin via ``CUDA_VISIBLE_DEVICES``.
"""
from __future__ import annotations

import time

import pegasus as pg
from pipeline_retry import retry
from pipeline_logging import get_logger

log = get_logger("gpu_monitor")

# CSV, no header, no units -> lines like:  "0, 5, 1234"  (index, util %, mem MiB)
NVIDIA_SMI_CMD = ("nvidia-smi --query-gpu=index,utilization.gpu,memory.used "
                  "--format=csv,noheader,nounits")


# --------------------------------------------------------------------------- #
#  Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def parse_nvidia_smi(text: str) -> list[dict]:
    """Parse nvidia-smi CSV (noheader,nounits) into [{index, util, mem_gb}].

    Non-numeric fields (e.g. "[N/A]" on MIG devices) are treated as *busy* so we never
    pick a GPU we can't actually measure.
    """
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].lstrip("-").isdigit():
            continue
        index = int(parts[0])
        try:
            util = int(float(parts[1]))
        except ValueError:
            util = 100                          # unreadable -> assume busy
        try:
            mem_gb = float(parts[2]) / 1024.0    # MiB -> GiB
        except ValueError:
            mem_gb = 1e9                          # unreadable -> assume busy
        gpus.append({"index": index, "util": util, "mem_gb": mem_gb})
    return gpus


def pick_idle_gpu(gpus: list[dict], util_thresh: float, mem_gb_thresh: float, priority=None):
    """Return the first idle GPU (util AND memory below thresholds), else None.

    ``priority`` is an ordered list of GPU indices to prefer (e.g. [1, 0] = try GPU1 first,
    fall back to GPU0). Indices not listed come after, in ascending order.
    """
    priority = list(priority or [])

    def rank(g):
        return priority.index(g["index"]) if g["index"] in priority else len(priority) + g["index"]

    for g in sorted(gpus, key=rank):
        if g["util"] < util_thresh and g["mem_gb"] < mem_gb_thresh:
            return g["index"]
    return None


# --------------------------------------------------------------------------- #
#  Live operations (over Pegasus)
# --------------------------------------------------------------------------- #
def connect_with_retry(config: dict, insecure: bool = False):
    """Open a Pegasus session, retrying connect_retry_max times before giving up."""
    t = config["training"]
    return retry(lambda: pg.connect(insecure),
                 attempts=int(t["connect_retry_max"]),
                 interval=int(t["connect_retry_interval_sec"]),
                 label="Pegasus connect")


def query_gpus(s) -> list[dict]:
    """Run nvidia-smi on Pegasus and return parsed GPU stats."""
    out, rc = pg.sh(s, NVIDIA_SMI_CMD)
    if rc != 0:
        raise RuntimeError(f"nvidia-smi failed (rc={rc}): {out.strip()}")
    return parse_nvidia_smi(out)


def wait_for_idle_gpu(s, config: dict, poll_seconds: int | None = None,
                      max_rounds: int | None = None) -> int:
    """Block until an idle GPU is found; return its index.

    Re-checks every ``poll_seconds`` (default gpu_poll_interval_min*60). ``max_rounds``
    bounds the loop for dry-runs/tests (None = wait indefinitely).
    """
    t = config["training"]
    util_thresh = float(t["gpu_idle_threshold_util"])
    mem_thresh = float(t["gpu_idle_threshold_mem_gb"])
    priority = t.get("gpu_priority")             # e.g. [1, 0] -> prefer GPU1, fall back to GPU0
    poll = poll_seconds if poll_seconds is not None else int(t["gpu_poll_interval_min"]) * 60

    rounds = 0
    while True:
        gpus = query_gpus(s)
        idx = pick_idle_gpu(gpus, util_thresh, mem_thresh, priority)
        summary = ", ".join(f"GPU{g['index']}: util={g['util']}% mem={g['mem_gb']:.1f}GB"
                            for g in gpus) or "(no GPUs reported)"
        if idx is not None:
            log.info("Idle GPU found -> index %d  [%s]", idx, summary)
            return idx
        rounds += 1
        log.info("All GPUs busy [%s]; re-checking in %d min (round %d)",
                 summary, poll // 60, rounds)
        if max_rounds is not None and rounds >= max_rounds:
            raise TimeoutError(f"no idle GPU after {rounds} rounds")
        time.sleep(poll)


def run(config: dict, state) -> int:
    """GPU_WAIT stage entry point: connect, pick a GPU, record it, return it.

    ``training.force_gpu`` (0/1) pins a specific GPU and skips idle-waiting; otherwise wait
    for the lowest idle GPU.
    """
    s = connect_with_retry(config)
    t = config["training"]
    forced = t.get("force_gpu")
    if forced is not None:
        gpu_id = int(forced)
        g = next((x for x in query_gpus(s) if x["index"] == gpu_id), None)
        if g and (g["util"] >= float(t["gpu_idle_threshold_util"])
                  or g["mem_gb"] >= float(t["gpu_idle_threshold_mem_gb"])):
            log.warning("Forced GPU %d looks busy (util=%d%% mem=%.1fGB) — using it anyway.",
                        gpu_id, g["util"], g["mem_gb"])
        log.info("Using forced GPU %d (config training.force_gpu).", gpu_id)
    else:
        gpu_id = wait_for_idle_gpu(s, config)
    state.record_output("gpu_id", gpu_id)
    log.info("Selected GPU %d for training.", gpu_id)
    return gpu_id
