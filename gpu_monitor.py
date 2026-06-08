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


def pick_idle_gpu(gpus: list[dict], util_thresh: float, mem_gb_thresh: float):
    """Return the lowest index whose util AND memory are below thresholds, else None."""
    for g in sorted(gpus, key=lambda x: x["index"]):
        if g["util"] < util_thresh and g["mem_gb"] < mem_gb_thresh:
            return g["index"]
    return None


# --------------------------------------------------------------------------- #
#  Live operations (over Pegasus)
# --------------------------------------------------------------------------- #
def connect_with_retry(config: dict, insecure: bool = False):
    """Open a Pegasus session, retrying connect_retry_max times before giving up."""
    t = config["training"]
    last = None
    for attempt in range(1, int(t["connect_retry_max"]) + 1):
        try:
            return pg.connect(insecure)
        except Exception as e:                  # noqa: BLE001 - log & retry any connect failure
            last = e
            log.warning("Pegasus connect failed (attempt %d/%d): %s",
                        attempt, t["connect_retry_max"], e)
            if attempt < int(t["connect_retry_max"]):
                time.sleep(int(t["connect_retry_interval_sec"]))
    raise SystemExit(f"could not connect to Pegasus after retries: {last}")


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
    poll = poll_seconds if poll_seconds is not None else int(t["gpu_poll_interval_min"]) * 60

    rounds = 0
    while True:
        gpus = query_gpus(s)
        idx = pick_idle_gpu(gpus, util_thresh, mem_thresh)
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
    """GPU_WAIT stage entry point: connect, wait for an idle GPU, record it, return it."""
    s = connect_with_retry(config)
    gpu_id = wait_for_idle_gpu(s, config)
    state.record_output("gpu_id", gpu_id)
    log.info("Selected GPU %d for training.", gpu_id)
    return gpu_id
