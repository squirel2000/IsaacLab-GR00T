"""GPU_WAIT stage: pick a GPU on the Pegasus (2x H100) box for this pipeline's training run.

The selection logic itself is NOT here. It lives in `agents/tools/common/h100.py` so that every
model's harness — GR00T N1.5/N1.7, RLDX-1, and whatever comes next — shares one implementation
of "which card is actually free", instead of each one re-learning it the hard way. See
`agents/tools/common/README.md` for the rules that module encodes.

What stays here is the part that is specific to *this* pipeline: its retry policy, its logger,
its config shape, and recording the chosen GPU into pipeline state.
"""
from __future__ import annotations

import h100
from pipeline_retry import retry
from pipeline_logging import get_logger

import pegasus as pg  # noqa: F401  (re-exported for callers that expect it here)

log = get_logger("gpu_monitor")

# Re-exported so existing imports (and tests) keep working against the shared implementation.
NVIDIA_SMI_CMD = h100.NVIDIA_SMI_CMD
parse_nvidia_smi = h100.parse_nvidia_smi
pick_idle_gpu = h100.pick_idle_gpu
pick_gpu_with_free_memory = h100.pick_gpu_with_free_memory
plan_batch = h100.plan_batch
query_gpus = h100.query_gpus
verify_checkpoint = h100.verify_checkpoint
who_is_training = h100.who_is_training


def connect_with_retry(config: dict, insecure: bool = False):
    """Open a Pegasus session, retrying connect_retry_max times before giving up."""
    t = config["training"]
    return retry(lambda: pg.connect(insecure),
                 attempts=int(t["connect_retry_max"]),
                 interval=int(t["connect_retry_interval_sec"]),
                 label="Pegasus connect")


def wait_for_idle_gpu(s, config: dict, poll_seconds: int | None = None,
                      max_rounds: int | None = None) -> int:
    """Block until an idle GPU is found; return its index. Logs through this pipeline's logger."""
    h100.log = log            # route the shared module's messages into logs/gpu_monitor.log
    return h100.wait_for_idle_gpu(s, config, poll_seconds=poll_seconds, max_rounds=max_rounds)


def run(config: dict, state) -> int:
    """GPU_WAIT stage entry point: connect, pick a GPU, record it, return it.

    ``training.force_gpu`` (0/1) pins a specific GPU and skips idle-waiting; otherwise wait for
    an idle one in ``training.gpu_priority`` order.
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
