#!/usr/bin/env python3
"""Shared rules for running training on the Pegasus 2xH100 box.

Every model we train on Pegasus — GR00T N1.5/N1.7, RLDX-1, whatever comes next — hits the same
five facts about that machine. This module is where those facts live, so a new model's harness
inherits them instead of rediscovering them:

  1. The box is SHARED. Colleagues launch jobs from their own checkouts under the same account.
     A run must ask which device is free (`pick_idle_gpu`) instead of assuming device 0.
  2. "Idle" means util AND memory are both low. A card at 0% utilisation can still be holding
     70 GB, which is what actually killed two of our runs.
  3. Absence of a process is NOT success. An OOM crash and a completed run look identical to a
     process scan, so a stage must verify its own artefact (`verify_checkpoint`).
  4. A process scan must match every entrypoint on the box, not just our own
     (`TRAINING_ENTRYPOINTS`) — ours is not the only script training a VLA here.
  5. Single-GPU and multi-GPU need different per-device batches for the same effective batch
     (`plan_batch`), because sharded optimiser state only exists when there are 2+ ranks.

Reached over Jupyter HTTP/WebSocket via `pegasus.py`, not SSH. Depends only on that plus the
stdlib, so any harness can import it without dragging in a pipeline framework.

CLI:  python h100.py            # print the GPU table and what would be picked
"""
from __future__ import annotations

import logging
import re
import shlex
import time

import pegasus as pg

log = logging.getLogger("h100")

# CSV, no header, no units -> lines like:  "0, 5, 1234"  (index, util %, mem MiB)
NVIDIA_SMI_CMD = ("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total "
                  "--format=csv,noheader,nounits")

# Defaults; a harness may override via its own config.
IDLE_UTIL_PCT = 10
IDLE_MEM_GB = 5.0
GPU_PRIORITY = [1, 0]        # prefer GPU1, fall back to GPU0
POLL_SECONDS = 600

# Any of these running means somebody is training — including checkouts that are not ours.
# `launch_finetune.py` was missing from an earlier version of this list, so a colleague's job
# was invisible to our scan and we launched on top of it.
TRAINING_ENTRYPOINTS = (r"launch_train\.py", r"launch_finetune(_asus)?\.py",
                        r"gr00t_finetune\.py", r"torchrun")


# --------------------------------------------------------------------------- #
#  Pure helpers (no I/O — unit-testable)
# --------------------------------------------------------------------------- #
def parse_nvidia_smi(text: str) -> list[dict]:
    """Parse nvidia-smi CSV (noheader,nounits) into [{index, util, mem_gb, total_gb, free_gb}].

    Unreadable fields (e.g. "[N/A]" on MIG devices) are treated as BUSY, so we never pick a
    card we cannot actually measure.
    """
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].lstrip("-").isdigit():
            continue
        try:
            util = int(float(parts[1]))
        except ValueError:
            util = 100                            # unreadable -> assume busy
        try:
            mem_gb = float(parts[2]) / 1024.0     # MiB -> GiB
        except ValueError:
            mem_gb = 1e9                          # unreadable -> assume busy
        try:
            total_gb = float(parts[3]) / 1024.0 if len(parts) > 3 else 0.0
        except ValueError:
            total_gb = 0.0
        gpus.append({"index": int(parts[0]), "util": util, "mem_gb": mem_gb,
                     "total_gb": total_gb, "free_gb": max(0.0, total_gb - mem_gb)})
    return gpus


def pick_idle_gpu(gpus: list[dict], util_thresh: float = IDLE_UTIL_PCT,
                  mem_gb_thresh: float = IDLE_MEM_GB, priority=None):
    """Return the first idle GPU (util AND memory below thresholds), else None.

    ``priority`` is an ordered list of indices to prefer (e.g. [1, 0] = try GPU1 first, fall
    back to GPU0). Indices not listed come after, in ascending order. Passing nothing means
    ascending index — the box-wide preference belongs to config, not to this function.
    """
    priority = list(priority or [])

    def rank(g):
        return priority.index(g["index"]) if g["index"] in priority else len(priority) + g["index"]

    for g in sorted(gpus, key=rank):
        if g["util"] < util_thresh and g["mem_gb"] < mem_gb_thresh:
            return g["index"]
    return None


def pick_gpu_with_free_memory(gpus: list[dict], need_gb: float, priority=None):
    """Return the first GPU with at least ``need_gb`` free, else None.

    Use this instead of `pick_idle_gpu` when the run's memory appetite is known and large:
    a card at 4 GB used passes the idle test but still cannot host a 72 GB single-card LoRA run.
    """
    priority = list(priority or [])

    def rank(g):
        return priority.index(g["index"]) if g["index"] in priority else len(priority) + g["index"]

    for g in sorted(gpus, key=rank):
        if g["free_gb"] >= need_gb:
            return g["index"]
    return None


def wandb_env(api_key: str | None) -> str:
    """Env-var prefix that hands a launch command a W&B key, or "" when none is given.

    Do not use this over `pegasus.sh`/`pegasus.run`: both wrap the ENTIRE command in
    ``bash -lc "<command>"`` on the remote host, and that outer process's own argv — the whole
    command text, including this prefix — is visible to `ps`/`pgrep` for as long as it exists,
    INCLUDING after it exits: an orphaned/zombied wrapper (observed to linger for the full
    duration of a launch that hangs — see `wandb_env_from_file`) keeps its cmdline visible to
    `ps` until reaped, regardless of setsid/nohup on the child. A wandb key launched this way
    was confirmed exposed this way in practice. Prefer `wandb_env_from_file` for any command
    that reaches the remote host through that transport — which today means any of them.

    This function is kept for a transport that genuinely never echoes or lists its own argv
    (e.g. a key typed directly into an interactive, non-logged shell).
    """
    return f"WANDB_API_KEY={shlex.quote(api_key)} " if api_key else ""


def wandb_env_from_file(remote_key_path: str | None) -> str:
    """Env-var prefix that reads a W&B key from a file at exec time, or "" when none is given.

    Use this instead of `wandb_env` for anything launched via `pegasus.sh`/`pegasus.run`. The
    command text this returns contains only the literal substitution expression
    (``$(cat <path>)``) — bash resolves it internally when it sets the child's environment, so
    the SECRET VALUE never appears in the outer wrapper's own argv, which is what `ps`/`pgrep`
    show (and what stayed visible for 31+ minutes when the raw value was used directly, because
    the wrapper process — a zombie, unkillable, still carrying its full argv — outlived the
    command that spawned it).

    This does not protect the key from another human sharing the SAME machine account: they can
    read the file themselves. It closes the "anyone who runs `ps`" gap, not the
    "anyone with this account" one — there is no shell-script fix for the latter.
    """
    return f"WANDB_API_KEY=$(cat {shlex.quote(remote_key_path)} 2>/dev/null) " if remote_key_path else ""


def put_secret_file(s, local_path: str, remote_path: str) -> None:
    """Upload a secret to `remote_path` and lock it to owner-only (chmod 600).

    A thin wrapper over `pegasus.put_file` + a `chmod` call — exists so callers writing a key
    to disk for `wandb_env_from_file` don't each re-derive the "lock it down after upload" step.
    """
    pg.put_file(s, local_path, remote_path)
    pg.sh(s, f"chmod 600 {shlex.quote(remote_path)}")


def plan_batch(effective_batch: int, num_gpus: int, per_device_cap: int) -> tuple[int, int]:
    """Split an effective batch into (global_batch_arg, gradient_accumulation_steps).

    Both trainers compute per-device batch as ``global_batch // num_gpus`` and apply gradient
    accumulation on top, so effective = global_batch * accum. Holding `effective_batch` fixed
    across runs is what makes their losses and step counts comparable; the split is free to
    change to fit the hardware.

    ``per_device_cap`` is the largest per-device batch the card can hold. One GPU needs a
    smaller cap than two, because sharded optimiser state (DeepSpeed ZeRO) only engages with
    2+ ranks — a single card carries the full optimiser state itself.
    """
    if effective_batch <= 0 or num_gpus <= 0 or per_device_cap <= 0:
        raise ValueError("effective_batch, num_gpus and per_device_cap must all be positive")
    cap_global = per_device_cap * num_gpus
    accum = max(1, -(-effective_batch // cap_global))          # ceil
    if effective_batch % accum:
        raise ValueError(f"effective batch {effective_batch} is not divisible by accum {accum}; "
                         f"pick an effective batch that splits evenly")
    global_batch = effective_batch // accum
    if global_batch % num_gpus:
        raise ValueError(f"global batch {global_batch} does not divide across {num_gpus} GPUs")
    return global_batch, accum


# --------------------------------------------------------------------------- #
#  Live operations (over Pegasus)
# --------------------------------------------------------------------------- #
def query_gpus(s) -> list[dict]:
    """Run nvidia-smi on the box and return parsed GPU stats."""
    out, rc = pg.sh(s, NVIDIA_SMI_CMD)
    if rc != 0:
        raise RuntimeError(f"nvidia-smi failed (rc={rc}): {out.strip()}")
    return parse_nvidia_smi(out)


def summarise(gpus: list[dict]) -> str:
    return ", ".join(f"GPU{g['index']}: util={g['util']}% used={g['mem_gb']:.1f}GB "
                     f"free={g['free_gb']:.1f}GB" for g in gpus) or "(no GPUs reported)"


def who_is_training(s) -> list[str]:
    """Return the command lines of every training process on the box, ours or not.

    Note for callers: run this through `pegasus.sh` with a pattern that cannot match the
    command being sent. Passing a whole script body to `bash -lc` makes `pgrep -f` match the
    sending shell itself, which has twice reported phantom jobs and once killed our own shell.
    """
    pattern = "|".join(TRAINING_ENTRYPOINTS)
    out, _ = pg.sh(s, f"pgrep -af '{pattern}' || true")
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def wait_for_idle_gpu(s, config: dict | None = None, poll_seconds: int | None = None,
                      max_rounds: int | None = None, need_gb: float | None = None) -> int:
    """Block until a usable GPU is found; return its index.

    ``config`` accepts vla-trainbot's shape (``{"training": {...}}``) so existing callers keep
    working; omit it to use this module's defaults. ``need_gb`` switches the test from
    "idle" to "has this much free memory", which is the right test for a known-large run.
    ``max_rounds=1`` refuses immediately instead of queueing — prefer that over launching into
    an OOM when you would rather be told than kept waiting.
    """
    t = (config or {}).get("training", {})
    util_thresh = float(t.get("gpu_idle_threshold_util", IDLE_UTIL_PCT))
    mem_thresh = float(t.get("gpu_idle_threshold_mem_gb", IDLE_MEM_GB))
    priority = t.get("gpu_priority", GPU_PRIORITY)
    poll = (poll_seconds if poll_seconds is not None
            else int(t.get("gpu_poll_interval_min", POLL_SECONDS // 60)) * 60)

    rounds = 0
    while True:
        gpus = query_gpus(s)
        idx = (pick_gpu_with_free_memory(gpus, need_gb, priority) if need_gb
               else pick_idle_gpu(gpus, util_thresh, mem_thresh, priority))
        if idx is not None:
            log.info("Selected GPU %d  [%s]", idx, summarise(gpus))
            return idx
        rounds += 1
        log.info("No usable GPU [%s]; re-checking in %d min (round %d)",
                 summarise(gpus), poll // 60, rounds)
        if max_rounds is not None and rounds >= max_rounds:
            raise TimeoutError(f"no usable GPU after {rounds} round(s): {summarise(gpus)}")
        time.sleep(poll)


def verify_checkpoint(s, output_dir: str, step: int) -> str | None:
    """Return the path of ``checkpoint-<step>`` under ``output_dir``, or None.

    This is how a stage proves it succeeded. Do NOT infer success from the training process
    having exited: an OOM crash leaves no process either, and a chain that treated "nothing is
    running" as "it finished" once reported CHAIN_COMPLETE having trained nothing at all.
    """
    out, _ = pg.sh(s, f"ls -1d {output_dir}/*/checkpoint-{step} {output_dir}/checkpoint-{step} "
                      f"2>/dev/null | head -1")
    path = out.strip().splitlines()[0].strip() if out.strip() else ""
    if not path:
        return None
    ok, _ = pg.sh(s, f"test -f {path}/config.json && echo ok || true")
    return path if ok.strip() == "ok" else None


def last_error(s, log_path: str, lines: int = 3) -> str:
    """Pull the tail of whatever went wrong, for a failure report that names a cause."""
    out, _ = pg.sh(s, f"grep -oE 'OutOfMemoryError[^.]*\\.|Traceback[^:]*|Error[^.]{{0,140}}' "
                      f"{log_path} 2>/dev/null | tail -{lines} || true")
    return out.strip()


def preflight(s, *, need_gb: float | None = None, config: dict | None = None,
              max_rounds: int | None = 1) -> dict:
    """One call to make before launching anything: report the box, choose a GPU.

    Returns ``{"gpu": int, "gpus": [...], "others": [...]}``. Raises TimeoutError when nothing
    usable is free, so the caller refuses to start rather than queueing behind an OOM.
    """
    gpus = query_gpus(s)
    others = who_is_training(s)
    log.info("Box state: %s", summarise(gpus))
    for line in others:
        log.info("Already training: %s", line[:160])
    gpu = wait_for_idle_gpu(s, config, poll_seconds=60, max_rounds=max_rounds, need_gb=need_gb)
    return {"gpu": gpu, "gpus": gpus, "others": others}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _s = pg.connect()
    _gpus = query_gpus(_s)
    print(summarise(_gpus))
    for _line in who_is_training(_s):
        print("training:", re.sub(r"\s+", " ", _line)[:180])
    print("idle pick    :", pick_idle_gpu(_gpus))
    print("needs 60GB   :", pick_gpu_with_free_memory(_gpus, 60.0))
