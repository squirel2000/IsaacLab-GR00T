"""Training launch + monitoring over Pegasus (reuses run_finetune.py / pegasus.py).

We do NOT reimplement launching or downloading — ``run_finetune`` already launches the
training detached (``setsid``+``nohup``, survives disconnect) and ``download``s a
sha256-verified zip. This module:

  * builds the run_finetune ``cfg`` from the active profile, injecting the detected GPU id,
  * preflights the remote env (``import gr00t``) so a broken conda/uv env fails fast,
  * starts the run (or re-attaches to an existing one after a restart),
  * tails the server log, parsing step/loss/lr into ``logs/metrics.jsonl`` for the report
    and showing live progress, until the wrapper reports DONE / FAILED.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

import pegasus as pg
import run_finetune as rf
import pipeline_config as pc
from gpu_monitor import connect_with_retry
from pipeline_logging import get_logger, LOGS_DIR

log = get_logger("training")

METRICS_PATH = LOGS_DIR / "metrics.jsonl"

_RE_PROGRESS = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")          # tqdm "48721/100000 [..]"
_RE_LOSS = re.compile(r"'loss':\s*([0-9.eE+-]+)")
_RE_LR = re.compile(r"'learning_rate':\s*([0-9.eE+-]+)")


# --------------------------------------------------------------------------- #
#  Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def extract_metrics(text: str, last_step: int = 0, last_total: int = 0):
    """Scan log text; return (records, last_step, last_total).

    Each loss line yields a record stamped with the most recent tqdm step seen, so the
    report can plot loss vs. step. HF logs ``{'loss': L, 'learning_rate': LR, ...}``.
    """
    records = []
    for line in text.splitlines():
        m = _RE_PROGRESS.search(line)
        if m:
            last_step, last_total = int(m.group(1)), int(m.group(2))
        lm = _RE_LOSS.search(line)
        if lm:
            lr_m = _RE_LR.search(line)
            records.append({
                "step": last_step,
                "total": last_total,
                "loss": float(lm.group(1)),
                "lr": float(lr_m.group(1)) if lr_m else None,
            })
    return records, last_step, last_total


def build_finetune_cfg(config: dict, profile: dict, gpu_id: int) -> dict:
    """Assemble the dict ``run_finetune``'s stage functions expect, for this profile/GPU.

    Mirrors ``run_finetune.default_config`` but is config-driven and GPU-parameterized.
    ``env_activate`` maps onto the wrapper's ``conda_activate`` slot (``true`` for uv).
    """
    output_dir = str(profile["output_dir"])
    max_steps = profile["max_steps"]
    keep = f"checkpoint-{max_steps}"
    cleanup = (f"find . -maxdepth 1 -type d -name 'checkpoint-*' "
               f"! -name {shlex.quote(keep)} -exec rm -rf {{}} +")
    zip_name = os.path.basename(output_dir) + ".zip"
    zip_cmd = (f"rm -f {shlex.quote(zip_name)} && zip -r -y {shlex.quote(zip_name)} "
               f"{shlex.quote(os.path.basename(output_dir) + '/')}")
    return dict(
        conda_activate=str(profile["env_activate"]),     # wrapper slot name; 'true' for uv
        train_cwd=str(profile["train_cwd"]),
        train_cmd=pc.build_train_cmd(profile, gpu_id),
        output_dir=output_dir,
        keep_checkpoint=keep,
        cleanup_cmd=cleanup,
        zip_parent=os.path.dirname(output_dir),
        zip_name=zip_name,
        zip_cmd=zip_cmd,
        local_dir=str(config["local"]["download_dir"]),
        state_root=str(config["pegasus"]["state_root"]),
        poll=int(config["training"]["monitor_interval_sec"]),
        run_name=os.path.basename(output_dir),
    )


# --------------------------------------------------------------------------- #
#  Live operations
# --------------------------------------------------------------------------- #
def preflight(s, profile: dict) -> None:
    """Fail fast if the remote env can't ``import gr00t`` before a multi-hour run."""
    check = profile.get("import_check")
    if not check:
        return
    cmd = f"cd {shlex.quote(str(profile['train_cwd']))} && {profile['env_activate']} && {check}"
    log.info("Preflight env check: %s", check)
    out, rc = pg.sh(s, cmd, timeout=600)
    if rc != 0:
        raise SystemExit(f"preflight failed (rc={rc}); env not ready:\n{out.strip()}")
    log.info("Preflight OK.")


def start_or_reattach(s, cfg: dict, state) -> dict:
    """Launch training, or re-attach to the run already recorded in pipeline state."""
    if state.get("runid"):
        info = rf.load_state(cfg)               # reads run_finetune's own <run>.run.json
        log.info("Re-attaching to existing run %s (started %s)",
                 info.get("runid"), info.get("started"))
        return info
    info = rf.start(s, cfg)
    for k in ("runid", "state_abs", "state_rel", "zip_name"):
        state.record_output(k, info.get(k))
    log.info("Launched training run %s (detached; survives disconnect).", info["runid"])
    return info


def monitor(s, cfg: dict, info: dict, state) -> bool:
    """Tail the server log until DONE/FAILED; stream metrics to logs/metrics.jsonl.

    Rebuilds metrics.jsonl from scratch each call (we re-read the whole log on re-attach),
    so restarts never duplicate points. A fetch outage just keeps the loop polling — the
    training keeps running server-side regardless.
    """
    state_rel = info["state_rel"]
    poll = int(cfg["poll"])
    log.info("Monitoring %s (poll %ds; Ctrl-C is safe — resume re-attaches).",
             info["runid"], poll)

    log_off = 0
    last_step = last_total = 0
    latest = {"loss": None, "lr": None}
    LOGS_DIR.mkdir(exist_ok=True)

    # rich progress is cosmetic; never let a display error stop monitoring.
    try:
        from rich.progress import (Progress, BarColumn, TextColumn,
                                    TimeRemainingColumn, TaskProgressColumn)
        progress = Progress(TextColumn("[bold]train"), BarColumn(),
                            TaskProgressColumn(), TextColumn("{task.description}"),
                            TimeRemainingColumn(), transient=False)
    except Exception:                            # noqa: BLE001
        progress = None

    metrics_f = open(METRICS_PATH, "w", encoding="utf-8")
    task = None
    try:
        if progress:
            progress.start()
            task = progress.add_task("", total=None)
        while True:
            try:
                chunk = rf.fetch(s, f"{state_rel}/run.log", log_off)
                if chunk:
                    text = chunk.decode("utf-8", "replace")
                    log_off += len(chunk)
                    recs, last_step, last_total = extract_metrics(text, last_step, last_total)
                    for r in recs:
                        metrics_f.write(json.dumps(r) + "\n")
                        latest = {"loss": r["loss"], "lr": r["lr"]}
                    metrics_f.flush()
                status = rf.read_text(s, f"{state_rel}/status")
            except Exception as e:               # noqa: BLE001 - outage: wait & keep polling
                log.warning("server unreachable (%s); retrying in 120s "
                            "(training continues on the server)", type(e).__name__)
                time.sleep(120)
                continue

            desc = (f"step {last_step}/{last_total}  loss={latest['loss']}"
                    if last_total else "(waiting for first log lines)")
            if progress and task is not None:
                progress.update(task, total=last_total or None,
                                completed=last_step, description=desc)
            else:
                log.info("[%s] %s", status or "?", desc)

            if status == "DONE":
                log.info("Training DONE (%d points captured).", _count_lines(METRICS_PATH))
                return True
            if status.startswith("FAILED"):
                log.error("Training %s — see server log via run_finetune monitor.", status)
                return False
            time.sleep(poll)
    finally:
        metrics_f.close()
        if progress:
            progress.stop()


def _count_lines(path: Path) -> int:
    try:
        return sum(1 for _ in open(path, encoding="utf-8"))
    except OSError:
        return 0


def run(config: dict, state) -> None:
    """TRAINING stage entry point. Raises on failure so the runner records it & stops."""
    gpu_id = state.get("gpu_id")
    if gpu_id is None:
        raise SystemExit("no gpu_id in state; GPU_WAIT must run first")
    _, profile = pc.resolve_profile(config, state.profile)
    cfg = build_finetune_cfg(config, profile, gpu_id)
    s = connect_with_retry(config)
    if not state.get("runid"):                   # only preflight before a fresh launch
        preflight(s, profile)
    info = start_or_reattach(s, cfg, state)
    if not monitor(s, cfg, info, state):
        raise RuntimeError("training failed (see logs/training.log)")
    state.record_output("output_dir", cfg["output_dir"])
    state.record_output("zip_name", cfg["zip_name"])
