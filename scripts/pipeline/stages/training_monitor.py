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

import datetime
import json
import os
import re
import shlex
import threading
import time
from pathlib import Path

import pegasus as pg
import run_finetune as rf
import pipeline_config as pc
import pipeline_progress as pp
from gpu_monitor import connect_with_retry
from pipeline_logging import get_logger
from pipeline_paths import LOGS_DIR, METRICS_PATH

log = get_logger("training")

_RE_PROGRESS = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")          # tqdm "48721/100000 [..]"
_RE_LOSS = re.compile(r"'loss':\s*([0-9.eE+-]+)")
_RE_LR = re.compile(r"'learning_rate':\s*([0-9.eE+-]+)")
_RE_GRAD = re.compile(r"'grad_norm':\s*([0-9.eE+-]+)")
_RE_EVAL = re.compile(r"'eval_loss':\s*([0-9.eE+-]+)")   # 'eval_loss' won't match _RE_LOSS


def _f(m):
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
#  Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def extract_metrics(text: str, last_step: int = 0, last_total: int = 0, train_total=None):
    """Scan log text; return (records, last_step, last_total).

    Each record is stamped with the most recent tqdm step. HF training lines look like
    ``{'loss': L, 'grad_norm': G, 'learning_rate': LR}`` -> a record with loss/lr/grad_norm;
    eval lines ``{'eval_loss': E, ...}`` -> a separate record with eval_loss.

    ``train_total`` (= max_steps): when given, only progress bars with that total update the
    step, so eval/checkpoint-loading bars (small totals) don't hijack the training step.
    """
    records = []
    for line in text.splitlines():
        m = _RE_PROGRESS.search(line)
        if m:
            st, tot = int(m.group(1)), int(m.group(2))
            if train_total is None or tot == train_total:
                last_step, last_total = st, tot
        lm = _RE_LOSS.search(line)
        if lm:
            records.append({
                "step": last_step, "total": last_total,
                "loss": float(lm.group(1)),
                "lr": _f(_RE_LR.search(line)),
                "grad_norm": _f(_RE_GRAD.search(line)),
            })
        em = _RE_EVAL.search(line)
        if em:
            records.append({"step": last_step, "total": last_total,
                            "eval_loss": float(em.group(1))})
    return records, last_step, last_total


def compute_eta(base_t, base_step, cur_step, total, now):
    """(elapsed_sec, eta_sec) from a SESSION baseline (rate over this monitor's observations).

    Correct for resumed runs (cur_step starts high, e.g. 135000) and during eval (no new
    steps): rate = (cur_step - base_step) / (now - base_t), NOT absolute step / launch time.
    Returns (None, None) before a baseline exists; eta is None until a second sample.
    """
    if base_t is None:
        return None, None
    elapsed = int(now - base_t)
    if cur_step <= base_step or now <= base_t or not total:
        return (elapsed if elapsed > 0 else None), None
    rate = (cur_step - base_step) / (now - base_t)
    eta = int((total - cur_step) / rate) if rate > 0 else None
    return elapsed, eta


def build_finetune_cfg(config: dict, profile: dict, gpu_id: int) -> dict:
    """Assemble the dict ``run_finetune``'s stage functions expect, for this profile/GPU.

    Mirrors ``run_finetune.default_config`` but is config-driven and GPU-parameterized.
    ``env_activate`` maps onto the wrapper's ``conda_activate`` slot (``true`` for uv).
    """
    output_dir = str(profile["output_dir"])
    base = os.path.basename(output_dir)
    max_steps = profile["max_steps"]
    keep = f"checkpoint-{max_steps}"
    cleanup = (f"find . -maxdepth 1 -type d -name 'checkpoint-*' "
               f"! -name {shlex.quote(keep)} -exec rm -rf {{}} +")
    zip_name = base + ".zip"
    # Ship the inference-ready top-level model (config + model-*.safetensors + experiment_cfg +
    # processor/) PLUS checkpoint-*/trainer_state.json — the full HF log_history (step/loss/lr/
    # grad_norm/eval) used for the training curve. EXCLUDE only the heavy resume-only tensors in
    # the checkpoint (optimizer ~13 GB, the duplicate model shards, rng, scheduler) so the transfer
    # stays ~the model size instead of doubling.
    ckpt_heavy = " ".join(
        shlex.quote(f"{base}/checkpoint-*/{p}")
        for p in ("optimizer.pt", "model-*.safetensors", "rng_state.pth", "scheduler.pt"))
    zip_cmd = (f"rm -f {shlex.quote(zip_name)} && zip -r -y {shlex.quote(zip_name)} "
               f"{shlex.quote(base + '/')} -x {ckpt_heavy}")
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
        max_steps=int(max_steps),                        # = training tqdm total (filters eval bars)
        # Injected into the detached job's ENV at launch (not the logged command) so wandb
        # logs to the per-user account from the gitignored config; "" => not set.
        wandb_api_key=str((config.get("wandb") or {}).get("api_key", "") or ""),
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


def info_from_state(cfg: dict, state) -> dict:
    """Reconstruct the run 'info' dict from pipeline state (no dependency on rf's json)."""
    runid = state.get("runid")
    state_abs = state.get("state_abs") or f"{cfg['state_root']}/{runid}"
    return {
        "runid": runid,
        "state_abs": state_abs,
        "state_rel": state.get("state_rel") or pg.relpath(state_abs),
        "zip_name": state.get("zip_name") or cfg["zip_name"],
    }


def _safe_sh(s, cmd: str) -> None:
    """Run a remote command, swallowing any error (used fire-and-forget for the launch)."""
    try:
        pg.sh(s, cmd, timeout=90)
    except Exception:                            # noqa: BLE001 - ack may stall; job still runs
        pass


def _wait_started(s, state_rel: str, timeout: int = 120) -> bool:
    """Confirm the detached job started by polling the server STATUS file over HTTP.

    Uses the reliable /files endpoint (not the kernel websocket), so it works even when
    the launch ack stalled under load.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if rf.read_text(s, f"{state_rel}/status"):     # wrapper writes RUNNING at start
            return True
        time.sleep(5)
    return bool(rf.read_text(s, f"{state_rel}/status"))


def _launch(s, cfg: dict, state) -> dict:
    """Robustly launch the detached run.

    The training survives a local disconnect (setsid+nohup), but the launch's kernel-ws
    ack can stall under box load. So we (1) persist the runid to pipeline state BEFORE the
    launch call (a stall/crash stays resumable, never double-launches), (2) bound the ack
    timeout and treat a non-return as non-fatal, (3) verify the job actually started via
    HTTP.
    """
    runid = cfg["run_name"] + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    state_abs = f"{cfg['state_root']}/{runid}"
    state_rel = pg.relpath(state_abs)
    info = {"runid": runid, "state_abs": state_abs, "state_rel": state_rel,
            "zip_name": cfg["zip_name"], "started": datetime.datetime.now().isoformat()}
    state.record_outputs(runid=runid, state_abs=state_abs,        # persist BEFORE launching
                         state_rel=state_rel, zip_name=cfg["zip_name"])

    log.info("Launching run %s (detached) ...", runid)
    pg.put_file(s, rf._tmp_write(rf.wrapper_script(cfg, state_abs), "wrapper.sh"),
                f"{state_rel}/wrapper.sh")
    # Inject the wandb key into the detached job's ENVIRONMENT (inherited by training) rather
    # than the command line, so the wrapper's `set -x` never echoes it into run.log.
    wandb_env = (f"WANDB_API_KEY={shlex.quote(cfg['wandb_api_key'])} "
                 if cfg.get("wandb_api_key") else "")
    launch = (f"cd {shlex.quote(state_abs)} && : > run.log && "
              f"{wandb_env}nohup setsid bash wrapper.sh >/dev/null 2>&1 & "
              f"echo $! > wrapper.pid; echo started pid=$(cat wrapper.pid)")
    # NEVER block on the launch's kernel-websocket ack: under box load that recv can stall
    # for minutes even though the detached wrapper (setsid+nohup) has already started. Fire
    # it in a daemon thread and confirm the job started over HTTP — the reliable path.
    threading.Thread(target=_safe_sh, args=(s, launch), daemon=True).start()
    if not _wait_started(s, state_rel, timeout=180):
        raise RuntimeError("training did not start on the server (no status file appeared)")
    log.info("Run %s confirmed started (detached; survives disconnect).", runid)
    return info


def start_or_reattach(s, cfg: dict, state) -> dict:
    """Re-attach to a still-RUNNING recorded run; otherwise (re)launch.

    If the recorded run is dead/FAILED (e.g. the box rebooted or the job was killed), we
    relaunch a fresh run pointing at the SAME output_dir — the N1.7 trainer auto-resumes
    from the latest checkpoint-N, so `--resume` continues training instead of giving up.
    """
    runid = state.get("runid")
    if runid:
        info = info_from_state(cfg, state)
        status = rf.read_text(s, f"{info['state_rel']}/status")
        if status == "RUNNING":
            log.info("Re-attaching to running run %s", runid)
            return info
        log.info("Recorded run %s is %s -> relaunching; trainer auto-resumes from the latest "
                 "checkpoint in %s", runid, status or "(gone)", cfg["output_dir"])
    return _launch(s, cfg, state)


def _prior_segment_logs(s, cfg: dict, info: dict) -> list[str]:
    """run.log paths of EARLIER relaunches for this output_dir (excludes the live run), ordered.

    Each ``--resume`` relaunch made a fresh ``<run_name>_<ts>`` state dir whose run.log holds
    only the post-resume steps; reading these first (chronologically) rebuilds the full
    pre-resume curve so the dashboard/report show step 0 → now, not just since the last resume.
    """
    run_name, root = cfg["run_name"], cfg["state_root"]
    out, _ = pg.sh(s, f"ls -d {shlex.quote(root)}/{shlex.quote(run_name)}_* 2>/dev/null | sort")
    cur = str(info["state_abs"]).rstrip("/")
    dirs = [d.strip().rstrip("/") for d in out.splitlines() if d.strip()]
    return [pg.relpath(d) + "/run.log" for d in dirs if d != cur]


def monitor(s, cfg: dict, info: dict, state) -> bool:
    """Tail the server log until DONE/FAILED; stream metrics to logs/metrics.jsonl.

    Rebuilds metrics.jsonl from scratch each call so restarts never duplicate points. To show
    the FULL training curve across ``--resume`` relaunches (each wrote its own run.log), it
    first reads every prior segment's log (tagged ``seg=0,1,...``) then tails the live one;
    the dashboard colors later segments so resume points are visible. A fetch outage just
    keeps the loop polling — training continues server-side regardless.
    """
    state_rel = info["state_rel"]
    poll = int(cfg["poll"])
    log.info("Monitoring %s (poll %ds; Ctrl-C is safe — resume re-attaches).",
             info["runid"], poll)

    last_step = last_total = 0
    train_total = int(cfg.get("max_steps") or 0) or None     # filters eval/loading bars
    base_t = base_step = None                                # session baseline for rate/ETA
    latest = {"loss": None, "lr": None, "grad_norm": None, "eval_loss": None}
    LOGS_DIR.mkdir(exist_ok=True)
    metrics_f = open(METRICS_PATH, "w", encoding="utf-8")

    def _write(recs, seg):
        for r in recs:
            r["seg"] = seg
            metrics_f.write(json.dumps(r) + "\n")
            for k in ("loss", "lr", "grad_norm", "eval_loss"):
                if r.get(k) is not None:
                    latest[k] = r[k]

    # Backfill the full curve from earlier relaunch segments (each read once, fully).
    prior = _prior_segment_logs(s, cfg, info)
    for i, rel in enumerate(prior):
        full = rf.fetch(s, rel, 0).decode("utf-8", "replace")
        recs, last_step, last_total = extract_metrics(full, last_step, last_total, train_total)
        _write(recs, i)
    metrics_f.flush()
    cur_seg = len(prior)                                     # the live run is the last segment
    log_off = 0
    try:
        while True:
            try:
                chunk = rf.fetch(s, f"{state_rel}/run.log", log_off)
                if chunk:
                    text = chunk.decode("utf-8", "replace")
                    log_off += len(chunk)
                    recs, last_step, last_total = extract_metrics(
                        text, last_step, last_total, train_total=train_total)
                    _write(recs, cur_seg)
                    metrics_f.flush()
                status = rf.read_text(s, f"{state_rel}/status")
            except Exception as e:               # noqa: BLE001 - outage: wait & keep polling
                log.warning("server unreachable (%s); retrying in 120s "
                            "(training continues on the server)", type(e).__name__)
                time.sleep(120)
                continue

            now = time.time()
            if base_t is None and last_total and last_step > 0:   # first real training step
                base_t, base_step = now, last_step
            desc = (f"step {last_step}/{last_total}  loss={latest['loss']}"
                    if last_total else "(waiting for first log lines)")
            elapsed, eta = compute_eta(base_t, base_step, last_step, last_total, now)
            pp.update("train", step=last_step, total=last_total,
                      pct=(round(100 * last_step / last_total, 1) if last_total else None),
                      loss=latest["loss"], lr=latest["lr"], grad_norm=latest["grad_norm"],
                      eval_loss=latest["eval_loss"], status=status,
                      elapsed_sec=elapsed, eta_sec=eta)
            log.info("[%s] %s", status or "?", desc)    # dashboard shows the live bar

            if status == "DONE":
                log.info("Training DONE (%d points captured).", _count_lines(METRICS_PATH))
                return True
            if status.startswith("FAILED"):
                log.error("Training %s — see server log via run_finetune monitor.", status)
                return False
            time.sleep(poll)
    finally:
        metrics_f.close()


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
    state.record_outputs(output_dir=cfg["output_dir"], zip_name=cfg["zip_name"])


def stop_run(config: dict, state) -> str:
    """Terminate the detached training run recorded in pipeline state.

    ``wrapper.pid`` is unreliable (setsid detaches the job into a new session, so the
    recorded PID is wrong/dead), so the reliable kill targets every process whose argv
    contains our UNIQUE output_dir name — only this run's processes match, never another
    user's job on the shared box.
    """
    runid = state.get("runid")
    if not runid:
        return "no active run recorded in state; nothing to stop"
    _, profile = pc.resolve_profile(config, state.profile)
    cfg = build_finetune_cfg(config, profile, state.get("gpu_id") or 0)
    s = connect_with_retry(config)
    # Kill by the run's unique output_dir in the process argv. (wrapper.pid is unreliable
    # under setsid, so rf.stop's PGID kill was a no-op here — dropped.)
    pattern = os.path.basename(str(cfg["output_dir"]).rstrip("/"))
    pg.sh(s, f"pkill -TERM -f {shlex.quote(pattern)}; sleep 3; "
             f"pkill -KILL -f {shlex.quote(pattern)}; echo stopped", timeout=60)
    left, _ = pg.sh(s, f"pgrep -af {shlex.quote(pattern)} | grep -v pgrep | wc -l")
    log.info("Stop sent for %s (pattern '%s'); processes remaining: %s",
             runid, pattern, left.strip())
    return f"stopped {runid} (pattern '{pattern}'); processes remaining: {left.strip()}"
