"""Launch RLinf GR00T-N1.7 PPO on LIBERO, on one Pegasus GPU, detached.

Cooperative single-GPU use (spec: rl-run-orchestration):
  - query nvidia-smi, pick an IDLE gpu, preferring GPU1; abort if none free.
  - never occupy more than one GPU.

Single-GPU strategy:
  We pin `CUDA_VISIBLE_DEVICES` to the chosen GPU so the base config's
  `component_placement: {actor,env,rollout: all}` collapses onto that one card,
  and set `MUJOCO_EGL_DEVICE_ID` to the same (global) index so EGL renders on the
  same physical GPU. This avoids both the Hydra `hydra.searchpath` "primary config"
  restriction (we don't compose a derived config) and the comma-key placement
  override problem.

  Instead of a derived Hydra config we call train_embodied_agent.py directly with
  Hydra CLI overrides (checkpoint paths + smoke budget), replicating the LIBERO env
  exports that run_embodiment.sh would have set.

Usage (run locally; talks to Pegasus via agents/tools/common/pegasus.py):
    python agents/rl-trainbot/harness/launch.py --mode smoke         # tiny budget, gate the long run
    python agents/rl-trainbot/harness/launch.py --mode full          # full Phase-1 run
    python agents/rl-trainbot/harness/launch.py --mode smoke --gpu 1  # force a GPU index
"""
from __future__ import annotations

import argparse
import pathlib

import yaml

import remote  # agents/rl-trainbot/harness/remote.py (same dir; invoked as a script)

_HERE = pathlib.Path(__file__).resolve().parent
_CFG = _HERE / "config" / "phase1.yaml"


def load_cfg(path=_CFG):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_gpu(s, gpu_cfg, forced=None):
    """Return the global index of the GPU to use, or None if none are free."""
    if forced is not None:
        return int(forced)
    q = "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits"
    out, rc = remote.sh(s, q)
    if rc != 0:
        raise SystemExit(f"nvidia-smi failed on Pegasus:\n{out}")
    idle = []
    for line in out.strip().splitlines():
        try:
            idx, mem_used, util = (x.strip() for x in line.split(","))
            idx, mem_used, util = int(idx), float(mem_used), float(util)
        except ValueError:
            continue
        if mem_used < gpu_cfg["idle_mem_used_mb"] and util < gpu_cfg["idle_util_pct"]:
            idle.append(idx)
    print(f"  idle GPUs: {idle}")
    if not idle:
        return None
    prefer = gpu_cfg.get("prefer_index", 1)
    return prefer if prefer in idle else idle[0]


def build_cmd(cfg, gpu_index, mode):
    """Return (run_tag, remote_bash_cmd). One self-contained bash script that sets the
    LIBERO env, pins the GPU, and runs train_embodied_agent.py with Hydra overrides."""
    r, ck, rl = cfg["remote"], cfg["checkpoints"], cfg["rlinf"]
    run_tag = rl["smoke_config"] if mode == "smoke" else rl["full_config"]
    rlinf = r["rlinf_dir"]

    if ck.get("single_tier"):
        # N1.5/N1.6-style: one self-contained SFT checkpoint, no separate gated backbone.
        mp = ck["model_path"]
        model_path = mp if mp.startswith("/") else f"{r['ckpt_dir']}/{mp}"
        overrides = [
            f"rollout.model.model_path={model_path}",
            f"actor.model.model_path={model_path}",
        ]
    else:
        model_path = f"{r['ckpt_dir']}/{ck['task_repo'].split('/')[-1]}/{ck['task_subdir']}"
        backbone_path = ck.get("backbone_local_path") or f"{r['ckpt_dir']}/{ck['backbone_repo'].split('/')[-1]}"
        overrides = [
            f"rollout.model.model_path={model_path}",
            f"rollout.model.backbone_model_path={backbone_path}",
            f"actor.model.model_path={model_path}",
            f"actor.model.backbone_model_path={backbone_path}",
        ]
    budget = cfg.get(mode) or {}  # cfg["smoke"] or cfg["full"]
    for k in ("max_epochs", "val_check_interval", "save_interval"):
        if budget.get(k) is not None:
            overrides.append(f"runner.{k}={budget[k]}")
    ov = cfg.get("overrides") or {}
    if ov.get("total_num_envs"):
        overrides += [
            f"env.train.total_num_envs={ov['total_num_envs']}",
            f"env.eval.total_num_envs={ov['total_num_envs']}",
        ]
    if ov.get("micro_batch_size"):
        overrides.append(f"actor.micro_batch_size={ov['micro_batch_size']}")
    if ov.get("global_batch_size"):
        # rollout_size (= total_num_envs x chunk factor) must be divisible by batch_size_per_rank
        overrides.append(f"actor.global_batch_size={ov['global_batch_size']}")
    if ov.get("critic_warmup_steps") is not None:
        overrides.append(f"actor.optim.critic_warmup_steps={ov['critic_warmup_steps']}")
    for extra in ov.get("extra") or []:
        overrides.append(extra)

    ov_str = " ".join(overrides)

    # Container has cgroup pids.max=2048; cap Ray fan-out (taskset -> fewer Ray workers)
    # and force single-threaded math libs so total threads stay under the cap.
    rt = cfg.get("runtime") or {}
    cores = int(rt.get("cpu_cores", 8))
    thread_env = rt.get("thread_env") or {}
    thread_exports = " ".join(f"{k}={v}" for k, v in thread_env.items())
    export_threads = f"export {thread_exports}\n" if thread_exports else ""
    taskset_prefix = f"taskset -c 0-{cores - 1} " if cores > 0 else ""

    cmd = f"""cd {rlinf}
RUN_TAG={run_tag}
LOGDIR="$PWD/logs/$(date +%Y%m%d-%H%M%S)-$RUN_TAG"
mkdir -p "$LOGDIR"
source .venv/bin/activate
# GPU is pinned via the base config's component_placement (= GPU{gpu_index}); RLinf sets
# CUDA_VISIBLE_DEVICES / MUJOCO_EGL_DEVICE_ID per worker, so we do NOT export them here.
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export RLINF_RAY_NUM_CPUS={cores}
export RLINF_LIBERO_INPROCESS=1
{export_threads}export EMBODIED_PATH="$PWD/examples/embodiment"
export REPO_PATH="$PWD"
export PYTHONPATH="$REPO_PATH:$PYTHONPATH"
echo "LOGDIR=$LOGDIR  cores=0-{cores - 1}"
{taskset_prefix}python examples/embodiment/train_embodied_agent.py \
  --config-path "$EMBODIED_PATH/config/" \
  --config-name {rl['base_config']} \
  runner.logger.log_path="$LOGDIR" {ov_str} 2>&1 | tee "$LOGDIR/run_embodiment.log"
"""
    return run_tag, cmd


def launch(mode, forced_gpu=None, cfg_path=_CFG):
    cfg = load_cfg(cfg_path)
    s = remote.connect()

    gpu = pick_gpu(s, cfg["gpu"], forced=forced_gpu)
    if gpu is None:
        raise SystemExit("No idle GPU on Pegasus — aborting so other users are not disrupted.")
    print(f"  using GPU {gpu} (mode={mode})")

    run_tag, cmd = build_cmd(cfg, gpu, mode)
    results = cfg["remote"]["results_dir"]
    log_path = f"{results}/launch-{run_tag}.log"
    pidfile = f"{results}/launch-{run_tag}.pid"

    rc, pid = remote.run_detached(s, cmd, log_path, pidfile, cwd=cfg["remote"]["rlinf_dir"])
    if not pid:
        # rc reflects the trailing pidfile-write/cat bookkeeping, not the backgrounded
        # job itself (already detached by this point) -- only "no PID at all" is fatal.
        raise SystemExit(f"detached launch failed (rc={rc}, no PID found); see {log_path}")
    print(f"\n[OK] launched detached: PID={pid}  GPU={gpu}  tag={run_tag}")
    print(f"  wrapper log: {log_path}")
    print(f"  monitor locally:  python agents/rl-trainbot/harness/poll.py --pid {pid} --config {run_tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Launch RLinf GR00T-N1.7 LIBERO PPO on one Pegasus GPU")
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--gpu", type=int, default=None, help="force a GPU index (else auto-pick, prefer GPU1)")
    ap.add_argument("--config-file", default=str(_CFG), help="path to the orchestration yaml (default: config/phase1.yaml)")
    args = ap.parse_args()
    launch(args.mode, forced_gpu=args.gpu, cfg_path=args.config_file)
