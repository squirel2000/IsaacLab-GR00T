"""Run-level manifest for inference sessions.

Captures checkpoint identity, env config, git state, and final stats
so two simulation runs with different checkpoints can be compared
without server-side metadata.
"""

import json
import os
import subprocess
from datetime import datetime


def get_git_info():
    """HEAD SHA + dirty flag of the IsaacLab repo. None on failure."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode().strip()
        )
        return {"sha": sha, "dirty": dirty}
    except Exception:
        return {"sha": None, "dirty": None}


def _jsonable(v):
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return str(v)


def build_initial_manifest(args_cli, robot_type, task_description, step_dt, action_joint_names):
    return {
        # checkpoint_name is the parent folder the user supplies via --save_dir,
        # e.g. "openarm_cansorting_movebasket_n1.5_200k_ds4".
        "checkpoint_name": os.path.basename(os.path.normpath(args_cli.save_dir)),
        "policy": args_cli.policy,
        "gr00t_ver": args_cli.gr00t_ver,
        "host": args_cli.host,
        "port": args_cli.port,
        "task": args_cli.task,
        "task_description": list(task_description),
        "robot_type": robot_type,
        "num_envs": args_cli.num_envs,
        "pov_list": list(args_cli.pov_list),
        "filter": bool(args_cli.filter),
        "max_eps_num": args_cli.max_eps_num,
        "step_dt": float(step_dt),
        "action_joint_names": list(action_joint_names),
        "git": get_git_info(),
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "args_full": {k: _jsonable(v) for k, v in vars(args_cli).items()},
        "result": None,
    }


def finalize_manifest(manifest, episodes_total, success, terminated, truncated, infer_times):
    manifest["end_time"] = datetime.now().isoformat(timespec="seconds")
    result = {
        "episodes_total": int(episodes_total),
        "success": int(success),
        "terminated": int(terminated),
        "truncated": int(truncated),
        "success_rate": (success / episodes_total) if episodes_total else None,
    }
    if infer_times:
        result["inference_time_avg_s"] = float(sum(infer_times) / len(infer_times))
        result["inference_time_max_s"] = float(max(infer_times))
        result["inference_time_min_s"] = float(min(infer_times))
    manifest["result"] = result
    return manifest


def write_manifest(output_dir, manifest):
    path = os.path.join(output_dir, "run_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
