#!/usr/bin/env python3
"""Launch StarVLA server and IsaacLab client in separate terminals."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import psutil


BASE_DIR = Path(__file__).resolve().parent
STARVLA_CONFIG = BASE_DIR / "IsaacLab/scripts/gr00t_script/policy_configs/starvla_openarm_o6.json"

SERVER = {
    "dir": BASE_DIR / "starVLA",
    "script": "deployment/model_server/server_policy.py",
    "conda_env": "starVLA",
    "title": "StarVLA Server",
}

CLIENT = {
    "dir": BASE_DIR / "IsaacLab",
    "script": "scripts/gr00t_script/gr00t_infer_agent.py",
    "conda_env": "env_isaaclab",
    "title": "IsaacLab StarVLA Client",
}

DEFAULTS = {
    "task": "Isaac-Can-Sorting-OpenArm-DexHand-v0",
    "model_path": BASE_DIR / "starVLA/results/Checkpoints/openarm_o6_qwengroot_right_only_bs16_lr5e5_wd1e5/final_model/pytorch_model.pt",
    "policy_config": STARVLA_CONFIG,
    "max_eps_num": 50,
}


def load_policy_config(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def server_running(port):
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmd = proc.info.get("cmdline") or []
            if any("server_policy.py" in c for c in cmd) and str(port) in cmd:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def launch_in_terminal(cfg, command):
    subprocess.Popen([
        "gnome-terminal",
        f"--title={cfg['title']}",
        f"--working-directory={cfg['dir']}",
        "--",
        "bash", "-c",
        f"source $(conda info --base)/etc/profile.d/conda.sh && "
        f"conda activate {cfg['conda_env']} && {shlex.join(command)}; "
        f'echo "Stopped. Press Enter to close."; read',
    ])


def main():
    p = argparse.ArgumentParser(description="Launch StarVLA server and IsaacLab client.")
    p.add_argument("--model-path", type=Path, default=DEFAULTS["model_path"], help="[Server] StarVLA .pt checkpoint.")
    p.add_argument("--policy-config", type=Path, default=DEFAULTS["policy_config"], help="[Client] Policy adapter JSON.")
    p.add_argument("--max-eps-num", type=int, default=DEFAULTS["max_eps_num"], help="[Client] Max episodes.")
    p.add_argument("--save-video", action="store_true", help="[Client] Save rollout video.")
    args = p.parse_args()

    policy_cfg = load_policy_config(args.policy_config)
    for path in (SERVER["dir"], CLIENT["dir"], args.model_path, args.policy_config, Path(policy_cfg["stats_path"])):
        if not path.exists():
            sys.exit(f"Error: path not found: {path}")

    server_args = ["python", "-u", SERVER["script"], "--ckpt_path", str(args.model_path), "--port", str(policy_cfg["port"]), "--idle_timeout", "-1", "--use_bf16"]
    client_args = ["python3", "-u", CLIENT["script"], "--task", DEFAULTS["task"], "--policy", "starvla", "--policy_config", str(args.policy_config), "--host", policy_cfg["host"], "--port", str(policy_cfg["port"]), "--max_eps_num", str(args.max_eps_num), "--openarm_hand_type", "linkerhand_o6", "--filter"]
    if args.save_video:
        client_args.append("--save_video")

    if server_running(policy_cfg["port"]):
        print("StarVLA server already running — skipping server launch.")
    else:
        launch_in_terminal(SERVER, server_args)

    launch_in_terminal(CLIENT, client_args)


if __name__ == "__main__":
    main()
