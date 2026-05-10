#!/usr/bin/env python3
"""Launch Isaac GR00T server and client in separate terminals."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import psutil

BASE_DIR = Path.home() / "Gits" / "IsaacLab-GR00T"

SERVER = {
    "dir": BASE_DIR / "Isaac-GR00T",
    "script": "scripts/inference_service.py",
    "conda_env": "env_gr00t",
    "title": "Isaac GR00T Server",
}

CLIENT = {
    "dir": BASE_DIR / "IsaacLab",
    "script": "scripts/gr00t_script/gr00t_infer_agent.py",
    "conda_env": "env_isaaclab",
    "title": "Isaac GR00T Client",
}

DEFAULTS = {
    "model_path": "/home/asus/Gits/IsaacLab-GR00T/Isaac-GR00T/outputs/openarm_linkerhando6_cansorting_N15_fft_100k_dataset_0408/checkpoint-100000/",
    "task": "Isaac-Can-Sorting-OpenArm-DexHand-v0",
    "embodiment_tag": "new_embodiment",
    "data_config": "openarm_linkerhand_o6",
    "denoising_steps": 4,
}


def server_running():
    for proc in psutil.process_iter(['cmdline']):
        try:
            cmd = proc.info.get('cmdline') or []
            if any('inference_service.py' in c for c in cmd) and '--server' in cmd:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def launch_in_terminal(cfg, script_args):
    cmd = f"python3 -u {cfg['script']} {shlex.join(script_args)}"
    subprocess.Popen([
        'gnome-terminal',
        f'--title={cfg["title"]}',
        f'--working-directory={cfg["dir"]}',
        '--',
        'bash', '-c',
        f'source $(conda info --base)/etc/profile.d/conda.sh && '
        f'conda activate {cfg["conda_env"]} && {cmd}; '
        f'echo "Stopped. Press Enter to close."; read',
    ])


def main():
    p = argparse.ArgumentParser(description="Launch Isaac GR00T server and client.")
    p.add_argument("--save-img", action="store_true", help="[Client] Save RGB camera images.")
    args = p.parse_args()

    for cfg in (SERVER, CLIENT):
        if not cfg["dir"].exists():
            sys.exit(f"Error: directory not found: {cfg['dir']}")

    if server_running():
        print("Server already running — skipping server launch.")
    else:
        launch_in_terminal(SERVER, [
            "--server",
            "--model_path", DEFAULTS["model_path"],
            "--embodiment_tag", DEFAULTS["embodiment_tag"],
            "--data_config", DEFAULTS["data_config"],
            "--denoising_steps", str(DEFAULTS["denoising_steps"]),
        ])

    client_args = ["--task", DEFAULTS["task"], "--filter"]
    if args.save_img:
        client_args.append("--save-img")
    launch_in_terminal(CLIENT, client_args)


if __name__ == "__main__":
    main()
