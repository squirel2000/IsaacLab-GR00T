#!/usr/bin/env python3
"""Launch a policy server and IsaacLab client in separate terminals."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import psutil

from project_paths import project_root, resolve_project_path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = project_root(BASE_DIR)
POLICY_CONFIGS = {
    "starvla": PROJECT_ROOT / "IsaacLab/scripts/gr00t_script/policy_configs/starvla_openarm_o6.json",
    "gr00t": PROJECT_ROOT / "IsaacLab/scripts/gr00t_script/policy_configs/gr00t_n15_openarm_o6.json",
}

CLIENT = {
    "dir": PROJECT_ROOT / "IsaacLab",
    "script": "scripts/gr00t_script/gr00t_infer_agent.py",
    "conda_env": "env_isaaclab",
    "title": "IsaacLab Client",
}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def server(policy, cfg):
    if policy == "starvla":
        return {"dir": resolve_project_path(cfg["starvla_repo"], PROJECT_ROOT), "script": "deployment/model_server/server_policy.py", "conda_env": "starVLA", "title": "StarVLA Server"}
    return {"dir": PROJECT_ROOT / "Isaac-GR00T", "script": "scripts/inference_service.py", "conda_env": "env_gr00t", "title": "Isaac GR00T Server"}


def server_running(policy, port):
    token = "server_policy.py" if policy == "starvla" else "inference_service.py"
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmd = proc.info.get("cmdline") or []
            if not any(token in c for c in cmd):
                continue
            if policy == "gr00t" and "--server" in cmd:
                return True
            if policy == "starvla" and str(port) in cmd:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def launch_in_terminal(cfg, command):
    subprocess.Popen([
        "gnome-terminal", f"--title={cfg['title']}", f"--working-directory={cfg['dir']}", "--", "bash", "-c",
        f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {cfg['conda_env']} && {shlex.join(command)}; "
        f'echo "Stopped. Press Enter to close."; read',
    ])


def server_args(policy, cfg, model_path):
    if policy == "starvla":
        args = ["python", "-u", "deployment/model_server/server_policy.py", "--ckpt_path", str(model_path), "--port", str(cfg["port"]), "--idle_timeout", str(cfg.get("idle_timeout", -1))]
        if "denoising_steps" in cfg:
            args += ["--denoising_steps", str(cfg["denoising_steps"])]
        return args + (["--use_bf16"] if cfg.get("use_bf16", True) else [])
    return ["python3", "-u", "scripts/inference_service.py", "--server", "--model_path", str(model_path), "--embodiment_tag", cfg["embodiment_tag"], "--data_config", cfg["data_config"], "--denoising_steps", str(cfg["denoising_steps"]), "--port", str(cfg["port"])]


def client_args(policy, cfg, config_path, max_eps_num, save_video):
    args = ["python3", "-u", CLIENT["script"], "--task", cfg["task"], "--policy", policy, "--policy_config", str(config_path), "--host", cfg["host"], "--port", str(cfg["port"]), "--max_eps_num", str(max_eps_num), "--openarm_hand_type", cfg.get("openarm_hand_type", "linkerhand_o6"), "--filter"]
    if policy == "gr00t":
        args += ["--gr00t_ver", cfg.get("version", "N1.5")]
    if save_video:
        args.append("--save_video")
    return args


def main():
    p = argparse.ArgumentParser(description="Launch policy server and IsaacLab client.")
    p.add_argument("--policy", choices=sorted(POLICY_CONFIGS), default="gr00t", help="Policy backend to launch. Default: gr00t.")
    p.add_argument("--policy-config", type=Path, help="Policy JSON. Defaults to the selected policy config.")
    p.add_argument("--model-path", type=Path, help="Override checkpoint/model path from the policy JSON.")
    p.add_argument("--max-eps-num", type=int, default=100, help="[Client] Max episodes. Default: 10.")
    p.add_argument("--save-video", action="store_true", help="[Client] Save rollout video. Default: False.")
    args = p.parse_args()

    config_path = args.policy_config or POLICY_CONFIGS[args.policy]
    policy_cfg = load_json(config_path)
    model_path = resolve_project_path(args.model_path or policy_cfg["model_path"], PROJECT_ROOT)
    server_cfg = server(args.policy, policy_cfg)

    required = [server_cfg["dir"], CLIENT["dir"], config_path, model_path]
    if args.policy == "starvla":
        required.append(resolve_project_path(policy_cfg["stats_path"], PROJECT_ROOT))
    for path in required:
        if not Path(path).exists():
            sys.exit(f"Error: path not found: {path}")

    if server_running(args.policy, policy_cfg["port"]):
        print(f"{args.policy} server already running - skipping server launch.")
    else:
        launch_in_terminal(server_cfg, server_args(args.policy, policy_cfg, model_path))

    launch_in_terminal(CLIENT, client_args(args.policy, policy_cfg, config_path, args.max_eps_num, args.save_video))


if __name__ == "__main__":
    main()
