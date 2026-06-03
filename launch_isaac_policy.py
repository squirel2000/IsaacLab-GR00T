#!/usr/bin/env python3
"""Launch a policy server and IsaacLab client in separate terminals."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import psutil

# Example of usage:
# python launch_isaac_policy.py \
#     --policy starvla \
#     --model-path /home/asus/Gits/IsaacLab-GR00T/artifacts/checkpoints/gr00t/openarm_linkerhando6_cansorting_N15_fft_20k_dataset_0408_rtc \
#     --max-eps-num 50 \
#     --save-video


# This script lives at the workspace root, so its directory IS the workspace root.
PROJECT_ROOT = Path(__file__).resolve().parent
POLICY_CONFIGS = {
    "starvla": PROJECT_ROOT / "IsaacLab/scripts/gr00t_script/policy_configs/starvla_openarm_o6.json",
    "gr00t": PROJECT_ROOT / "IsaacLab/scripts/gr00t_script/policy_configs/gr00t_n15_openarm_o6.json",
}


def _resolve_path(value: str | os.PathLike[str]) -> Path:
    """Expand env vars / `~` and resolve relative paths against PROJECT_ROOT."""
    path = Path(os.path.expandvars(os.fspath(value))).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path

CLIENT = {
    "dir": PROJECT_ROOT / "IsaacLab",
    "script": "scripts/gr00t_script/gr00t_infer_agent.py",
    "conda_env": "env_isaaclab",
    "title": "IsaacLab Client",
}

SERVERS = {
    "starvla": {"script": "deployment/model_server/server_policy.py", "conda_env": "starVLA",   "title": "StarVLA Server"},
    "gr00t":   {"script": "scripts/inference_service.py",             "conda_env": "env_gr00t", "title": "Isaac GR00T Server"},
}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def server(policy, cfg):
    s = SERVERS[policy].copy()
    s["dir"] = _resolve_path(cfg["starvla_repo"]) if policy == "starvla" else PROJECT_ROOT / "Isaac-GR00T"
    return s


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


def client_args(policy, cfg, config_path, max_eps_num, save_video, save_dir):
    args = ["python3", "-u", CLIENT["script"], 
            "--task", cfg["task"], 
            "--policy", policy, 
            "--policy_config", str(config_path), 
            "--host", cfg["host"], 
            "--port", str(cfg["port"]), 
            "--max_eps_num", str(max_eps_num), 
            "--openarm_hand_type", cfg.get("openarm_hand_type", "linkerhand_o6"), 
            "--filter", 
            "--save_dir", str(save_dir)]
    if policy == "gr00t":
        args += ["--gr00t_ver", cfg.get("version", "N1.5")]
    if save_video:
        args.append("--save_video")
    return args


def main():
    p = argparse.ArgumentParser(description="Launch policy server and IsaacLab client.")
    p.add_argument("--policy", choices=sorted(POLICY_CONFIGS), default="gr00t", help="Policy backend to launch. Default: gr00t.")
    p.add_argument("--model-path", type=Path, help="Override checkpoint/model path from the policy JSON.")
    p.add_argument("--max-eps-num", type=int, default=100, help="[Client] Max episodes. Default: 100.")
    p.add_argument("--save-video", action="store_true", help="[Client] Save rollout video. Default: False.")
    args = p.parse_args()

    # Load policy config and determine model path and server config
    policy_cfg = load_json(POLICY_CONFIGS[args.policy])
    model_path = _resolve_path(args.model_path or policy_cfg["model_path"])
    server_cfg = server(args.policy, policy_cfg)

    mp = Path(model_path)
    ckpt_id = f"{mp.parent.name}_{mp.name}" if mp.name.startswith("checkpoint-") else mp.name
    save_dir = f"output/infer_record/{ckpt_id}"

    # Check required paths before launching anything
    required = [server_cfg["dir"], CLIENT["dir"], POLICY_CONFIGS[args.policy], model_path]
    if args.policy == "starvla":
        required.append(_resolve_path(policy_cfg["stats_path"]))
    for path in required:
        if not Path(path).exists():
            sys.exit(f"Error: path not found: {path}")

    # Launch a server if not already running (to avoid accidentally launching multiple servers)
    if server_running(args.policy, policy_cfg["port"]):
        print(f"{args.policy} server already running - skipping server launch.")
    else:
        launch_in_terminal(server_cfg, server_args(args.policy, policy_cfg, model_path))

    # Launch the client
    launch_in_terminal(CLIENT, client_args(args.policy, policy_cfg, POLICY_CONFIGS[args.policy], args.max_eps_num, args.save_video, save_dir),)


if __name__ == "__main__":
    main()
