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
    "gr00t": PROJECT_ROOT / "IsaacLab/scripts/gr00t_script/policy_configs/gr00t_n16_openarm_o6.json",
}


def gr00t_server_script(version):
    """GR00T inference-server entrypoint (relative to Isaac-GR00T), chosen by version.

    N1.6/N1.7 reorganized the deployment scripts: the old `scripts/inference_service.py`
    only exists on the N1.5 branch, replaced by `gr00t/eval/run_gr00t_server.py`.
    """
    if version in ("N1.6", "N1.7"):
        return "gr00t/eval/run_gr00t_server.py"
    return "scripts/inference_service.py"


def _resolve_path(value: str | os.PathLike[str]) -> Path:
    """Expand env vars / `~` and resolve relative paths against PROJECT_ROOT."""
    path = Path(os.path.expandvars(os.fspath(value))).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_checkpoint(path: Path) -> Path:
    """If `path` is a parent dir holding checkpoint-* subdirs, return the latest (highest
    step). A path already pointing at a checkpoint-* (or with no checkpoints) is unchanged."""
    if path.name.startswith("checkpoint-"):
        return path
    ckpts = [d for d in path.glob("checkpoint-*") if d.is_dir() and d.name[len("checkpoint-"):].isdigit()]
    if not ckpts:
        return path
    return max(ckpts, key=lambda d: int(d.name[len("checkpoint-"):]))

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
    if policy == "starvla":
        s["dir"] = _resolve_path(cfg["starvla_repo"])
    else:
        # server_repo lets a config target a different GR00T checkout (N1.7 lives in
        # Isaac-GR00T_n1d7). It activates either a uv/.venv (server_venv, resolved against
        # the repo) or a conda env (server_conda_env); venv takes precedence.
        s["dir"] = PROJECT_ROOT / cfg.get("server_repo", "Isaac-GR00T")
        if cfg.get("server_venv"):
            s["venv"] = s["dir"] / cfg["server_venv"]
        s["conda_env"] = cfg.get("server_conda_env", s["conda_env"])
    return s


def server_running(policy, cfg):
    port = cfg["port"]
    version = cfg.get("version", "N1.5")
    if policy == "starvla":
        token = "server_policy.py"
    elif version in ("N1.6", "N1.7"):
        # Matches both the script path (run_gr00t_server.py) and the `-m` form
        # (gr00t.eval.run_gr00t_server) we actually launch with.
        token = "run_gr00t_server"
    else:
        token = "inference_service.py"
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmd = proc.info.get("cmdline") or []
            if not any(token in c for c in cmd):
                continue
            if policy == "gr00t":
                # N1.6/N1.7 entrypoint name is unique; N1.5 shares the inference_service.py
                # name with the IsaacLab client, so require the "--server" flag to disambiguate.
                if version in ("N1.6", "N1.7") or "--server" in cmd:
                    return True
            if policy == "starvla" and str(port) in cmd:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def launch_in_terminal(cfg, command, pythonpath=None):
    # Activate either a uv/.venv (N1.7 server lives in a venv, not conda) or a conda env.
    if cfg.get("venv"):
        activate = f"source {shlex.quote(str(cfg['venv']))}/bin/activate"
    else:
        activate = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {cfg['conda_env']}"
    # PYTHONPATH lets the IsaacLab client (env_isaaclab, N1.6 gr00t) import an alternate
    # gr00t checkout (N1.7) so it speaks the matching msgpack_numpy ZMQ wire format.
    prefix = f"PYTHONPATH={shlex.quote(str(pythonpath))}:$PYTHONPATH " if pythonpath else ""
    subprocess.Popen([
        "gnome-terminal", f"--title={cfg['title']}", f"--working-directory={cfg['dir']}", "--", "bash", "-c",
        f"{activate} && {prefix}{shlex.join(command)}; "
        f'echo "Stopped. Press Enter to close."; read',
    ])


def server_args(policy, cfg, model_path):
    if policy == "starvla":
        args = ["python", "-u", "deployment/model_server/server_policy.py", "--ckpt_path", str(model_path), "--port", str(cfg["port"]), "--idle_timeout", str(cfg.get("idle_timeout", -1))]
        if "denoising_steps" in cfg:
            args += ["--denoising_steps", str(cfg["denoising_steps"])]
        return args + (["--use_bf16"] if cfg.get("use_bf16", True) else [])
    version = cfg.get("version", "N1.5")
    if version in ("N1.6", "N1.7"):
        # N1.6/N1.7 server uses a tyro CLI: hyphenated flags. No --use-sim-policy-wrapper:
        # the IsaacLab client adapter (Gr00tClientAdapter._format_obs) already sends the
        # nested observation format Gr00tPolicy expects; that wrapper is only for sims that
        # send flat video.* keys.
        # Embodiment tag differs by version: N1.6 tyro parses the enum NAME (NEW_EMBODIMENT);
        # N1.7 takes a lowercase value (new_embodiment) and calls EmbodimentTag.resolve.
        # `-m` makes the server repo's working dir pick the gr00t version, independent of the
        # editable install the conda env registered.
        embodiment = cfg["embodiment_tag"].upper() if version == "N1.6" else cfg["embodiment_tag"]
        return ["python3", "-u", "-m", "gr00t.eval.run_gr00t_server",
                "--model-path", str(model_path),
                "--embodiment-tag", embodiment,
                "--port", str(cfg["port"])]
    script = gr00t_server_script(version)
    return ["python3", "-u", script, "--server", "--model_path", str(model_path), "--embodiment_tag", cfg["embodiment_tag"], "--data_config", cfg["data_config"], "--denoising_steps", str(cfg["denoising_steps"]), "--port", str(cfg["port"])]


def client_args(policy, cfg, config_path, max_eps_num, save_video, save_dir, headless):
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
    if headless:
        args.append("--headless")
    return args


def main():
    p = argparse.ArgumentParser(description="Launch policy server and IsaacLab client.")
    p.add_argument("--policy", choices=sorted(POLICY_CONFIGS), default="gr00t", help="Policy backend to launch. Default: gr00t.")
    p.add_argument("--config", type=Path, help="Override the policy JSON path (e.g. the N1.7 config).")
    p.add_argument("--model-path", type=Path, help="Override checkpoint/model path from the policy JSON. A parent dir auto-resolves to its latest checkpoint-*.")
    p.add_argument("--max-eps-num", type=int, default=100, help="[Client] Max episodes. Default: 100.")
    p.add_argument("--save-video", action="store_true", help="[Client] Save rollout video. Default: False.")
    p.add_argument("--headless", action="store_true", help="[Client] Run IsaacSim headless (no GUI window). Cameras still render offscreen. Default: False.")
    args = p.parse_args()

    # Load policy config and determine model path and server config
    config_path = args.config or POLICY_CONFIGS[args.policy]
    policy_cfg = load_json(config_path)
    model_path = _resolve_checkpoint(_resolve_path(args.model_path or policy_cfg["model_path"]))
    server_cfg = server(args.policy, policy_cfg)
    print(f"Using model path: {model_path}")

    mp = Path(model_path)
    ckpt_id = f"{mp.parent.name}_{mp.name}" if mp.name.startswith("checkpoint-") else mp.name
    save_dir = f"output/infer_record/{ckpt_id}"

    # Check required paths before launching anything
    required = [server_cfg["dir"], CLIENT["dir"], config_path, model_path]
    if args.policy == "starvla":
        required.append(_resolve_path(policy_cfg["stats_path"]))
    for path in required:
        if not Path(path).exists():
            sys.exit(f"Error: path not found: {path}")

    # Launch a server if not already running (to avoid accidentally launching multiple servers)
    if server_running(args.policy, policy_cfg):
        print(f"{args.policy} server already running - skipping server launch.")
    else:
        launch_in_terminal(server_cfg, server_args(args.policy, policy_cfg, model_path))

    # Launch the client. client_pythonpath points the client at an alternate gr00t checkout
    # (N1.7) so its PolicyClient matches the server's ZMQ wire format.
    cpp = policy_cfg.get("client_pythonpath")
    client_pp = PROJECT_ROOT / cpp if cpp else None
    launch_in_terminal(CLIENT, client_args(args.policy, policy_cfg, config_path, args.max_eps_num, args.save_video, save_dir, args.headless), pythonpath=client_pp)


if __name__ == "__main__":
    main()
