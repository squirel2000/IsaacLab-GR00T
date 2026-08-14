"""Local live monitor for a detached RLinf training run on Pegasus.

Surfaces "improvement + progress" on the local terminal by polling the remote
run log (no long-lived connection): it finds the run's log dir, tails the log,
and pulls out the latest success-rate / reward values it can find. Detects
terminal states via the training PID and/or completion markers.

Usage:
  python agents/rl-trainbot/harness/poll.py --config libero_spatial_ppo_gr00t_n1d7_smoke [--pid 12345] [--once]

The metric regexes are intentionally tolerant; they are tightened once the smoke
test shows RLinf's exact stdout format. The authoritative success scalar is the
TensorBoard tag `env/success_once` (see report.py for the precise extraction).
"""
from __future__ import annotations

import argparse
import pathlib
import re
import time

import yaml

import remote  # agents/rl-trainbot/harness/remote.py

_HERE = pathlib.Path(__file__).resolve().parent
_CFG = _HERE / "config" / "phase1.yaml"

# Tolerant: match e.g. "env/success_once: 0.37", "success_rate=0.4", "rollout reward 1.23"
_SUCCESS_RE = re.compile(r"(success[\w/]*)\s*[:=]?\s*([01]?\.\d+|\d+\.?\d*)", re.I)
_REWARD_RE = re.compile(r"(reward[\w/]*)\s*[:=]?\s*(-?\d+\.?\d*)", re.I)


def load_cfg(cfg_path=_CFG):
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def latest_log_dir(s, rlinf_dir, config_name):
    """Most-recent logs/<ts>-<config_name>/ dir on Pegasus, or None."""
    out, _ = remote.sh(s, f"ls -dt {rlinf_dir}/logs/*-{config_name} 2>/dev/null | head -1")
    out = out.strip()
    return out.splitlines()[0].strip() if out else None


def poll_once(s, cfg, config_name, pid):
    rlinf = cfg["remote"]["rlinf_dir"]
    logdir = latest_log_dir(s, rlinf, config_name)
    if not logdir:
        print("  (no log dir yet — process may still be starting)")
        return None
    tail = remote.tail(s, f"{logdir}/run_embodiment.log", n=40)
    succ = _SUCCESS_RE.findall(tail)
    rew = _REWARD_RE.findall(tail)
    print(f"  logdir: {logdir}")
    if succ:
        print(f"  latest success : {succ[-1][0]} = {succ[-1][1]}")
    if rew:
        print(f"  latest reward  : {rew[-1][0]} = {rew[-1][1]}")
    print("  --- recent log ---")
    for line in tail.splitlines()[-12:]:
        print("    " + line)
    return remote.pid_alive(s, pid) if pid else None


def main():
    ap = argparse.ArgumentParser(description="Monitor a detached RLinf run on Pegasus")
    ap.add_argument("--config", required=True, help="config name (e.g. ..._smoke or ..._phase1)")
    ap.add_argument("--pid", default=None, help="remote training PID (enables done/died detection)")
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    ap.add_argument("--config-file", default=str(_CFG), help="path to the orchestration yaml (default: config/phase1.yaml)")
    args = ap.parse_args()

    cfg = load_cfg(args.config_file)
    interval = cfg["poll"]["interval_sec"]
    s = remote.connect()
    while True:
        print(f"[{time.strftime('%H:%M:%S')}] polling {args.config}", flush=True)
        try:
            alive = poll_once(s, cfg, args.config, args.pid)
        except Exception as e:
            # A single flaky call (kernel-create timeout, dropped websocket, ...) must not
            # kill hours-long monitoring; log it, reconnect, and keep going.
            print(f"  !! poll failed: {e!r} -- reconnecting", flush=True)
            alive = None
            s = remote.connect()
        if args.once:
            break
        if args.pid and alive is False:
            print("  >> training process is no longer running (finished or died).", flush=True)
            break
        time.sleep(interval)
        s = remote.connect()  # refresh session for long polls
    print("done.")


if __name__ == "__main__":
    main()
