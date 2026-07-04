#!/usr/bin/env python3
"""Top-level scheduler for GR00T / StarVLA closed-loop eval in IsaacSim.

For each run in configs/eval_config.yaml: if a complete log already exists, reuse
it; otherwise launch the policy server + IsaacSim client (crash-resilient: the
server stays up and the client relaunches until `target` episodes are collected).
Then hand the results to analysis/ to print a comparison and write the figure
(agents/evalbot/var/analysis/eval_results.svg).

This file only schedules: config, reuse-or-eval, server/client lifecycle. Aggregation,
the comparison table and the chart live in analysis/ (aggregate.py, compare_runs.py,
make_loss_svg.py). So re-running with no new checkpoints just re-summarises + re-charts;
add a run to eval_config.yaml (or delete its log) to make it (re)eval.

  python agents/evalbot/harness/run_eval.py              # eval missing runs, then compare + chart
  python agents/evalbot/harness/run_eval.py --target 50  # 50 episodes per run

headless and everything else are set in configs/eval_config.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml

from analysis import compare_runs
from analysis.aggregate import aggregate, count_eps, final_train_loss

HERE = Path(__file__).resolve().parent       # the eval harness dir


def _find_workspace_root() -> Path:
    """Walk up until the workspace.yaml root marker (survives directory moves)."""
    for d in (HERE, *HERE.parents):
        if (d / "workspace.yaml").is_file():
            return d
    return HERE.parents[1]                    # legacy fallback (scripts/eval -> root)


ROOT = _find_workspace_root()                 # IsaacLab-GR00T workspace root
AGENT = HERE / "gr00t_infer_agent.py"
DEFAULT_CONFIG = HERE / "configs" / "eval_config.yaml"


def log(msg: str) -> None:
    print(f"[{datetime.now():%F %T}] {msg}", flush=True)


def fmt_elapsed(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def resolve_path(value) -> Path:
    """Relative paths resolve against the repo root (checkpoints, repos, pythonpath)."""
    p = Path(os.path.expandvars(os.fspath(value))).expanduser()
    return p if p.is_absolute() else ROOT / p


def find_conda_sh(configured) -> str:
    """Use the configured conda.sh, else auto-detect from CONDA_EXE / PATH."""
    if configured:
        return os.path.expandvars(os.path.expanduser(configured))
    for exe in (os.environ.get("CONDA_EXE"), shutil.which("conda")):
        if exe:
            sh = Path(exe).resolve().parents[1] / "etc" / "profile.d" / "conda.sh"
            if sh.exists():
                return str(sh)
    raise SystemExit("conda.sh not found — set env.conda_sh in eval_config.yaml")


def resolve_checkpoint(path: Path) -> Path:
    """A parent dir holding checkpoint-* subdirs resolves to the latest (highest step)."""
    if path.name.startswith("checkpoint-"):
        return path
    ckpts = [d for d in path.glob("checkpoint-*") if d.is_dir() and d.name[len("checkpoint-"):].isdigit()]
    return max(ckpts, key=lambda d: int(d.name[len("checkpoint-"):])) if ckpts else path


# --------------------------------------------------------------------------
# Backend command construction. configs/*.json is the per-backend spec.
# --------------------------------------------------------------------------
def server_workdir_activate(policy: str, cfg: dict, env: dict) -> tuple[Path, str]:
    conda_sh = env["conda_sh"]
    if policy == "starvla":
        return resolve_path(cfg["starvla_repo"]), f"source {shlex.quote(conda_sh)} && conda activate starVLA"
    workdir = resolve_path(cfg.get("server_repo", "engines/vla/Isaac-GR00T"))
    if cfg.get("server_venv"):  # N1.7 lives in a uv/.venv, not conda
        return workdir, f"source {shlex.quote(str(workdir / cfg['server_venv']))}/bin/activate"
    return workdir, f"source {shlex.quote(conda_sh)} && conda activate {cfg.get('server_conda_env', 'env_gr00t')}"


def server_cmd(policy: str, cfg: dict, ckpt: Path) -> list[str]:
    if policy == "starvla":
        a = ["python3", "-u", "deployment/model_server/server_policy.py",
             "--ckpt_path", str(ckpt), "--port", str(cfg["port"]),
             "--idle_timeout", str(cfg.get("idle_timeout", -1))]
        if "denoising_steps" in cfg:
            a += ["--denoising_steps", str(cfg["denoising_steps"])]
        return a + (["--use_bf16"] if cfg.get("use_bf16", True) else [])
    # GR00T N1.6 tyro parses the enum NAME (NEW_EMBODIMENT); N1.7 takes the value.
    ver = cfg.get("version", "N1.5")
    emb = cfg["embodiment_tag"].upper() if ver == "N1.6" else cfg["embodiment_tag"]
    return ["python3", "-u", "-m", "gr00t.eval.run_gr00t_server",
            "--model-path", str(ckpt), "--embodiment-tag", emb, "--port", str(cfg["port"])]


def client_cmd(r: dict, n_eps: int, d: dict, headless: bool) -> list[str]:
    cfg = r["cfg"]
    a = ["python3", "-u", str(AGENT),
         "--task", cfg.get("task", d["task"]),
         "--policy", r["policy"], "--policy_config", str(r["cfg_path"]),
         "--host", cfg.get("host", "localhost"), "--port", str(cfg["port"]),
         "--max_eps_num", str(n_eps),
         "--openarm_hand_type", cfg.get("openarm_hand_type", d["openarm_hand_type"]),
         "--save_dir", r["save_dir"],
         "--pov_list", *d["pov_list"]]
    if r["policy"] == "gr00t":
        a += ["--gr00t_ver", cfg.get("version", "N1.5")]
    if d["multitask"]:
        a.append("--multitask")
    if d["filter"]:
        a.append("--filter")
    if d["save_video"]:
        a.append("--save_video")
    if headless:
        a.append("--headless")
    return a


def client_env(base: dict, cfg: dict) -> dict:
    """Client env, plus the alternate gr00t checkout on PYTHONPATH (N1.7 ZMQ wire format)."""
    e = dict(base)
    if cfg.get("client_pythonpath"):
        e["PYTHONPATH"] = str(resolve_path(cfg["client_pythonpath"])) + os.pathsep + e.get("PYTHONPATH", "")
    return e


def load_run(run: dict, cfg_dir: Path) -> dict:
    """Resolve a YAML run entry into concrete server/client parameters."""
    cfg_path = Path(run["config"])
    cfg_path = cfg_path if cfg_path.is_absolute() else cfg_dir / cfg_path
    cfg = json.loads(cfg_path.read_text())
    return {
        "tag": run["tag"], "policy": run.get("policy", "gr00t"), "cfg": cfg, "cfg_path": cfg_path,
        "ckpt": resolve_checkpoint(resolve_path(run.get("checkpoint") or cfg["model_path"])),
        "save_dir": f"output/infer_record/{run['save_id']}",  # relative to IsaacLab cwd
    }


# --------------------------------------------------------------------------
# Batch eval (crash-resilient server/client loop).
# --------------------------------------------------------------------------
def kill_server(proc) -> None:
    """Kill only this run's server (its own process group), so parallel runs don't kill each other."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def wait_ready(server_log: Path, proc, ready: str, timeout: int) -> int:
    waited = 0
    while waited < timeout:
        if server_log.exists() and ready in server_log.read_text(errors="ignore"):
            return 0
        if proc.poll() is not None:
            log(f"SERVER PROCESS {proc.pid} DIED")
            return 2
        time.sleep(4)
        waited += 4
    return 1


def server_error_hint(server_log: Path) -> str:
    """A short reason for a failed server launch, pulled from its log."""
    if not server_log.exists():
        return ""
    txt = server_log.read_text(errors="ignore")
    if "Address already in use" in txt:
        return " (port busy — an old server is still running; run: pkill -9 -f run_gr00t_server)"
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    return f" (last server log: {lines[-1][:160]})" if lines else ""


def run_batch(r: dict, d: dict, env: dict, isaaclab: Path, logdir: Path, headless: bool, port=None) -> None:
    tag = r["tag"]
    t0 = time.time()
    if port is not None:
        r["cfg"]["port"] = port  # parallel mode: a distinct port per concurrent run
    combined = logdir / f"{tag}_combined_episodes.log"
    combined.write_text("")
    server_log = logdir / f"{tag}_server.log"
    workdir, activate = server_workdir_activate(r["policy"], r["cfg"], env)
    scmd = f"cd {shlex.quote(str(workdir))} && {activate} && exec {shlex.join(server_cmd(r['policy'], r['cfg'], r['ckpt']))}"

    def launch_server():
        # start_new_session so kill_server() can reap exactly this server's process group.
        return subprocess.Popen(["bash", "-c", scmd], stdout=open(server_log, "w"),
                                stderr=subprocess.STDOUT, env=env, start_new_session=True)

    log(f"=== {tag}: launching server ({r['cfg'].get('embodiment_tag', '?')}) on :{r['cfg']['port']} — ckpt {r['ckpt'].name} ===")
    proc = launch_server()
    try:
        if wait_ready(server_log, proc, d["server_ready_string"], d["server_ready_timeout_s"]) != 0:
            log(f"=== {tag}: SERVER NOT READY{server_error_hint(server_log)} -- skipping ===")
            return
        log(f"=== {tag}: server ready (pid {proc.pid}) ===")

        cenv = client_env(env, r["cfg"])
        attempt, done = 0, 0
        while done < d["target"] and attempt < d["max_attempts"]:
            attempt += 1
            need = d["target"] - done
            if proc.poll() is not None:
                log(f"=== {tag}: server died; restarting ===")
                proc = launch_server()
                if wait_ready(server_log, proc, d["server_ready_string"], d["server_ready_timeout_s"]) != 0:
                    log(f"=== {tag}: server restart failed -- abort ===")
                    break
            clog = logdir / f"{tag}_client_attempt{attempt}.log"
            log(f"=== {tag}: client attempt {attempt} -- need {need} more (have {done}/{d['target']}) ===")
            cc = shlex.join(client_cmd(r, need, d, headless))
            cbash = f"cd {shlex.quote(str(isaaclab))} && source {shlex.quote(env['conda_sh'])} && conda activate {env['isaaclab_conda_env']} && exec {cc}"
            rc = 0
            with open(clog, "w") as f:
                try:
                    rc = subprocess.run(["bash", "-c", cbash], stdout=f, stderr=subprocess.STDOUT,
                                        env=cenv, timeout=d["client_timeout_s"]).returncode
                except subprocess.TimeoutExpired:
                    log(f"=== {tag}: client attempt {attempt} hit {d['client_timeout_s']}s timeout ===")
                    rc = 124
            got = [ln for ln in clog.read_text(errors="ignore").splitlines() if "finished after" in ln]
            with combined.open("a") as f:
                if got:
                    f.write("\n".join(got) + "\n")
            done = count_eps(combined)
            log(f"=== {tag}: attempt {attempt} rc={rc}, +{len(got)} eps -> {done}/{d['target']} ===")
            if rc == 0 and len(got) >= need:
                break
            time.sleep(5)
        a = aggregate(combined)
        loss = final_train_loss(r["ckpt"] / "trainer_state.json")
        loss_s = f"{loss:.4f}" if loss is not None else "—"
        log(f"=== {tag}: DONE -- success {a['success']}/{a['n']} ({100 * a['rate']:.0f}%) | "
            f"timeout {a['truncated']} | unsafe {a['terminated']} | loss {loss_s} | "
            f"elapsed {fmt_elapsed(time.time() - t0)} | {attempt} attempt(s) ===")
    finally:
        # Always reap THIS run's server — incl. on Ctrl-C / exception — so it never orphans on :port.
        kill_server(proc)
        time.sleep(2)


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="GR00T/StarVLA closed-loop eval — top-level scheduler.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="eval_config.yaml path (default: <harness>/configs/eval_config.yaml).")
    p.add_argument("--target", type=int, help="Episodes per run (default: from config, 100).")
    args = p.parse_args()

    spec = yaml.safe_load(args.config.read_text())
    d, env_cfg, runs = spec["defaults"], spec["env"], spec["runs"]
    cfg_dir = args.config.resolve().parent
    if args.target is not None:
        d["target"] = args.target
    headless = d["headless"]

    isaaclab = resolve_path(env_cfg["isaaclab_repo"])
    run_env = dict(os.environ, conda_sh=find_conda_sh(env_cfg.get("conda_sh")),
                   isaaclab_conda_env=env_cfg["isaaclab_conda_env"])
    if not headless:
        run_env.update({k: str(v) for k, v in env_cfg.get("display", {}).items()})

    logdir = resolve_path(d["logdir"])  # one logdir keyed by tag; headless/windowed share it (so reuse works across modes)
    logdir.mkdir(parents=True, exist_ok=True)
    jobs = max(1, int(d.get("jobs", 1)))
    log(f"########## EVAL START (target={d['target']}, headless={headless}, jobs={jobs}, logdir={logdir}) ##########")

    to_eval = []
    for run in runs:
        have = count_eps(logdir / f"{run['tag']}_combined_episodes.log")
        if have >= d["target"]:
            log(f"=== {run['tag']}: reuse existing log ({have} eps ≥ {d['target']}) — skip eval ===")
        else:
            to_eval.append(run)

    if jobs > 1 and len(to_eval) > 1:
        # Each concurrent run gets its own server port (5555+i) and is reaped independently.
        log(f"=== evaluating {len(to_eval)} runs, up to {jobs} in parallel ===")
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(run_batch, load_run(run, cfg_dir), d, run_env, isaaclab, logdir, headless, 5555 + i)
                       for i, run in enumerate(to_eval)]
            for f in futures:
                f.result()
    else:
        for run in to_eval:
            run_batch(load_run(run, cfg_dir), d, run_env, isaaclab, logdir, headless)

    # hand off to analysis: comparison table + figure
    items = [(run["tag"], logdir / f"{run['tag']}_combined_episodes.log", load_run(run, cfg_dir)["ckpt"])
             for run in runs if count_eps(logdir / f"{run['tag']}_combined_episodes.log")]
    chart = HERE.parent / "var" / "analysis" / "eval_results.svg"
    chart.parent.mkdir(parents=True, exist_ok=True)
    compare_runs.report(items, chart)
    log(f"comparison chart -> {chart}")


if __name__ == "__main__":
    main()
