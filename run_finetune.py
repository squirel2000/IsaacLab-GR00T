#!/usr/bin/env python3
r"""End-to-end GR00T fine-tuning automation on the Pegasus remote box.

Pipeline (every stage survives a local disconnect):
    1. start    : upload a wrapper script and launch it DETACHED on the server
                  (setsid+nohup) so training keeps running even if this PC drops /
                  the kernel dies.  The wrapper:
                       conda activate <env>  ->  run fine-tuning  ->  (on success)
                       delete extra checkpoints  ->  zip  ->  sha256 + size.
    2. monitor  : tail the server-side log over HTTP until status is DONE/FAILED.
                  Re-runnable any time after a disconnect (stateless polling).
    3. download : resumable + sha256-verified download of the .zip to the local dir,
                  so a dropped connection never leaves a half / corrupt file.

Usage (PowerShell):
    $env:PEGASUS_PASSWORD = 'eksncl#20260410_PE'
    python run_finetune.py run         # start training, then monitor + download
    python run_finetune.py monitor     # re-attach + stream the full live log after a disconnect
    python run_finetune.py watch       # compact one-line status, refreshed every POLL_SECONDS
    python run_finetune.py download    # (re)download + verify the latest run's artifact
    python run_finetune.py status      # print current server-side status
    python run_finetune.py selftest    # validate the whole pipeline with a tiny dummy job
    python run_finetune.py stop        # kill the detached job of the latest run

Edit the CONFIG block below to change the command / parameters / paths.
"""
import argparse, json, os, re, shlex, sys, time, datetime, pathlib
import pegasus as pg

# ===========================================================================
#  CONFIG  -- edit here
# ===========================================================================
# How to enter the conda env on this box (env is a path-based env, activate by path):
CONDA_ACTIVATE = ('eval "$(/home/gallop/miniforge3/bin/conda shell.bash hook)" && '
                  'conda activate /data/VLA/tingying/envs/gr00t_n1d5')

# Directory to run the fine-tuning command from:
TRAIN_CWD = "/data/VLA/tingying/IsaacLab-GR00T/Isaac-GR00T"
DATASET_PATH = "/data/VLA/datasets/OpenArm_O6_CanSorting_Dataset_0408"

# Where the training writes its checkpoints:
OUTPUT_DIR = ("/data/VLA/experiments/openarmlinkerhando6-can-sorting-checkpoints/"
              "new_embodiment/N1_5_100k_dataset_0408_rtc_stage1")

# Total training steps -- MUST match --max-steps below.  The final checkpoint is
# named checkpoint-<MAX_STEPS>; that is the one we keep after cleanup.
MAX_STEPS = 100000

# The fine-tuning command (exactly as you would type it after activating the env).
TRAIN_CMD = (
    f"TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MPLBACKEND=Agg "
    f"python3 scripts/gr00t_finetune.py "
    f"--dataset-path {DATASET_PATH} "
    f"--output-dir {OUTPUT_DIR} "
    "--gpu-id 0 --num-gpus 1 --batch-size 32 --video-backend torchvision_av "
    "--data-config openarm_linkerhand_o6 --embodiment_tag new_embodiment "
    f"--max-steps {MAX_STEPS} --save-steps 2000 --eval_steps 1000 --save-total-limit 5 "
    "--eval-args-trajs 2 --eval-args-max-steps 400 --dataset-split-ratio \"9:1\" "
    "--dataloader-num-workers 4 --denoising_step 4 --learning-rate 1e-4 "
    "--eval-args-action-horizon 16"
)

# Post-training cleanup, executed INSIDE OUTPUT_DIR (where the checkpoint-* dirs live).
# Keep ONLY the final checkpoint (checkpoint-<MAX_STEPS>); delete every other checkpoint-*.
# `find` makes this robust regardless of which intermediate step numbers exist.
KEEP_CHECKPOINT = f"checkpoint-{MAX_STEPS}"               # e.g. checkpoint-100000
CLEANUP_CMD = (f"find . -maxdepth 1 -type d -name 'checkpoint-*' "
               f"! -name {shlex.quote(KEEP_CHECKPOINT)} -exec rm -rf {{}} +")

# Zip: created in the PARENT of OUTPUT_DIR, archiving the output dir by name.
ZIP_PARENT = os.path.dirname(OUTPUT_DIR)
ZIP_NAME = os.path.basename(OUTPUT_DIR) + ".zip"          # N1_5_100k_..._stage1.zip
ZIP_CMD = f"rm -f {shlex.quote(ZIP_NAME)} && zip -r -y {shlex.quote(ZIP_NAME)} " \
          f"{shlex.quote(os.path.basename(OUTPUT_DIR) + '/')}"

# Local download destination (the verified .zip lands here):
LOCAL_DIR = r"D:\tmp\IsaacLab-GR00T\artifacts\checkpoints"

# Server-side scratch dir for run state/logs.  Lives in YOUR private folder
# (not the shared /data/VLA), and must NOT be hidden (Jupyter hides dot-dirs).
STATE_ROOT = "/data/VLA/tingying/pegasus_runs"

POLL_SECONDS = 30           # how often monitor polls the server
# ===========================================================================


def q(p):
    """shlex.quote shortcut -- safely embed a path/string inside a shell command."""
    return shlex.quote(p)


def wrapper_script(cfg, state_abs):
    """Generate the detached server-side bash wrapper for one run."""
    return f"""#!/bin/bash
# auto-generated by run_finetune.py -- do not edit on the server
STATE={q(state_abs)}
echo RUNNING > "$STATE/status"
{{
  echo "[wrapper] $(date '+%F %T') start"
  set -x
  {cfg['conda_activate']} || {{ set +x; echo "FAILED_CONDA" > "$STATE/status"; exit 11; }}
  cd {q(cfg['train_cwd'])}  || {{ set +x; echo "FAILED_CD_TRAIN" > "$STATE/status"; exit 12; }}
  echo "[wrapper] training begins $(date '+%F %T')"
  {cfg['train_cmd']}
  rc=$?
  set +x
  echo "[wrapper] training exit code = $rc"
  if [ $rc -ne 0 ]; then echo "FAILED_TRAIN:$rc" > "$STATE/status"; exit $rc; fi

  cd {q(cfg['output_dir'])} || {{ echo "FAILED_CD_OUT" > "$STATE/status"; exit 13; }}
  echo "[wrapper] checkpoints before cleanup:"; ls -d checkpoint-* 2>/dev/null
  echo "[wrapper] cleanup (keep {cfg['keep_checkpoint']}): {cfg['cleanup_cmd']}"
  ( {cfg['cleanup_cmd']} ) 2>/dev/null || true
  echo "[wrapper] checkpoints after cleanup:"; ls -d checkpoint-* 2>/dev/null

  cd {q(cfg['zip_parent'])} || {{ echo "FAILED_CD_ZIP" > "$STATE/status"; exit 14; }}
  echo "[wrapper] zipping -> {cfg['zip_name']}"
  {cfg['zip_cmd']}
  zrc=$?
  if [ $zrc -ne 0 ]; then echo "FAILED_ZIP:$zrc" > "$STATE/status"; exit $zrc; fi

  echo "[wrapper] hashing {cfg['zip_name']}"
  sha256sum {q(cfg['zip_name'])} | awk '{{print $1}}' > "$STATE/zip.sha256"
  stat -c %s {q(cfg['zip_name'])} > "$STATE/zip.size"
  echo "{cfg['zip_parent']}/{cfg['zip_name']}" > "$STATE/zip.path"
  echo DONE > "$STATE/status"
  echo "[wrapper] all done $(date '+%F %T')"
}} >> "$STATE/run.log" 2>&1
"""


# --------------------------------------------------------------------------- #
#  Local run-state bookkeeping (so we can re-attach after a disconnect)
# --------------------------------------------------------------------------- #
def state_file(cfg):
    """Path to the local JSON that remembers the latest run (used to re-attach)."""
    return pathlib.Path(cfg["local_dir"]) / (cfg["run_name"] + ".run.json")


def save_state(cfg, info):
    """Persist this run's identifiers to the local state JSON."""
    p = state_file(cfg); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info, indent=2), encoding="utf-8")


def load_state(cfg):
    """Load the latest run's identifiers (errors if no run was ever started)."""
    p = state_file(cfg)
    if not p.exists():
        raise SystemExit(f"no run state found ({p}); start a run first with `run`.")
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
#  Server file helpers (over HTTP /files, with re-login on expiry)
# --------------------------------------------------------------------------- #
def fetch(s, rel, offset=0):
    """Return bytes of a server file from `offset` (b'' if absent / no new data)."""
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    try:
        r = s.get(pg.files_url(rel), params={"_xsrf": s.cookies.get("_xsrf")},
                  headers=headers, timeout=(30, 60))
    except Exception as e:                               # network blip -> caller retries
        sys.stderr.write(f"\n(fetch retry: {type(e).__name__})\n"); time.sleep(5); return b""
    if r.status_code in (401, 403):
        pg.relogin(s); return fetch(s, rel, offset)      # session expired -> re-auth
    if r.status_code in (404, 416):                      # not created yet / no new bytes
        return b""
    if r.status_code in (200, 206):
        return r.content
    return b""


def read_text(s, rel):
    """Fetch a small server file and return it as stripped UTF-8 text."""
    return fetch(s, rel).decode("utf-8", "replace").strip()


# --------------------------------------------------------------------------- #
#  Pipeline stages
# --------------------------------------------------------------------------- #
def start(s, cfg):
    """Upload the generated wrapper and launch the training DETACHED on the server."""
    runid = cfg["run_name"] + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    state_abs = f"{cfg['state_root']}/{runid}"
    state_rel = pg.relpath(state_abs)
    print(f"[start] run id      : {runid}")
    print(f"[start] state dir   : {state_abs}")
    pg.put_file(s, _tmp_write(wrapper_script(cfg, state_abs), "wrapper.sh"),
                f"{state_rel}/wrapper.sh")
    # nohup+setsid -> new session that outlives this kernel / a local disconnect
    out, rc = pg.sh(s, f"cd {q(state_abs)} && : > run.log && "
                       f"nohup setsid bash wrapper.sh >/dev/null 2>&1 & "
                       f"echo $! > wrapper.pid; echo started pid=$(cat wrapper.pid)")
    print(f"[start] launch: {out.strip()} (rc={rc})")
    info = {"runid": runid, "state_abs": state_abs, "state_rel": state_rel,
            "zip_name": cfg["zip_name"], "started": datetime.datetime.now().isoformat()}
    save_state(cfg, info)
    print("[start] training launched DETACHED; it will keep running if you disconnect.")
    return info


def monitor(s, cfg, info):
    """Tail the server log (over /files) until the run reports DONE or FAILED."""
    state_rel = info["state_rel"]
    print(f"[monitor] watching {info['runid']} (poll {cfg['poll']}s; Ctrl-C is safe, "
          f"re-attach with `monitor`)")
    log_off = 0
    while True:
        chunk = fetch(s, f"{state_rel}/run.log", log_off)   # only the new bytes since last poll
        if chunk:
            sys.stdout.write(chunk.decode("utf-8", "replace")); sys.stdout.flush()
            log_off += len(chunk)
        status = read_text(s, f"{state_rel}/status")
        if status == "DONE":
            print(f"\n[monitor] status = DONE"); return True
        if status.startswith("FAILED"):
            print(f"\n[monitor] status = {status}  (see log above)"); return False
        time.sleep(cfg["poll"])


def watch(s, cfg, info):
    """Compact periodic status: refresh just the latest progress line + loss in place.

    Unlike `monitor` (which streams the whole noisy log), this fetches only the
    tail of run.log each cycle and rewrites a single terminal line.
    """
    state_rel = info["state_rel"]
    print(f"[watch] {info['runid']}  (refresh {cfg['poll']}s; Ctrl-C to stop)")
    while True:
        status = read_text(s, f"{state_rel}/status")
        total = pg.remote_size(s, f"{state_rel}/run.log")
        tail = fetch(s, f"{state_rel}/run.log", max(0, total - 4096)).decode("utf-8", "replace")
        prog = re.findall(r"\d+/\d+ \[[^\]]*\]", tail)          # tqdm: "48721/100000 [..., 3.57it/s]"
        loss = re.findall(r"'loss': [0-9.eE+-]+", tail)
        line = prog[-1] if prog else "(no progress line yet)"
        extra = ("  " + loss[-1]) if loss else ""
        sys.stdout.write(f"\r[{status}] {line}{extra}      "); sys.stdout.flush()
        if status == "DONE" or status.startswith("FAILED"):
            print(); return status == "DONE"
        time.sleep(cfg["poll"])


def download(s, cfg, info):
    """Download + verify (size & sha256) the finished run's zip to the local dir."""
    state_rel = info["state_rel"]
    status = read_text(s, f"{state_rel}/status")
    if status != "DONE":
        raise SystemExit(f"run status is '{status}', not DONE; cannot download yet.")
    sha = read_text(s, f"{state_rel}/zip.sha256") or None
    zip_path = read_text(s, f"{state_rel}/zip.path")
    size = read_text(s, f"{state_rel}/zip.size")
    local = pathlib.Path(cfg["local_dir"]) / info["zip_name"]
    print(f"[download] remote : {zip_path}")
    print(f"[download] size   : {int(size):,} bytes" if size.isdigit() else "")
    print(f"[download] sha256 : {sha}")
    print(f"[download] local  : {local}")
    pg.download_resumable(s, zip_path, local, expected_sha256=sha)
    print(f"[download] OK -- verified artifact at {local}")
    return local


def stop(s, cfg, info):
    """Terminate the detached run by killing its whole process group."""
    # the wrapper runs in its own session (setsid), so PID == PGID; kill the whole group
    out, rc = pg.sh(s, f"P=$(cat {q(info['state_abs'])}/wrapper.pid 2>/dev/null); "
                       f"if [ -n \"$P\" ]; then kill -TERM -\"$P\" 2>/dev/null && echo killed group $P "
                       f"|| echo 'no live process (already finished?)'; "
                       f"else echo 'no wrapper.pid'; fi")
    print(f"[stop] {out.strip()}")


# --------------------------------------------------------------------------- #
def default_config():
    """Assemble the config dict for a real fine-tuning run from the CONFIG block."""
    return dict(conda_activate=CONDA_ACTIVATE, train_cwd=TRAIN_CWD, train_cmd=TRAIN_CMD,
                output_dir=OUTPUT_DIR, cleanup_cmd=CLEANUP_CMD, keep_checkpoint=KEEP_CHECKPOINT,
                zip_parent=ZIP_PARENT, zip_name=ZIP_NAME, zip_cmd=ZIP_CMD, local_dir=LOCAL_DIR,
                state_root=STATE_ROOT, poll=POLL_SECONDS,
                run_name=os.path.basename(OUTPUT_DIR))


def selftest_config():
    """Tiny end-to-end rehearsal: no real training, ~12 MB dummy artifact, throwaway paths."""
    base = f"{STATE_ROOT}/_selftest"
    out = f"{base}/dummy_output"
    cfg = default_config()
    cfg.update(
        conda_activate="true",
        train_cwd="/data",
        # fake "training": a 12 MB blob + three checkpoints (90000/95000/100000)
        train_cmd=(f"mkdir -p {q(out)} && "
                   f"head -c 12000000 /dev/urandom > {q(out)}/weights.bin && "
                   f"for i in 90000 95000 100000; do mkdir -p {q(out)}/checkpoint-$i && "
                   f"echo step$i > {q(out)}/checkpoint-$i/info.txt; done && echo trained"),
        output_dir=out,
        keep_checkpoint="checkpoint-100000",
        # keep only the final checkpoint, same logic as the real run
        cleanup_cmd=("find . -maxdepth 1 -type d -name 'checkpoint-*' "
                     "! -name checkpoint-100000 -exec rm -rf {} +"),
        zip_parent=base,
        zip_name="dummy_output.zip",
        zip_cmd="rm -f dummy_output.zip && zip -r -y dummy_output.zip dummy_output/",
        local_dir=os.path.join(os.environ.get("TEMP", "/tmp"), "pegasus_selftest"),
        run_name="_selftest",
        poll=3,
    )
    return cfg


_TMP = pathlib.Path(__file__).parent / "tmp" / "pegasus_wrappers"


def _tmp_write(content, name):
    """Write text to a local temp file with LF line endings (so bash is happy)."""
    _TMP.mkdir(parents=True, exist_ok=True)
    p = _TMP / name
    p.write_text(content, encoding="utf-8", newline="\n")     # LF endings for bash
    return str(p)


def main():
    """CLI entry point: connect to the server, then dispatch the chosen sub-command."""
    ap = argparse.ArgumentParser(description=__doc__,            # the module docstring above
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="run",
                    choices=["run", "start", "monitor", "watch", "download", "status", "stop", "selftest"],
                    help="pipeline action to run (default: run)")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification")
    args = ap.parse_args()

    s = pg.connect(args.insecure)

    if args.cmd == "selftest":
        cfg = selftest_config()
        print("=== SELF-TEST: rehearsing the full pipeline with a dummy job ===")
        info = start(s, cfg)
        ok = monitor(s, cfg, info)
        if ok:
            download(s, cfg, info)
        print("[selftest] cleaning up server scratch dir")
        pg.rm(s, info["state_abs"]); pg.rm(s, f"{cfg['state_root']}/_selftest")
        print("=== SELF-TEST DONE ===" if ok else "=== SELF-TEST FAILED ===")
        sys.exit(0 if ok else 1)

    cfg = default_config()
    if args.cmd == "run":                       # full pipeline
        info = start(s, cfg)
        ok = monitor(s, cfg, info)
        if ok:
            download(s, cfg, info)
        sys.exit(0 if ok else 1)
    elif args.cmd == "start":                   # launch only
        start(s, cfg)
    elif args.cmd == "monitor":                 # re-attach after a disconnect
        info = load_state(cfg)
        ok = monitor(s, cfg, info)
        if ok:
            download(s, cfg, info)
        sys.exit(0 if ok else 1)
    elif args.cmd == "watch":                   # compact periodic status (one refreshing line)
        watch(s, cfg, load_state(cfg))
    elif args.cmd == "download":                # (re)download + verify only
        download(s, cfg, load_state(cfg))
    elif args.cmd == "status":                  # quick status check
        info = load_state(cfg)
        print("run id:", info["runid"])
        print("status:", read_text(s, f"{info['state_rel']}/status") or "(none)")
    elif args.cmd == "stop":                    # abort the detached run
        stop(s, cfg, load_state(cfg))


if __name__ == "__main__":
    main()
