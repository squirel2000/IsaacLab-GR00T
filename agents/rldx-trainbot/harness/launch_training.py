#!/usr/bin/env python3
"""Launch an RLDX-1 / GR00T training run on Pegasus, choosing a GPU that is actually free.

GPU selection, batch splitting and checkpoint verification are NOT reimplemented here — they
come from `agents/tools/common/h100.py`, the shared module every model's harness uses on this
box. `agents/tools/common/README.md` explains the rules it encodes and why each one exists.

The short version: the box is shared, so a run asks which card is free and refuses to start when
none is. Hardcoding `CUDA_VISIBLE_DEVICES=0` cost two runs when another user held 70.4 GB on
GPU 0 while GPU 1 sat idle.

Effective batch is held at 64 for every run so the comparison stays valid; only the split
changes, and `h100.plan_batch` computes it:
    2 GPU, cap 32 -> --global-batch-size 64 --gradient-accumulation-steps 1
    1 GPU, cap 16 -> --global-batch-size 16 --gradient-accumulation-steps 4
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _root() -> Path:
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        if (d / "workspace.yaml").is_file():
            return d
    return here.parents[3]


ROOT = _root()
sys.path.insert(0, str(ROOT / "agents" / "tools" / "common"))

import h100  # noqa: E402
import pegasus  # noqa: E402

WS = "/data/VLA/tingying"
DATASET = f"{WS}/datasets/OpenArm_CanSorting_MultiTask_dataset_O6_0403"
CFG_DIR = f"{WS}/pegasus_runs/rldx_cfg"

# Thresholds mirror vla-trainbot's config defaults; a GPU is idle only if both hold.
GPU_CONFIG = {
    "training": {
        "gpu_idle_threshold_util": 10,
        "gpu_idle_threshold_mem_gb": 5.0,
        "gpu_poll_interval_min": 10,
        "gpu_priority": [1, 0],       # prefer GPU1, fall back to GPU0
    }
}

PRESETS: dict[str, dict] = {
    # 6.1b — the matched arm of the comparison
    "rldx1_right_only": {
        "kind": "rldx",
        "modality": f"{CFG_DIR}/openarm_o6_modality_right_only.py",
        "experiment": "openarm_o6_ptimg_lora_rightonly_0403",
        "output": f"{WS}/artifacts/rldx1/lora_right_only",
        "log": f"{WS}/pegasus_runs/chain2/rldx_rightonly.log",
    },
    "rldx1_bimanual": {
        "kind": "rldx",
        "modality": f"{CFG_DIR}/openarm_o6_modality.py",
        "experiment": "openarm_o6_ptimg_lora_0403",
        "output": f"{WS}/artifacts/rldx1/lora",
        "log": f"{WS}/pegasus_runs/rldx_train/train.log",
    },
    # 6.2 — GR00T N1.7 baseline, via its own wrapper so it matches the nine existing baselines
    "n17_right_only": {
        "kind": "gr00t",
        "mode": "right_only",
        "output": f"{WS}/artifacts/gr00t_n17_match/N1_7_rightonly_30k_batch64_lr1e4",
        "log": f"{WS}/pegasus_runs/chain2/n17_rightonly.log",
    },
}


WANDB_PROJECT_DEFAULT = "openarm-o6-cansorting"    # shared across models, so runs land side by side


def _load_wandb_key() -> str | None:
    """Read the user's own W&B key from vla-trainbot's config.yaml — one key, one place.

    That file is gitignored (it holds the LAN password too) and already carries a per-user key
    used for N1.7 runs; RLDX-1 runs now read the SAME key instead of getting a second copy that
    could drift or leak into a different file.
    """
    cfg_path = ROOT / "agents" / "vla-trainbot" / "harness" / "config" / "config.yaml"
    if not cfg_path.is_file():
        return None
    import yaml
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return (data.get("wandb") or {}).get("api_key") or None


WANDB_KEYFILE_REMOTE = f"{WS}/.wandb_key"


def _ensure_wandb_keyfile(s, api_key: str) -> str:
    """Put the key on disk (owner-only) and return its remote path for `wandb_env_from_file`.

    Never pass the raw key into a command string that goes through `pegasus.sh`/`pegasus.run` —
    see `build_cmd`'s docstring for why (a launch that did this left a key visible to `ps` for
    31+ minutes as a zombie process's argv). The upload itself goes over the same Jupyter
    file-content API used for the LAN password elsewhere in this repo, not a shell command.
    """
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".key", delete=False) as f:
        f.write(api_key)
        local_tmp = f.name
    try:
        h100.put_secret_file(s, local_tmp, WANDB_KEYFILE_REMOTE)
    finally:
        Path(local_tmp).unlink(missing_ok=True)
    return WANDB_KEYFILE_REMOTE


def build_cmd(preset: dict, gpu: int, steps: int, per_device: int, accum: int,
             wandb_key_path: str | None, wandb_project: str) -> str:
    """Remote shell command. Effective batch = per_device * 1 GPU * accum.

    ``wandb_key_path`` is a path ON THE REMOTE HOST to a file holding the key (see
    `_ensure_wandb_keyfile`), never the raw key — `h100.wandb_env_from_file` turns that into a
    `$(cat ...)` substitution so the resolved secret never sits in this command's own text.

    Every setup step before the actual launch is joined with `;`, not `&&`, from the last one
    onward. `&&` would make the trailing `&` background the WHOLE chain as one job, and that
    job's own subshell still waits() for its last command before it can exit — so the outer
    `bash -lc` wrapper (and the `pegasus.sh` call blocked on it) would not return until the
    multi-hour training run itself finished. `;` makes the backgrounded piece its own
    self-contained command with nothing left for the subshell to wait on, and `disown` drops it
    from the job table for good measure. (Observed without this fix: the wrapper process didn't
    return within the call's timeout, and outlived the process that spawned it as an unkillable
    zombie whose argv — including a raw key, before this fix — stayed visible to `ps`.)
    """
    common_env = (
        f"cd {WS} && export TMPDIR={WS}/tmp TMP={WS}/tmp TEMP={WS}/tmp "
        f"HF_HOME=/data/.cache/huggingface PATH=$HOME/.local/bin:$PATH "
        f"TOKENIZERS_PARALLELISM=false "
        # the OOM message's own suggestion; cheap insurance on a shared card
        f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        # Without a key, force offline rather than let the trainer hang trying to reach wandb.
        + ("" if wandb_key_path else "WANDB_MODE=disabled ") + "&& "
    )
    wandb_prefix = h100.wandb_env_from_file(wandb_key_path)
    log = preset["log"]
    mkdir = f"mkdir -p $(dirname {log}) && "

    if preset["kind"] == "rldx":
        wandb_flags = f"--use-wandb --wandb-project {wandb_project} " if wandb_key_path else ""
        body = (
            f"cd {WS}/RLDX-1; CUDA_VISIBLE_DEVICES={gpu} {wandb_prefix}setsid nohup "
            f".venv/bin/python rldx/experiment/launch_train.py "
            f"--base-model-path RLWRLD/RLDX-1-PT-IMG --dataset-path {DATASET} "
            f"--embodiment-tag GENERAL_EMBODIMENT "
            f"--modality-config-path {preset['modality']} "
            f"--video-length 1 --n-cog-tokens 64 "
            f"--global-batch-size {per_device} --num-gpus 1 "
            f"--gradient-accumulation-steps {accum} "
            f"--learning-rate 1e-4 --max-steps {steps} --save-steps 5000 "
            f"--experiment-name {preset['experiment']} --output-dir {preset['output']} "
            f"--action-model-use-lora --action-model-lora-rank 16 --action-model-lora-alpha 32 "
            f"--backbone-use-lora --backbone-lora-rank 16 --backbone-lora-alpha 32 "
            f"{wandb_flags}"
            f"> {log} 2>&1 < /dev/null & disown; echo launched=$!"
        )
    else:
        use_wandb = "1" if wandb_key_path else "0"
        body = (
            f"cd {WS}/IsaacLab-GR00T/Isaac-GR00T_n1d7; "
            f"MODE={preset['mode']} NUM_GPUS=1 USE_WANDB={use_wandb} "
            f"WANDB_PROJECT={wandb_project} CUDA_VISIBLE_DEVICES={gpu} {wandb_prefix}"
            f"BASE_MODEL_PATH=nvidia/GR00T-N1.7-3B DATASET_PATH={DATASET} "
            f"OUTPUT_DIR={preset['output']} MAX_STEPS={steps} SAVE_STEPS=5000 "
            f"LEARNING_RATE=1e-4 SAVE_TOTAL_LIMIT=5 "
            f"GLOBAL_BATCH_SIZE={per_device} GRADIENT_ACCUMULATION_STEPS={accum} "
            f"DATALOADER_NUM_WORKERS=0 MASTER_PORT=29560 "
            f"setsid nohup bash examples/Openarm_LinkerHandO6/finetune_openarm_o6.sh "
            f"> {log} 2>&1 < /dev/null & disown; echo launched=$!"
        )
    return common_env + mkdir + body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--effective-batch", type=int, default=64,
                    help="held equal across compared runs; the split adapts to the hardware")
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--per-device-cap", type=int, default=16,
                    help="largest per-device batch the card can hold (1 GPU: 16, 2 GPU: 32)")
    ap.add_argument("--need-gb", type=float, default=60.0,
                    help="free VRAM the run needs; 0 falls back to the idle test")
    ap.add_argument("--max-rounds", type=int, default=1,
                    help="GPU checks before giving up (1 = refuse rather than queue)")
    ap.add_argument("--force-gpu", type=int, default=None,
                    help="skip the check entirely (use with care)")
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=True,
                    help="report to Weights & Biases (default on; needs a key in "
                         "vla-trainbot's config.yaml, else falls back to offline)")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-project", default=WANDB_PROJECT_DEFAULT,
                    help="shared across models so runs land in one place for comparison")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    preset = PRESETS[args.preset]
    global_batch, accum = h100.plan_batch(args.effective_batch, args.num_gpus,
                                          args.per_device_cap)
    wandb_key = _load_wandb_key() if args.wandb else None
    if args.wandb and not wandb_key:
        print("[!] --wandb requested but no key found in vla-trainbot's config.yaml "
              "(wandb.api_key) — running offline instead.")
    s = pegasus.connect()
    # The raw key never enters a command string from here on — only a remote path to it does
    # (see build_cmd's docstring for why: the plain-value approach left a key exposed via `ps`
    # for the lifetime of a launch that hangs, which — before the `;`-vs-`&&` fix below — was
    # every launch).
    wandb_key_path = _ensure_wandb_keyfile(s, wandb_key) if wandb_key else None

    if args.force_gpu is not None:
        gpu = args.force_gpu
        print(f"[!] --force-gpu {gpu}: skipping the free-memory check")
    else:
        try:
            gpu = h100.preflight(s, need_gb=args.need_gb or None, config=GPU_CONFIG,
                                 max_rounds=args.max_rounds)["gpu"]
        except TimeoutError as e:
            print(f"REFUSED: {e}\nNot launching into an OOM.")
            return 2

    print(f"launching {args.preset} on GPU {gpu}: {args.steps} steps, "
          f"global batch {global_batch} x accum {accum} = effective {args.effective_batch}, "
          f"wandb={'on · ' + args.wandb_project if wandb_key_path else 'off'}")
    cmd = build_cmd(preset, gpu, args.steps, global_batch, accum, wandb_key_path, args.wandb_project)
    out, rc = pegasus.sh(s, cmd, timeout=180)
    print(out.strip())
    print(f"log: {preset['log']}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
