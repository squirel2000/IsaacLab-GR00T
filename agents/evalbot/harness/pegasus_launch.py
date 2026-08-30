#!/usr/bin/env python3
"""Launch run_eval.py on Pegasus against a config already deployed there (see pegasus_deploy.py),
detached so it survives this process disconnecting, on a GPU chosen because it's actually idle
rather than hardcoded — this box is shared, and a hardcoded device has cost real runs before
(see agents/tools/common/README.md).

Usage:
    python agents/evalbot/harness/pegasus_launch.py --config configs/pegasus/eval_config_phase1_comparison.yaml
    python agents/evalbot/harness/pegasus_launch.py --config configs/pegasus/eval_config_n17_0614_smoke.yaml --gpu 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _root() -> Path:
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        if (d / "workspace.yaml").is_file():
            return d
    return here.parents[2]


ROOT = _root()
sys.path.insert(0, str(ROOT / "agents" / "tools" / "common"))

import h100  # noqa: E402
import pegasus  # noqa: E402

PEGASUS_ROOT = "/data/VLA/tingying/IsaacLab-GR00T"
EVALBOT = f"{PEGASUS_ROOT}/agents/evalbot/harness"
PY_BIN = f"{PEGASUS_ROOT}/Isaac-GR00T_n1d7/.venv/bin/python"   # the scheduler itself only needs
                                                                 # PyYAML + stdlib; this venv
                                                                 # already has it, no need to
                                                                 # assume a bare system python does


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="path to the eval config, relative to agents/evalbot/harness/ "
                         "(e.g. configs/pegasus/eval_config_phase1_comparison.yaml)")
    ap.add_argument("--gpu", type=int, default=None, help="skip the idle check, force this GPU")
    ap.add_argument("--need-gb", type=float, default=60.0,
                    help="free VRAM required; the idle-GPU check refuses rather than queues "
                         "if nothing qualifies")
    ap.add_argument("--tag", default=None,
                    help="distinguishes the scheduler log when launching more than one config "
                         "concurrently; defaults to the config's own filename stem")
    args = ap.parse_args()

    tag = args.tag or Path(args.config).stem
    log = f"/data/VLA/tingying/pegasus_runs/evalbot_{tag}/scheduler.log"

    s = pegasus.connect()
    if args.gpu is not None:
        gpu = args.gpu
        print(f"[!] --gpu {gpu}: skipping the idle check")
    else:
        info = h100.preflight(s, need_gb=args.need_gb, max_rounds=1)
        gpu = info["gpu"]
        print(f"selected GPU {gpu}  [{h100.summarise(info['gpus'])}]")

    pegasus.sh(s, f"mkdir -p $(dirname {log})", timeout=30)
    cmd = (f"cd {EVALBOT}; CUDA_VISIBLE_DEVICES={gpu} setsid nohup {PY_BIN} run_eval.py "
           f"--config {EVALBOT}/{args.config} "
           f"> {log} 2>&1 < /dev/null & disown; echo launched=$!")
    out, rc = pegasus.sh(s, cmd, timeout=60)
    print(out.strip(), "rc=", rc)
    print(f"log: {log}")
    print(f"watch: python agents/evalbot/harness/pegasus_watch.py --log {log}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
