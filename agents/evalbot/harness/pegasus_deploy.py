#!/usr/bin/env python3
"""One-time (and re-runnable) setup: deploy this evalbot harness to Pegasus and verify the
remote environment can actually run it. Safe to re-run any time the harness changes locally —
it re-uploads and re-checks rather than assuming last time's state still holds.

See PEGASUS.md in this directory for the full narrative (why each of these exists, what broke
without it). This script is the "make it true" half; that file is the "why" half.

Usage:
    python agents/evalbot/harness/pegasus_deploy.py            # deploy + verify
    python agents/evalbot/harness/pegasus_deploy.py --no-verify  # deploy only, skip the isaacsim
                                                                  # import check (slower step)
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

import pegasus  # noqa: E402

# Pegasus's own layout is flat and pre-restructure: IsaacLab-GR00T/{IsaacLab, Isaac-GR00T_n1d7,
# artifacts, ...} directly, with NO engines/{sim,vla}/ level in between (unlike this repo). Every
# path below is Pegasus-side, not a mirror of the local layout.
PEGASUS_ROOT = "/data/VLA/tingying/IsaacLab-GR00T"
WORKSPACE_MARKER = f"{PEGASUS_ROOT}/workspace.yaml"
EVALBOT_LOCAL = str(Path(__file__).resolve().parent.parent)   # agents/evalbot
EVALBOT_REMOTE = f"{PEGASUS_ROOT}/agents/evalbot"
ENV_ISAACLAB = "/data/VLA/tingying/envs/env_isaaclab"
GR00T_VENV_PY = f"{PEGASUS_ROOT}/Isaac-GR00T_n1d7/.venv/bin/python"
RLDX_VENV_PY = "/data/VLA/tingying/RLDX-1/.venv/bin/python"    # sibling of PEGASUS_ROOT, not
                                                                 # under it — see rldx1_openarm_o6
                                                                 # .json's _notes for why that
                                                                 # matters for path resolution.

# pyarrow is pandas's parquet ENGINE, not pandas itself — `import pandas` succeeds without it,
# so a check that stopped at "does pandas import" missed it entirely. EpisodeDataSaver's
# per-episode df.to_parquet() call only runs when save_video=true, which the very first smoke
# test happened to have off — cost a ~2-day unattended run before this was caught. Checked as
# a real to_parquet() call below, not just an import, so this class of gap can't repeat quietly.
REQUIRED_PY_PACKAGES = ["pandas", "cv2", "matplotlib", "msgpack", "msgpack_numpy", "zmq", "pyarrow"]

# The "rldx" policy branch additionally needs env_isaaclab to import RLDX-1's own client
# library (rldx.policy.server_client) with RLDX-1 on PYTHONPATH — a completely separate
# dependency chain from the packages above, none of which this harness had ever exercised
# before a real Phase-1 comparison run burned 12 fast-failing attempts on it (fast because the
# gr00t_infer_agent.py hardening from bug #4/#5 below now exits cleanly instead of hanging).
# Found five gaps, one at a time by import traceback, then resolved not just by installing
# each missing package but by matching the EXACT versions RLDX-1's own working .venv uses for
# diffusers/accelerate/huggingface_hub — installing "latest diffusers" pulled a huggingface_hub
# that conflicted with transformers' own pin, which pip's resolver does not catch when the
# already-installed package (transformers) isn't part of that install command.
RLDX_REPO = "/data/VLA/tingying/RLDX-1"
RLDX_VENV_VERSIONED = {"diffusers": "0.35.1", "accelerate": "1.13.0", "huggingface-hub": "0.36.2"}
RLDX_EXTRA_PACKAGES = ["tyro", "av", "albumentations", "dm-tree"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the isaacsim import check (the slow, ~30s step)")
    args = ap.parse_args()

    s = pegasus.connect()

    print(f"[1/5] workspace.yaml marker at {WORKSPACE_MARKER}")
    pegasus.sh(s, f"[ -f {WORKSPACE_MARKER} ] || "
                  f"echo 'workspace: pegasus (auto-created by pegasus_deploy.py)' > {WORKSPACE_MARKER}",
              timeout=30)

    print(f"[2/5] uploading {EVALBOT_LOCAL} -> {EVALBOT_REMOTE}")
    pegasus.put(s, EVALBOT_LOCAL, EVALBOT_REMOTE)

    print(f"[3/5] checking {len(REQUIRED_PY_PACKAGES)} required packages in env_isaaclab")
    # Two real checks, not one: importing a package succeeding does not mean the specific
    # feature this harness needs from it works (pandas imports fine with zero parquet engines
    # installed — see the 2026-08-19 incident this class of check exists to prevent). And:
    # piping a check through `| tail` and then reading `$?` reads tail's exit code, not the
    # check's — that bug made an earlier version of this script always report "all present"
    # regardless of the real result. No pipe here for exactly that reason.
    check = ("; ".join(f"import {p}" for p in REQUIRED_PY_PACKAGES) +
            "; import pandas as pd; pd.DataFrame({'x':[1]}).to_parquet('/tmp/_evalbot_check.parquet')")
    out, rc = pegasus.sh(s, f"{ENV_ISAACLAB}/bin/python -c \"{check}\"", timeout=60)
    if rc != 0:
        print(f"  MISSING or broken (rc={rc}):\n  {out.strip()[-300:]}")
        print("  installing:")
        install_names = ["pandas", "opencv-python-headless", "matplotlib",
                         "msgpack", "msgpack-numpy", "pyzmq", "pyarrow"]
        out2, rc2 = pegasus.sh(s, f"{ENV_ISAACLAB}/bin/python -m pip install --quiet "
                                  f"{' '.join(install_names)} 2>&1 | tail -20", timeout=180)
        print(out2 or "  (installed, no output)")
        out3, rc3 = pegasus.sh(s, f"{ENV_ISAACLAB}/bin/python -c \"{check}\"", timeout=60)
        print("  re-check:", "OK" if rc3 == 0 else f"STILL FAILING (rc={rc3}): {out3.strip()[-300:]}")
    else:
        print("  all present, and pandas.to_parquet() confirmed working")

    print("[4/5] checking the RLDX-1 client import chain in env_isaaclab (separate from "
          "REQUIRED_PY_PACKAGES above -- this is rldx.policy.server_client's own dependency tree)")
    rldx_check = (f"import sys; sys.path.insert(0, {RLDX_REPO!r}); "
                  f"from rldx.policy.server_client import PolicyClient; "
                  f"print('RLDX_CLIENT_IMPORT_OK')")
    out4, rc4 = pegasus.sh(s, f"{ENV_ISAACLAB}/bin/python -c \"{rldx_check}\"", timeout=60)
    if rc4 != 0:
        print(f"  MISSING or broken (rc={rc4}):\n  {out4.strip()[-300:]}")
        print("  installing (version-pinned to match RLDX-1's own .venv, to avoid a resolver "
              "picking a newer diffusers/accelerate that conflicts with transformers' pin):")
        pinned = [f"{name}=={ver}" for name, ver in RLDX_VENV_VERSIONED.items()]
        out5, rc5 = pegasus.sh(s, f"{ENV_ISAACLAB}/bin/python -m pip install --quiet "
                                  f"{' '.join(RLDX_EXTRA_PACKAGES + pinned)} 2>&1 | tail -20",
                              timeout=180)
        print(out5 or "  (installed, no output)")
        out6, rc6 = pegasus.sh(s, f"{ENV_ISAACLAB}/bin/python -c \"{rldx_check}\"", timeout=60)
        print("  re-check:", "OK" if rc6 == 0 else f"STILL FAILING (rc={rc6}): {out6.strip()[-300:]}")
    else:
        # stdout and stderr are concatenated in that order (see pegasus.sh), not interleaved by
        # real timing, so the transformers deprecation warning on stderr can land after our
        # sentinel print even though it happened first — search for the sentinel explicitly
        # rather than assuming the last line is it.
        sentinel = next((ln for ln in out4.splitlines() if "RLDX_CLIENT_IMPORT_OK" in ln), out4.strip())
        print("  all present:", sentinel)

    if not args.no_verify:
        print("[5/5] verifying isaacsim imports cleanly (needs OMNI_KIT_ACCEPT_EULA=YES, or "
              "this hangs on an interactive prompt and dies on EOF)")
        out3, _ = pegasus.sh(
            s, f"OMNI_KIT_ACCEPT_EULA=YES {ENV_ISAACLAB}/bin/python -c "
               f"\"import isaacsim; print('isaacsim OK')\" 2>&1 | tail -6", timeout=90)
        print(" ", out3.strip().replace("\n", "\n  "))
    else:
        print("[5/5] skipped (--no-verify)")

    print("\nDone. Next: pegasus_launch.py --config configs/pegasus/<your_config>.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
