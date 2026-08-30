#!/bin/bash
# ---------------------------------------------------------------------------
# Repair the LIBERO eval venv produced by RLDX-1 v1.0.2 (commit cf67c31).
#
# `rldx/eval/sim/LIBERO/setup_libero.sh` reports success but leaves a venv in
# which the documented rollout client (rldx/eval/rollout_policy.py) cannot
# start. Three independent defects, all verified on Pegasus:
#
#  1. MISSING diffusers
#     setup_libero.sh installs rldx with `--no-deps` and never adds diffusers,
#     which rldx/model/modules/action_model/attention.py imports at module
#     scope. rollout_policy.py imports rldx.data.embodiment_tags, and
#     rldx/__init__.py:66 eagerly pulls in the whole model stack.
#     -> ModuleNotFoundError: No module named 'diffusers'
#
#  2. STALE transformers pin
#     setup_libero.sh pins transformers==4.51.3, but
#     rldx/model/modules/backbone/modeling_qwen3_vl.py imports
#     `transformers.masking_utils.create_causal_mask`, added in 4.52.
#     -> ModuleNotFoundError: No module named 'transformers.masking_utils'
#     Aligned here with the main venv (4.57.0) so client and server agree.
#
#  3. UNPINNED mujoco (transitive drift)
#     LIBERO's requirements.txt pins robosuite==1.4.0; robosuite 1.4.0 declares
#     only `mujoco (>=2.3.0)` with no upper bound, so the resolver takes the
#     newest (3.11.0). Newer mujoco renamed the mass matrix: MjData.qM is gone
#     (attrs are now M / qLD / qLDiagInv), while robosuite 1.4.0
#     controllers/base_controller.py:156 calls mj_fullM(..., sim.data.qM).
#     -> AttributeError: 'MjData' object has no attribute 'qM'
#     3.2.7 is the newest version verified to pass the LIBERO reset test.
#
# Verification: rollout_policy.py --help parses, and register_libero_envs() +
# gym.make(...) + env.reset() succeed under EGL (see the Phase-0 change's
# openspec/changes/add-rldx1-cansorting-eval/tasks.md task 2.6 for the
# regression-check procedure this feeds).
#
# Promoted from tmp/final_verify.sh (openspec task 2.2) -- this file used to
# be regenerated inline by that script on every run; it's now the checked-in
# source of truth instead.
# ---------------------------------------------------------------------------
set -euo pipefail
WS=/data/VLA/tingying
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=$WS/pip_cache/uv
LVENV="$WS/RLDX-1/rldx/eval/sim/LIBERO/libero_uv/.venv"
VIRTUAL_ENV="$LVENV" uv pip install --python "$LVENV/bin/python" \
    diffusers \
    "transformers==4.57.0" \
    "mujoco==3.2.7"
echo "libero venv patched"
