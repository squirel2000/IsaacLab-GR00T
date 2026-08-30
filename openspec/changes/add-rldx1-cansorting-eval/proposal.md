# Evaluate RLDX-1 against GR00T N1.7 on OpenArm can-sorting

## Why

Phase 0 is done: RLWRLD's released `RLDX-1-FT-LIBERO` checkpoint reproduces its published LIBERO figure on our hardware at **97.47% suite mean against a 97.4% target**, cross-checked to 0.01 pp against the per-task rates upstream's own rollout prints. That clears the gate we set — comparisons against this model are now meaningful rather than resting on vendor-reported numbers.

Two things follow. First, the substantive question is still unanswered: **is RLDX-1 actually better than our GR00T N1.7 baseline on our own robot, our own data, and our own task?** RLDX-1 claims large margins over GR00T N1.6 on robustness suites (LIBERO-Plus 86.7 vs 72.6), and its paper's real-robot experiments include OpenArm — but it has never been compared to N1.7, and never on can-sorting.

Second, the Phase-0 result is currently **not reproducible by anyone else on the team**. The harness that produced it lives in an untracked scratch directory, and RLDX-1's documented LIBERO evaluation does not run out of the box — four upstream defects plus one collision with our own RLinf install had to be fixed before a single episode executed. If those fixes are not captured in-repo, the 97.47% is a number in a chat log.

## What Changes

- **Track RLDX-1 as an engine.** Add `engines/vla/RLDX-1` as a git submodule pinned at the verified commit, with a `rldx1` key in `workspace.yaml` and `WS_RLDX1` in `paths.env`.
- **Promote the Phase-0 harness into the repo** as `agents/rldx-trainbot/`: the resumable eval supervisor, the CSV-based tally, the status poller, the live dashboard, and — critically — the environment repair script that makes the upstream LIBERO eval executable at all.
- **Install IsaacSim + IsaacLab on Pegasus via `uv`.** Pegasus has no conda, so the existing `env_isaaclab` flow does not apply. Vulkan is confirmed working there (both H100s enumerate as `DISCRETE_GPU`, API 1.4.303), which is what the Omniverse renderer needs.
- **Make RLDX-1 serve the existing eval task.** Add a policy client adapter and an evalbot policy config so RLDX-1 can drive `Isaac-Can-Sorting-OpenArm-DexHand-v0` through the same `run_eval.py` path as the GR00T runs. RLDX-1 speaks ZeroMQ + msgpack with its own observation schema, so this is a translation layer, not a new harness.
- **Train two LoRA fine-tunes** on `OpenArm_O6_CanSorting_MultiTask_Sim_Dataset_0403` (2000 episodes, 882k frames): one RLDX-1, and one **GR00T N1.7 LoRA as a symmetric baseline**. The existing N1.7 baselines are all full fine-tunes, so without a LoRA-vs-LoRA pair a loss could not be attributed to the model rather than the tuning method.
- **Run a three-way closed-loop comparison**, 100 episodes each on the same task with the same metrics: RLDX-1 LoRA, N1.7 LoRA, and the existing N1.7 full fine-tune as an upper-bound reference.
- **Measure peak VRAM per component** (policy server vs IsaacSim) so we can say whether this evaluation is reproducible on the 24 GB RTX 4090 later.
- **Convert the dataset to LeRobot v2.1**, which RLDX-1 requires; ours are v2.0.

Explicitly out of scope, with reasons:

- **The memory module A/B (previously "Phase 2") is removed.** Can-sorting performs a single pick-and-place per episode, so there is no partial observability for a memory module to exploit. Testing it needs a task whose answer is not in the current frame; that is separate work.
- **Torque logging** — deferred until small/deformable objects make contact force matter.
- **Dual-camera re-export** — the robot has head and wrist cameras but the datasets carry only head, and the existing N1.7 baselines are head-only. Changing camera count would break comparability with those baselines.
- **Phase 3 deformable objects** — the real data does not exist yet.

## Capabilities

### New Capabilities

- `rldx1-integration`: how RLDX-1 lives in this monorepo — engine submodule, the two-venv environment and its required repairs, the resumable evaluation supervisor, and the licence boundary that keeps non-commercial weights out of the product checkpoint path.
- `rldx1-cansorting-eval`: the Phase-1 comparison protocol — LoRA fine-tuning on the sim can-sorting dataset, serving RLDX-1 into the existing IsaacLab task, and the three-way closed-loop measurement with a trustworthy success-rate definition.

### Modified Capabilities

None. `openspec/specs/` is empty on this branch, so both capabilities are new.

## Impact

**Repo structure**
- `.gitmodules`, `engines/vla/RLDX-1` — new submodule
- `workspace.yaml`, `paths.env` — new `rldx1` / `WS_RLDX1` entry
- `agents/rldx-trainbot/**` — new agent (harness, config, charter)
- `agents/evalbot/harness/configs/rldx1_openarm_o6.json` — new policy config
- `agents/evalbot/harness/configs/eval_config.yaml` — new runs entries
- `agents/evalbot/harness/utils/rldx_client_adapter.py` — new adapter alongside the existing GR00T and starVLA adapters
- `datasets/` — a v2.1 conversion of dataset 0403

**Systems**
- Pegasus `pa-jp-v1`: new uv-based IsaacSim/IsaacLab environment (~30 GB); the RLDX-1 clone and its two venvs already exist from Phase 0
- Both training runs and evaluation run on Pegasus H100; the RTX 4090 is a later reproduction target, not a dependency

**Licence constraint**
RLDX-1 weights are RLWRLD Model License v1.0 — non-commercial with share-alike, inherited by anything fine-tuned from them. All RLDX-1 derivatives must stay out of `artifacts/checkpoints/gr00t/` and be stored under a clearly labelled research-only path.

**Known risk**
H100 has no RT cores, so the Omniverse renderer will be slower per frame than on the 4090. Render throughput gets measured before the episode budget is fixed.
