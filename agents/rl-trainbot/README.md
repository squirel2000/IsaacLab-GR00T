# RLinf GR00T-N1.7 RL — Phase 1 (LIBERO validation on Pegasus)

**Status: Phase 1 complete.** The full RLinf PPO loop runs end-to-end, headless, on a
single Pegasus H100, and produces an HTML report. Governed by
`openspec/changes/add-rlinf-gr00t-n17-libero-rl/`.

Goal was to prove the RLinf → GR00T-N1.7 → PPO loop works on our setup before porting to
IsaacLab (Phase 2). It does. On `libero_spatial` the success rate is stable/slightly-positive
(NVIDIA's base checkpoint is already near-ceiling there) — the real improvement headroom is
Phase 2 (IsaacLab + our fine-tuned checkpoint + our task).

## Why this shape (the hard-won facts)
Pegasus host `pa-jp-v1` is a **Jupyter-in-container** box (user `gallop`, 2× H100 80 GB):
- **No Docker** → install RLinf via `uv` (matches the N1.7 pipeline profile).
- **cgroup `pids.max=2048`** (not `ulimit -u`, which is 1M). Ray's default fan-out (one
  worker per core × 64) + torch.compile's 64-wide pool + the Ray dashboard blow this cap
  → `pthread_create: Resource temporarily unavailable`. We cap all of them.
- **NVIDIA EGL present, no Vulkan** → LIBERO (MuJoCo/EGL) renders headless; ManiSkill would not.
- **`nvidia/Cosmos-Reason2-2B` is gated** on HF → reuse the local copy at
  `IsaacLab-GR00T/artifacts/vlm_lora/Cosmos-Reason2-2B-base`.
- Un-reapable zombies: PID 1 is `jupyter-notebook`, which doesn't reap orphaned Ray procs,
  so finished runs leave ~100–300 zombie pids. Keep `runtime.cpu_cores` low for headroom.

## Tooling (`agents/rl-trainbot/harness/`)
| File | Role |
|---|---|
| `config/phase1.yaml` | all knobs (suite, GPU, remote paths, checkpoints, thread caps, smoke/full budgets, `overrides`) |
| `patch_rlinf_clone.py` | **idempotently applies the 6 RLinf-clone patches** (run on Pegasus against a fresh clone) |
| `remote.py` | thin wrapper over `agents/tools/common/pegasus.py` (exec / detached launch / tail / fetch) |
| `launch.py` | pick idle GPU (prefer GPU1, abort if none) → build Hydra overrides → `setsid nohup` detached launch |
| `poll.py` | local live progress view (tails remote log) |
| `extract_metrics.py` | run in-venv on Pegasus: dump TB scalars (under `<logdir>/tensorboard/`) → JSON |
| `report.py` | self-contained HTML report (inline SVG); runs extract_metrics remotely, fetches, renders |

The 6 clone patches (see `patch_rlinf_clone.py` for details): (1) `RLINF_RAY_NUM_CPUS` cap on
ray.init, (2) dashboard off, (3) non-fatal `list_actors` in the signal handler, (4) in-process
`ReconfigureDummyEnv` (dodges the spawn-subprocess mujoco `from_xml_string` crash),
(5) `RLINF_LIBERO_INPROCESS=1` selects it, (6) `component_placement` pinned to one GPU.

Prereq: `PEGASUS_PASSWORD` in the environment. Always run local `pegasus.py` calls with
`PYTHONUTF8=1` (hf prints `✓` → cp950 console crash) and prefer `run --file` (PowerShell mangles
inline `2>/dev/null`).

## End-to-end runbook (Pegasus)
```bash
# 1. Install RLinf (uv, embodied/LIBERO); pin the commit for reproducibility
cd /data/VLA/tingying && git clone --depth 1 https://github.com/RLinf/RLinf.git   # pinned: 43b185d
cd RLinf && bash requirements/install.sh embodied --model gr00t_n1d7 --env maniskill_libero
source .venv/bin/activate
uv pip install "huggingface-hub>=0.34.0,<1.0"          # transformers 4.57.3 needs <1.0

# 2. Apply our clone patches (idempotent)
python scripts_rl_patch_rlinf_clone.py /data/VLA/tingying/RLinf

# 3. Checkpoints: task ckpt is public; backbone is gated -> reuse local copy (see phase1.yaml)
uv run hf download nvidia/GR00T-N1.7-LIBERO --include "libero_spatial/*" \
  --local-dir checkpoints/GR00T-N1.7-LIBERO
```
Then drive from local:
```bash
python agents/rl-trainbot/harness/launch.py --mode smoke   # gate: fits one H100 (~27 GB), ~3 epochs
python agents/rl-trainbot/harness/launch.py --mode full    # 120 epochs, eval/10, GPU1, detached
python agents/rl-trainbot/harness/poll.py  --config libero_spatial_ppo_gr00t_n1d7_phase1 --pid <PID>
python agents/rl-trainbot/harness/report.py --config libero_spatial_ppo_gr00t_n1d7_phase1  # -> agents/docs/...html
```

## Result (2026-06-30)
- Smoke: 3 epochs, checkpoint saved, fits one H100 (~27 GB), GPU0 idle throughout.
- Full (`total_num_envs=4`, `critic_warmup_steps=40`, 120 epochs): `eval/success_once`
  early ≈ 0.50 → late ≈ 0.58 (mean 0.60, peak 0.75). Without critic warmup the policy
  **declined** (→ ~0.12) — the randomly-initialized value head needs warmup before policy
  updates. Value calibration is still weak (tiny reward scale), and the base is near-ceiling,
  so this is stable/slightly-positive rather than a dramatic lift. Report:
  `agents/docs/rlinf_gr00t_n17_libero_phase1_report.html`.

## Phase 2 (next)
Port to IsaacLab: RLinf↔IsaacLab env adapter (reset/step/reward, obs↔modality), our
fine-tuned checkpoint + Can-Sorting task — where the policy actually has room to improve.
