## Why

We have a fine-tuned GR00T-N1.7 policy and IsaacLab-collected datasets, and want to *improve the policy with reinforcement learning* using [RLinf](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/gr00t.html). RLinf supports GR00T-N1.5/N1.6/N1.7 with PPO, but its GR00T+RL recipe targets **LIBERO**; its **IsaacLab integration is still upstream "in progress."** The dominant project risk is therefore the RLinf↔IsaacLab integration, not the dev methodology. This change de-risks that by first proving the *entire* RL loop works and measurably lifts a GR00T-N1.7 policy on LIBERO (Phase 1), bootstrapping from NVIDIA's official `GR00T-N1.7-LIBERO` checkpoint, before investing in the IsaacLab port (Phase 2, out of scope here).

## What Changes

- Add an OpenSpec-managed RL capability for GR00T-N1.7 (this is Phase 1 only).
- Add `scripts/rl/` containing:
  - A Pegasus **uv-based RLinf install + setup runbook** (no Docker — Pegasus is a Jupyter-in-container host without docker; NVIDIA EGL is present for headless rendering).
  - An **RLinf config override** for one LIBERO suite, pinned to a **single GPU**, PPO defaults, FSDP+HF backend.
  - A **launcher** that selects an unused GPU (default **GPU1**), exports `MUJOCO_GL=egl` + the correct render device, and starts training **detached (nohup)** on Pegasus via `scripts/common/pegasus.py`.
  - A **local progress poller** that tails the remote training log and surfaces the success-rate / reward trend on the local terminal.
  - An **HTML analysis report generator** matching the existing `docs/*.html` report style.
- No changes to existing `scripts/pipeline/` stages — Phase 1 stays standalone; pipeline integration is deferred until it is proven.

## Capabilities

### New Capabilities
- `rlinf-libero-rl-training`: Install, configure, and run RLinf PPO on GR00T-N1.7 in a LIBERO suite, headless on a single Pegasus H100, bootstrapped from the NVIDIA `GR00T-N1.7-LIBERO` checkpoint, producing RL checkpoints + metrics with a measurable success-rate lift over the base checkpoint.
- `rl-run-orchestration`: Select an unused GPU (default GPU1), launch training detached on Pegasus, and poll the remote log to display improvement/progress locally.
- `rl-analysis-report`: Produce a complete standalone HTML report summarizing config, before/after success rate, training curves, GPU/resource usage, and a pass/fail verdict against the success criterion.

### Modified Capabilities
<!-- None. No existing OpenSpec specs change (this is the first OpenSpec change in the repo). -->

## Impact

- **New code**: `scripts/rl/` (launcher, poller, report generator, RLinf config override, runbook). Python, reusing `scripts/common/pegasus.py` for remote exec/transfer.
- **Remote**: an RLinf clone + uv env on Pegasus under `/data/...`; downloads the NVIDIA `GR00T-N1.7-LIBERO` checkpoint.
- **Hardware**: exactly one H100 (80 GB) on Pegasus per run (cooperative GPU use — never both).
- **Output**: an HTML report under `docs/`; metrics/logs synced to local `artifacts/`.
- **Dependencies**: RLinf (pinned commit), LIBERO/robosuite/MuJoCo (via RLinf embodied install), NVIDIA EGL (already present on Pegasus).
- **No breaking changes**; no changes to the existing fine-tune/eval/deploy pipeline.
