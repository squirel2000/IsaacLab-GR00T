## Context

The repo already has a mature collect → fine-tune → eval → deploy pipeline for GR00T in IsaacLab (`scripts/pipeline/`, `scripts/eval/`, remote training on the Pegasus H100 box driven by `scripts/common/pegasus.py`). We now want RL on top of a fine-tuned GR00T-N1.7 policy using RLinf, ultimately in IsaacLab.

Verified constraints (capability probe on Pegasus host `pa-jp-v1`, 2026-06-30):
- **No Docker** (Jupyter-in-container; `docker` absent) → RLinf must be installed via `uv`.
- **2× H100 PCIe, 80 GB**, driver 570.195. Per the cooperative-use rule, **only one GPU** may be used (default GPU1).
- **NVIDIA EGL present** (`libEGL_nvidia.so.0`) → headless LIBERO rendering works with `MUJOCO_GL=egl`.
- **No Vulkan** (`vulkaninfo` absent) → ManiSkill would be painful; **LIBERO avoids this**.
- `uv 0.11.24`, no conda; `/data` has ~840 GB free.

RLinf facts (docs + RLinf-VLA paper): supports GR00T-N1.5/N1.6/N1.7 with PPO; GR00T+RL example targets **LIBERO**; **IsaacLab integration is upstream "in progress."** N1.7 currently bootstraps from NVIDIA's official `GR00T-N1.7-LIBERO` checkpoint rather than an RLinf-trained SFT checkpoint.

## Goals / Non-Goals

**Goals:**
- Prove the full RLinf PPO loop runs headless on a single Pegasus H100 and **measurably lifts** a GR00T-N1.7 policy on **one** LIBERO suite.
- Cooperative GPU use (one GPU, default GPU1), detached runs, local progress visibility, and a complete HTML report.
- Leave a reproducible runbook + thin tooling under `scripts/rl/`.

**Non-Goals:**
- IsaacLab RL (Phase 2) — no env adapter, reward, or modality mapping here.
- Improving the *Can-Sorting* deploy policy — Phase 1 trains a LIBERO policy from NVIDIA weights, not our fine-tuned checkpoint.
- Reproducing RLinf's full published multi-suite numbers.
- Integrating RL into `pipeline_runner` — deferred until proven.

## Decisions

- **uv install, not Docker.** Forced by the probe. Alternative (Docker image `rlinf/rlinf:...`) rejected: not available in the Jupyter container.
- **LIBERO, not ManiSkill, for Phase 1.** ManiSkill needs Vulkan (absent); LIBERO renders via EGL (present). Also matches RLinf's GR00T recipe.
- **Single GPU, default GPU1, auto-pick a free one.** The launcher queries `nvidia-smi` for an idle GPU, prefers GPU1, pins `CUDA_VISIBLE_DEVICES` + EGL render device, and aborts if none are free — so other users are never disrupted. This collocates env+rollout+actor+critic on one 80 GB card, which is the tightest config and the main feasibility risk.
- **FSDP + HuggingFace backend** (RLinf prototyping mode), not Megatron+SGLang — appropriate at single-GPU/LIBERO scale and simpler to stand up.
- **Detached nohup + remote log polling**, reusing `scripts/common/pegasus.py` (`run`/`sh`/`get`). Training is started with `nohup … &` and a recorded PID + log path; the local poller periodically `get`s/tails the remote log instead of holding a long WebSocket. Rationale: the kernel WebSocket can drop on long runs; detaching makes the run independent of the control channel.
- **Self-contained HTML report** under `docs/`, matching existing reports (inline CSS/JS, embedded chart data as JSON, no external CDN) so it opens offline.
- **Smoke test gates the long run.** A minimal-budget run proves EGL rendering, one PPO update, and single-GPU memory fit before committing compute.

## Risks / Trade-offs

- **GR00T-N1.7 + PPO may not fit one 80 GB GPU** → Mitigation: smoke-test first; documented fallbacks (smaller batch/rollout, fewer parallel envs, gradient checkpointing, FSDP CPU offload). If still infeasible, escalate to the user about a 2-GPU exception.
- **EGL multi-GPU device-index mismatch** (MuJoCo EGL indexes globally; `CUDA_VISIBLE_DEVICES` is local) → Mitigation: set `GPUS`/render device id to the global index of the chosen GPU.
- **RLinf install flags / checkpoint id / config keys may differ from docs** → Mitigation: verify against the RLinf GR00T quickstart and repo at build time; the runbook records the exact commands actually used. Captured in Open Questions.
- **Log format unknown for the poller** → Mitigation: implement the poller against RLinf's actual metric output (text log and/or TensorBoard event files) discovered during the smoke test; keep parsing tolerant.
- **Pegasus password rotation** (`PEGASUS_PASSWORD`) → already current; runbook notes re-export on rotation.
- **Long wall-clock** → runs are detached; the operator polls and is not blocked.

## Migration Plan

Additive only. New `openspec/` change + `scripts/rl/` dir + one `docs/*.html` report. No existing code paths change; nothing to roll back beyond deleting the new files. RLinf lives on Pegasus (pinned commit noted in the runbook); not vendored into the repo in Phase 1.

## Open Questions

- Exact RLinf embodied install invocation for GR00T (the `--model`/`--env` values; docs show `--model openvla --env maniskill_libero`).
- Exact Hugging Face repo id for the NVIDIA `GR00T-N1.7-LIBERO` checkpoint.
- Exact RLinf config file + keys for: LIBERO suite selection, single-GPU placement, checkpoint path, PPO hyperparameters.
- Where RLinf writes step metrics (stdout log vs TensorBoard vs jsonl) — determines the poller's parser.
- Final choice of LIBERO suite for Phase 1 (proposed: Goal or Spatial — smallest/fastest to show a lift).
