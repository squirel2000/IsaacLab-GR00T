# Design

## Context

Phase 0 established that RLDX-1's released LIBERO checkpoint reproduces its published number on Pegasus (97.47% suite mean vs 97.4% target). That work produced a set of hard-won environment facts and tooling that currently exist only as scratch files, plus a body of knowledge about how *not* to measure success rates.

Current state we build on:

- **`Isaac-Can-Sorting-OpenArm-DexHand-v0`** already exists in `engines/sim/IsaacLab` and is driven by `agents/evalbot/harness/run_eval.py`: 100 episodes per run, closed-loop in IsaacSim, `multitask: true` for per-colour language prompt switching, `pov_list: [head]`, `openarm_hand_type: linkerhand_o6`.
- **Six GR00T baseline runs** are already configured in `eval_config.yaml`, including several N1.7 variants at 150k–300k steps. All are **full fine-tunes** on `dataset_0403` (sim, 2000 episodes).
- **Client adapters** already exist for GR00T and starVLA (`agents/evalbot/harness/utils/*_client_adapter.py`), so adding a third backend follows an established pattern.
- **RLDX-1 on Pegasus** is built and verified: repo at `/data/VLA/tingying/RLDX-1` pinned to `cf67c31`, main venv (torch 2.7.0) and an isolated LIBERO sim venv (torch 2.5.1), 13 GB checkpoint cached.

Hard constraints:

- RLDX-1 weights are **non-commercial with share-alike**; derivatives inherit that.
- Pegasus has **no conda**; `uv` only. Its `/tmp` is 1 GB, so `TMPDIR` must be redirected.
- Pegasus is `192.168.52.110` and **cannot reach** the asus-4090 at `192.168.32.185` — different subnets, no route. Cross-machine split serving is not available.
- Pegasus GPUs are **shared with other users**; GPU1 was continuously occupied throughout Phase 0.

## Goals / Non-Goals

**Goals:**

- Make the Phase-0 result reproducible by any team member from a clean checkout.
- Answer whether RLDX-1 beats GR00T N1.7 on our task, with the comparison structured so a loss is attributable.
- Produce a success-rate number that cannot be quietly wrong.
- Determine whether the evaluation is later reproducible on a 24 GB RTX 4090.

**Non-Goals:**

- Any product path for RLDX-1. The licence forbids it; this is a research comparison only.
- Beating a specific threshold. Success is *trustworthy three-way data*, not RLDX-1 winning.
- Memory-module evaluation. Can-sorting is one pick-and-place per episode, so a memory module has nothing to exploit.
- Changing camera count, adding torque, or touching the real dataset — all would break comparability with the existing baselines.

## Decisions

### 1. Success rate is computed from `simulation_results.csv`, never from video filenames

Phase 0 proved filename counting is wrong in two independent ways: episodes that hit the step cap are **skipped for video recording entirely** (148 of 1100 in our run), and `rollout_policy.py` overshoots its per-env episode target so faster envs write extra videos (68 files for a 50-episode target). On the same partial data this read 94.06% where the truth was 95.45% — a 1.4 pp error.

`simulation_results.csv` carries one row per `(env_idx, episode_idx)` with an explicit `int(success)`, and the writer replaces any prior row for the same key. That is the record of truth.

*Alternative considered:* parsing the `success rate:` line the rollout prints. Kept — but as an **independent cross-check**, not the primary source. In Phase 0 the two agreed to 0.01 pp, which is exactly the kind of corroboration we want to keep.

### 2. Aggregate by equal weight per task, and state the weighting explicitly

Phase 0's other trap: LIBERO's published average is a mean over its four suites, each equally weighted, but `libero_10` runs 50 episodes/task against 20 elsewhere. Episode-weighting our own final data produced 96.64% (a false MISS) against the correct 97.47% suite mean (a PASS) — same episodes, different arithmetic.

Can-sorting is uniform (100 episodes per run, one task), so the trap does not bite here. The decision is nonetheless recorded: **any reported aggregate names its weighting.**

### 3. Symmetric LoRA baseline, with the existing full fine-tune as an upper bound

Three runs: RLDX-1 LoRA, N1.7 LoRA, N1.7 fft. The LoRA-vs-LoRA pair is the attributable comparison; the fft run says how much tuning budget is worth on this task.

*Alternative considered:* RLDX-1 full fine-tune for a like-for-like against the existing fft baselines. Rejected — 6.9B with AdamW needs ~110 GB of weights/grads/optimiser state, requiring ZeRO-3 plus CPU offload across both H100s and likely multiple days, on a box where one GPU is frequently taken. It also contradicts the stated intent of confirming the result cheaply first.

### 4. Evaluation runs on Pegasus H100, not the 4090

The 4090 has the better renderer (H100 has no RT cores) and a ready `env_isaaclab`, but 24 GB is genuinely tight: RLDX-1 resident is ~15 GB and IsaacSim is ~6–11 GB by the existing config's own estimate, so 21–26 GB against a 24 GB budget. Pegasus removes the memory question entirely (80 GB per GPU) and keeps training and evaluation on one machine.

Vulkan was verified rather than assumed: a ctypes enumeration on Pegasus returns both H100s as `DISCRETE_GPU`, API 1.4.303, using the system ICDs. An earlier reading of `/usr/share/vulkan/icd.d` suggested no NVIDIA ICD was present; that was wrong — it is registered elsewhere.

*Alternative considered:* policy server on Pegasus, IsaacSim on the 4090. **Not available** — the two hosts are on different subnets with no route.

Cost accepted: no RT cores means slower rendering. Mitigation is to measure render throughput before fixing the episode budget.

### 5. Peak VRAM is measured per component so the 4090 remains a future option

We sample peak `memory.used` for the policy server alone and for IsaacSim alone, rather than only the combined figure, so the 4090 question can be answered arithmetically without re-running anything.

### 6. RLDX-1 is served through an adapter, not a new harness

`run_eval.py`, the task, the episode budget, the filtering and the charting all stay as-is. RLDX-1 differs only in its wire contract: ZeroMQ + msgpack with `{modality}.{key}` observations (`video.*` as `(B,T,H,W,3)` uint8, `state.*` as float32) and 16-step action chunks. A `rldx_client_adapter.py` alongside the existing GR00T and starVLA adapters keeps the comparison running through identical evaluation code — which is itself part of making the comparison fair.

### 7. Upstream environment repairs live in the repo, not in shell history

The four upstream defects and one local collision are captured as an idempotent, commented patch script under `agents/rldx-trainbot/harness/`. Each fix records the failure it prevents, because every one of them presents as a misleading error:

| # | Defect | Symptom |
|---|---|---|
| 1 | `setup_libero.sh` installs `rldx` with `--no-deps`, never adds `diffusers` | `ModuleNotFoundError: diffusers` |
| 2 | pins `transformers==4.51.3`, but their `modeling_qwen3_vl.py` needs `masking_utils` (≥4.52) | `No module named 'transformers.masking_utils'` |
| 3 | mujoco unpinned; robosuite 1.4.0 allows `>=2.3.0`, resolver takes 3.11.0 where `MjData.qM` is gone | `AttributeError: 'MjData' object has no attribute 'qM'` — pin `mujoco==3.2.7` |
| 4 | `eval_libero.sh` probes readiness with `ss -lnt`; Pegasus has no iproute2 or net-tools | `Server died before binding` while the server is up and listening |
| 5 | (ours) stale `~/.libero/config.yaml` from the RLinf work points at a deleted tree | `<task>.bddl does not exist` — fix via `LIBERO_CONFIG_PATH`, do not edit the shared file |

### 8. The supervisor owns interruption handling

Per-task `.done` markers so a restart re-runs only unfinished work; server health-checked before each task and restarted on death; bounded retry passes. Two bugs found in Phase 0 are already fixed and must not regress:

- A bare `wait` also waits on the `setsid`-launched policy server, so the supervisor hangs in `do_wait` after the last task and pins GPU memory indefinitely. Wait on collected task PIDs.
- The `jobs -rp` throttle counts the server job, silently capping parallelism one below the configured value.

### 9. Licence boundary is a path convention

RLDX-1 derivatives go under `artifacts/rldx1/` (research-only), never `artifacts/checkpoints/gr00t/`, which is associated with product work.

## Risks / Trade-offs

- **H100 has no RT cores, so IsaacSim rendering is slower than on the 4090** → measure render throughput and per-episode wall-clock in a short smoke run before committing to 100 episodes × 3 runs; reduce resolution or raise parallel jobs (80 GB allows several) if throughput is the bottleneck.
- **Pegasus GPUs are shared; GPU1 was occupied for all of Phase 0** → the supervisor already waits for a GPU with sufficient free VRAM and picks whichever qualifies; long runs must be resumable, which they are.
- **A first-time uv-based IsaacSim install on a box with no conda and a 1 GB `/tmp` may fail in unfamiliar ways** → treat it as a spike with its own verification (headless render of one frame from the target task) before any eval work depends on it. `TMPDIR` and `UV_CACHE_DIR` must point at `/data`.
- **LoRA vs full fine-tune asymmetry against the existing fft baselines** → mitigated by adding the N1.7 LoRA run; the fft number stays in the table as an upper bound rather than as the comparison.
- **RLDX-1's action space and normalisation may not map cleanly onto the 20-dim OpenArm state/action layout** → the adapter is validated against a recorded episode (replay observations, confirm action shapes and ranges) before closed-loop evaluation.
- **LeRobot v2.0 → v2.1 conversion could alter data semantics** → verify episode count, frame count, and per-field statistics match the source before training on the converted set.
- **Non-commercial weights could leak into a product path** → path convention plus an explicit note in the agent charter.

## Migration Plan

1. Add the submodule and workspace keys — inert, nothing depends on them yet.
2. Promote the harness; re-run the Phase-0 LIBERO tally from the repo copy and confirm it still reports 97.47%. This validates the promotion against a known answer.
3. Spike the IsaacSim install on Pegasus; gate on a single headless rendered frame from the target task.
4. Convert the dataset; verify counts and statistics.
5. Build and validate the adapter offline against recorded episodes.
6. Train the two LoRAs.
7. Smoke-eval 5 episodes per run to measure throughput and peak VRAM, then run the full three-way comparison.
8. Report, including the 4090 feasibility verdict.

Rollback: every step is additive. The submodule and the new agent directory can be removed without touching existing engines or the GR00T baselines, which are never modified.

## Open Questions

- Which RLDX-1 base checkpoint to fine-tune from: `RLDX-1-PT` (video, 4 frames) or `RLDX-1-PT-IMG` (single image)? Our dataset is single-camera, and `PT-IMG` may be the better match; `PT` is the slot upstream reserves for downstream fine-tuning. Resolve by trying `PT` first with `--video-length 4`, since that is what upstream documents, and note `PT-IMG` as a fallback if the temporal stack does not help.
- LoRA rank and which surfaces to adapt (action model, backbone, or both). Start from upstream's documented defaults (rank 16, alpha 32) on both surfaces; treat tuning as out of scope for a first comparison.
- Whether a converted v2.1 dataset should be committed or generated on demand. Leaning generate-on-demand with the conversion script tracked, to avoid duplicating 882k frames in the repo.
