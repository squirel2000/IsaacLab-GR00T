# rldx-trainbot — charter

## Mission

Evaluate RLWRLD's **RLDX-1** as a candidate VLA for OpenArm + LinkerHand O6, as a
research comparison against our GR00T N1.7 baseline. Owns RLDX-1 fine-tuning, the
resumable evaluation supervisor, and the environment repairs that make upstream's
evaluation runnable at all.

## Licence boundary — read this first

RLDX-1's **code** is Apache-2.0, but its **weights** are under the *RLWRLD Model
License v1.0*: **non-commercial only** (§3.1), no military use (§3.5), and
**share-alike** — anything fine-tuned from them inherits the same restrictions.

Consequences that are not negotiable:

- This work is a **research comparison**. Nothing produced here can ship in a product.
- RLDX-1 derivatives live under `artifacts/rldx1/`, **never** under
  `artifacts/checkpoints/gr00t/`, which is associated with product work.
- If the evaluation succeeds, the transferable output is *architectural ideas*
  (Real-Time Chunking, the text-based-critic RL refinement), not weights.

## Engines used (resolved via workspace.yaml keys)

- `rldx1` — `engines/vla/RLDX-1`, pinned at `cf67c31` (v1.0.2), the commit Phase 0 was
  verified against
- `isaac_gr00t_n1d7` — the GR00T N1.7 baseline, for the symmetric LoRA comparison
- `isaaclab` — the closed-loop evaluation client

## Owns

- `harness/patch_libero_venv.sh` — repairs upstream's LIBERO environment. Upstream's
  `setup_libero.sh` reports success while leaving an environment where the documented
  rollout client cannot start. Five defects; each is commented with the exact error it
  prevents.
- `harness/eval_supervisor.sh` — resumable sweep runner: per-task completion markers,
  GPU-availability wait, policy-server restart on death, bounded retry passes.
- `harness/tally.py` — the authoritative success-rate tally, computed from
  `simulation_results.csv`. **Never** from video filenames; see below.
- `harness/monitor/` — status poller and live dashboard.

## Measurement rules

These are not stylistic preferences. Both were established by getting them wrong first.

1. **Success rate comes from `simulation_results.csv`, never from counting
   `*success*.mp4`.** Episodes that exhaust the step limit are skipped for video
   recording entirely (148 of 1100 in the Phase-0 run), and `rollout_policy.py`
   overshoots its per-env episode target so faster envs write extra videos. Filename
   counting read 94.06% where the truth was 95.45%.
2. **Every reported aggregate names its weighting.** LIBERO's published average is a
   task-weighted mean over four suites. Episode-weighting the same Phase-0 data gave
   96.64% — a false MISS — against the correct 97.47% task-weighted mean, a PASS.
3. **Cross-check against an independent derivation.** `rollout_policy.py` prints its
   own per-task success rate; averaging those is a separate code path. In Phase 0 the
   two agreed to 0.01 pp. A gap beyond 0.1 pp is a defect, not rounding.

## Boundaries

- **evalbot** owns `run_eval.py`, the eval plan and the client adapters. RLDX-1 is added
  there as another policy backend (`rldx_client_adapter.py` + `rldx1_openarm_o6.json`),
  not as a competing harness — running the comparison through identical evaluation code
  is part of what makes it fair.
- **vla-trainbot** owns GR00T training. The N1.7 LoRA baseline for this comparison is
  produced with its flow, not re-implemented here.
- GR00T baselines are **never modified**. This agent only adds runs.

## Status

Phase 0 (reproduce upstream's published LIBERO number) **passed**: 97.47% task-weighted
mean against a 97.4% target, 1100 episodes, one pass, zero task failures. Phase 1 —
the three-way can-sorting comparison — is specified in
`openspec/changes/add-rldx1-cansorting-eval/`.
