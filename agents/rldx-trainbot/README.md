# rldx-trainbot

Research evaluation of **RLWRLD RLDX-1** as a candidate VLA for OpenArm + LinkerHand O6,
compared against our GR00T N1.7 baseline.

> **Non-commercial weights.** RLDX-1's code is Apache-2.0 but its weights are
> RLWRLD Model License v1.0 — non-commercial, share-alike, inherited by derivatives.
> Nothing here can ship. See [charter.md](charter.md).

## What RLDX-1 is

Qwen3-VL-8B backbone + a Multi-Stream Action Transformer (MSAT), 6.9B parameters. Not
GR00T-derived, but it reuses GR00T N1.7's conventions — `EmbodimentTag`, per-robot MLP
heads, `modality.json`, LeRobot datasets, `max_state_dim=64` — which is why our data
plumbing transfers.

The design point that matters for us: cognition, action and physical sensing are three
separate token streams that meet only at joint self-attention, so the physics stream can
be masked out entirely when there is no tactile or torque signal. **We confirmed the
released LIBERO checkpoint has `use_physics: false`** — the published headline number was
achieved with that stream off, and the paper's own OpenArm experiments ran without it.

## Status

| Phase | What | Result |
|---|---|---|
| 0 | Reproduce upstream's published LIBERO number | **PASS** — 97.47% vs 97.4% target |
| 1 | Three-way can-sorting comparison | specified, not started |
| — | Memory-module A/B | **dropped** — can-sorting is one pick-and-place per episode, so a memory module has nothing to exploit |

Phase 0 detail: 1100 episodes, one pass, zero task failures, zero server restarts,
~1 h on a single H100. Per-suite 97.50 / 99.00 / 99.00 / 94.40 (spatial / object / goal /
long) against published 98.6 / 98.6 / 98.6 / 95.3. Cross-checked to 0.01 pp against the
per-task rates upstream's own rollout prints.

Phase 1 is specified in `openspec/changes/add-rldx1-cansorting-eval/`.

## Two measurement rules

Both were learned by getting them wrong.

1. **Never count `*success*.mp4`.** Episodes that exhaust the step limit are not recorded
   to video (148 of 1100), and the rollout overshoots its per-env episode target. Use
   `simulation_results.csv`.
2. **Never episode-weight a LIBERO-style average.** It is task-weighted. Episode-weighting
   our Phase-0 data gave 96.64% (false MISS) vs the correct 97.47% (PASS).

## Upstream defects this agent works around

`setup_libero.sh` reports success while leaving an unusable environment, because its own
verification step is non-fatal. All five are handled by `harness/patch_libero_venv.sh`:

| # | Cause | Symptom |
|---|---|---|
| 1 | `rldx` installed `--no-deps`; `diffusers` never added | `ModuleNotFoundError: diffusers` |
| 2 | pins `transformers==4.51.3`; their own backbone needs `masking_utils` (≥4.52) | `No module named 'transformers.masking_utils'` |
| 3 | mujoco unpinned; robosuite 1.4.0 allows `>=2.3.0` → resolver takes 3.11.0 | `AttributeError: 'MjData' object has no attribute 'qM'` (pin `3.2.7`) |
| 4 | readiness probed with `ss -lnt`; Pegasus has no iproute2/net-tools | `Server died before binding` while it is listening |
| 5 | *(ours)* stale `~/.libero/config.yaml` from the RLinf work | `<task>.bddl does not exist` |

These corroborate upstream issues #24/#25 — treat RLDX-1's published benchmark table as
vendor-reported.

## Layout

```
harness/
  patch_libero_venv.sh   # repair upstream's LIBERO env (idempotent)
  eval_supervisor.sh     # resumable sweep: markers, GPU wait, server restart, retries
  tally.py               # authoritative success rate from simulation_results.csv
  monitor/               # status poller + live dashboard
var/logs/                # run logs (gitignored)
```

Results and checkpoints go to `artifacts/rldx1/`.
