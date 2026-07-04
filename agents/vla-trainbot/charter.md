# vla-trainbot — charter

## Mission

Turn datasets into deployable VLA checkpoints: wait for a free H100 → fine-tune
(GR00T N1.5/N1.6/N1.7 or starVLA) → remote closed-loop eval → download the verified
checkpoint → deploy to the local sim box → offline HTML report. Fully resumable
after any disconnect.

## Engines used (resolved via workspace.yaml keys)

- `isaac_gr00t` — GR00T N1.5/N1.6 upstream (finetune scripts; also the remote train cwd)
- `isaac_gr00t_n1d7` — GR00T N1.7 (`examples/finetune.sh`, uv env)
- `starvla` — StarVLA training/deployment

## Owns

- `harness/` — the train→eval→deploy state machine (`core/`, `stages/`, `web/`)
- `trainbot.py` — the single CLI entry point
- `data_generation.sh` — IsaacLab mimic dataset-generation launcher
- `tests/` — CPU-only unit tests for the harness

## Boundaries

- Local evaluation belongs to **evalbot**; this agent's EVAL stage runs evalbot's
  harness remotely on the H100 checkout.
- Shared transport tools (`agents/tools/common/pegasus.py`, `run_finetune.py`) are
  consumed, not owned.

## Handoffs

- Checkpoints → `artifacts/checkpoints/{gr00t,starvla}/<run>/` → **evalbot**, **agentbot**
  (`vla.checkpoints` registry / `stack.gr00t_checkpoint`)
- Training metrics → `var/logs/metrics.jsonl` → the report generator
