# vla-trainbot — runbook

Trains VLA models (GR00T N1.5/N1.6/N1.7, starVLA) into deployable checkpoints. The
harness is a resumable state machine: `GPU_WAIT → TRAINING → EVAL → DOWNLOADING →
DEPLOYING → REPORTING → DONE`, driving a remote H100 (Pegasus) and deploying the
verified checkpoint back to this machine.

## Prerequisites

```bash
pip install -r agents/vla-trainbot/requirements.txt
cp agents/vla-trainbot/harness/config/config.example.yaml \
   agents/vla-trainbot/harness/config/config.yaml          # edit hosts/paths/passwords (gitignored)
```

## Launch

```bash
python agents/vla-trainbot/trainbot.py run          # resume (or start) the full pipeline
python agents/vla-trainbot/trainbot.py run --reset  # wipe state, start from IDLE
python agents/vla-trainbot/trainbot.py status       # inspect state offline (touches nothing)
python agents/vla-trainbot/trainbot.py stop         # terminate the detached remote training
python agents/vla-trainbot/trainbot.py dashboard    # live web UI at http://localhost:8770
python agents/vla-trainbot/trainbot.py finetune …   # low-level manual tool (run_finetune.py)
```

Manual (non-automated) finetune commands live in each engine repo (e.g.
`engines/vla/Isaac-GR00T_n1d7/examples/finetune.sh`, `engines/vla/Isaac-GR00T/scripts/gr00t_finetune.py`);
data collection is launched by [`./data_generation.sh`](data_generation.sh) (IsaacLab
mimic flow; dataset metadata via
`engines/sim/IsaacLab/scripts/gr00t_script/utils/generate_dataset_meta.py`).

## Outputs

- Checkpoints → `artifacts/checkpoints/gr00t/<run>/` (data plane, shared)
- Run state / logs / reports → `var/` (gitignored): `var/pipeline_state.json`,
  `var/logs/metrics.jsonl`, `var/report_<ts>.html`

## Tests

```bash
python3 -m unittest discover -s agents/vla-trainbot/tests   # 64 CPU-only tests
```

## Troubleshooting

- `run/status` exits silently with code 1 → `harness/config/config.yaml` is missing
  (copy the example first) .
- Full reference: [harness/PIPELINE.md](harness/PIPELINE.md); illustrated:
  [agents/docs/pipeline_architecture.html](../docs/pipeline_architecture.html).
- The remote H100 must have this branch's layout for the EVAL stage
  (`agents/evalbot/harness/...` paths are resolved on the remote checkout).

## Handoffs

Verified checkpoints go to **evalbot** (local closed-loop validation) and **agentbot**
(`vla.checkpoints` registry / `stack.gr00t_checkpoint`).
