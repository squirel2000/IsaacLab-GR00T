# `scripts/`

Tooling for the IsaacLab-GR00T workspace.

```
scripts/
  pipeline/   automated fine-tune → eval → deploy pipeline   (see pipeline/PIPELINE.md)
    core/     state machine + config / progress / paths / logging / net_util
    stages/   one module per stage (gpu_monitor, training_monitor, eval_runner, downloader, deployer, report_generator)
    web/      read-only live dashboard (dashboard.py + dashboard.html)
    config/   config.yaml (gitignored) + config.example.yaml
  common/     shared low-level tools: pegasus.py (H100 Jupyter transport), run_finetune.py
              (detached train / monitor / download), wifi_switch.py
  eval/       closed-loop IsaacSim eval harness   (see eval/README.md)
```

## Run the pipeline

```powershell
copy scripts\pipeline\config\config.example.yaml scripts\pipeline\config\config.yaml   # edit paths/passwords
python scripts/gr00t_pipeline.py run          # resume or start  (run --reset for fresh)
python scripts/gr00t_pipeline.py dashboard    # live web UI at http://localhost:8770
python scripts/gr00t_pipeline.py status|stop|finetune
```

Flow: **`GPU_WAIT → TRAINING → EVAL → DOWNLOADING → DEPLOYING → REPORTING → DONE`**. Every stage
persists to `pipeline_state.json` and resumes after a disconnect; training + eval run **detached on
the H100**. The **EVAL** stage runs `scripts/eval/run_eval.py` on the H100 (gated by config
`eval.enabled`; off until the H100 has this branch + its IsaacSim env).

## Run eval standalone

```bash
python scripts/eval/run_eval.py            # eval the checkpoints in scripts/eval/configs/eval_config.yaml
```

Full reference: [pipeline/PIPELINE.md](pipeline/PIPELINE.md) · illustrated:
[../docs/pipeline_architecture.html](../docs/pipeline_architecture.html).
