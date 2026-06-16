# Automated Fine-tune → Eval → Deploy Pipeline

Hands-off and **resumable**: wait for a free H100 on Pegasus, fine-tune, **eval closed-loop in
IsaacSim**, pull the verified checkpoint home, deploy it to the **asus-4090**, and write an offline
report — surviving any disconnect.

```
IDLE → GPU_WAIT → TRAINING → EVAL → DOWNLOADING → DEPLOYING → REPORTING → DONE
```

> Illustrated walkthrough: **[../../docs/pipeline_architecture.html](../../docs/pipeline_architecture.html)**.
> Folder layout: **[../README.md](../README.md)**.

## One entry point

```powershell
copy scripts\pipeline\config\config.example.yaml scripts\pipeline\config\config.yaml   # edit paths/passwords
python scripts/gr00t_pipeline.py run                 # resume (or start) the full pipeline  ← default
python scripts/gr00t_pipeline.py run --reset         # wipe state, run from IDLE
python scripts/gr00t_pipeline.py run --profile n1d5  # pick a training profile
python scripts/gr00t_pipeline.py status              # print state and exit (no network)
python scripts/gr00t_pipeline.py stop                # terminate the detached training run
python scripts/gr00t_pipeline.py dashboard           # read-only live web UI at http://localhost:8770
python scripts/gr00t_pipeline.py finetune […]        # fine-tune-only tool (scripts/common/run_finetune.py)
```

`scripts/gr00t_pipeline.py` is a thin dispatcher — one shared implementation per behavior:
`run/status/stop` → `pipeline_runner.run_cli`, `dashboard` → `dashboard.serve`, `finetune` →
`run_finetune.main`. Progress persists to `pipeline_state.json` after every stage; training + eval
run detached on Pegasus (`setsid`+`nohup`) and survive a laptop drop. `run --reset` is the only
thing that discards progress.

## Stages (reuses existing tools — does not reinvent)

| Stage | Module (`scripts/pipeline/…`) | Built on |
|---|---|---|
| GPU_WAIT | `stages/gpu_monitor` | `pegasus.sh` → `nvidia-smi` (idle = util<10% & mem<5 GB; `gpu_priority` orders) |
| TRAINING | `stages/training_monitor` | `run_finetune` detached launch + log-tail → `logs/metrics.jsonl` |
| EVAL | `stages/eval_runner` | detached `scripts/eval/run_eval.py` on the H100 → success rate |
| DOWNLOADING | `stages/downloader` | `run_finetune` resumable + sha256 download |
| DEPLOYING | `stages/deployer` | `net_util` Wi-Fi switch + paramiko SFTP/unzip to asus-4090 |
| REPORTING | `stages/report_generator` | offline HTML: loss curve + eval success rate/chart |

Pegasus has **no SSH** (Jupyter HTTP/WS via `pegasus.py`); asus-4090 uses SSH (paramiko).

## EVAL stage — and how to turn it on

Closed-loop IsaacSim eval on the H100, right after training. **Off by default**, so the pipeline
runs without it (the stage becomes a no-op). When on, `eval_runner` writes a one-run
`eval_config.yaml` for the checkpoint, launches `run_eval.py` detached on the H100, polls
`<run>_combined_episodes.log`, records the success rate, and fetches the SVG chart for the report.

**Enabling EVAL is a config change — there is no command flag.** Steps:

1. On the H100: make sure this branch is checked out (`git pull`) and the IsaacSim +
   policy-server conda envs exist (the eval harness needs them).
2. Edit **`scripts/pipeline/config/config.yaml`** → under `eval:` set:
   ```yaml
   eval:
     enabled: true
     target_episodes: 100
     isaaclab_conda_env: env_isaaclab    # the H100's IsaacSim env
     # h100_repo_root / conda_sh: leave blank to auto-detect
   ```
3. Run as usual: `python scripts/gr00t_pipeline.py run` — EVAL now runs after TRAINING.

To eval an existing checkpoint **standalone** instead (no pipeline): edit
`scripts/eval/configs/eval_config.yaml` and run `python scripts/eval/run_eval.py`.

## Checkpoints

- **Location**: under `…/IsaacLab-GR00T/artifacts/checkpoints/gr00t/<run_name>/` (the n1d7 profile
  default). EVAL reads from there.
- **Packaging**: the download zips the **inference-ready top-level model only** (`config.json` +
  `model-*.safetensors` + `experiment_cfg/` + `processor/`), **excluding `checkpoint-*/`** — that
  HF training checkpoint (model + optimizer + scheduler) duplicates the weights and only matters for
  resume, so it stays on the server but is never shipped (~half the transfer).

## Resume & stop

- **Resume**: `run` re-enters `state.current`; completed stages skip, the interrupted one retries.
  TRAINING/EVAL re-attach to the live H100 job if still running, else relaunch (the HF Trainer
  auto-resumes training from the latest `checkpoint-*`).
- **Stop**: `stop` (or the dashboard button) kills the detached job by its `output_dir` argv pattern
  and marks the stage `stopped` (`PipelineState.stop()`), so `status` stays honest.

## Config, secrets, network

- `config/config.yaml` is **gitignored** (Pegasus + asus-4090 + wandb secrets); commit only
  `config.example.yaml`. The wandb key is injected into the training job's env at launch — never logged.
- Network: a manual `netsh wlan connect` to the enterprise SSID needs elevation, and `netsh` can't
  read the SSID without Location access. So `net_util` goes online by *disconnecting* the LAN and
  letting the SSID auto-reconnect, confirmed by **ping** (not `netsh`); it warns loudly if it can't.

## Dashboard & verify

`gr00t_pipeline.py dashboard` → auto-refreshing page: stage timeline, active-phase %/rate/ETA, live
loss (train color changes at each resume) / eval-loss / lr / grad-norm, eval success rate, Stop button.

```powershell
python -m unittest discover -s tests -p "test_*.py"   # 64 unit tests
python scripts/gr00t_pipeline.py status
```
