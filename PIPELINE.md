# Automated Fine-tuning → Deploy Pipeline

`run_pipeline.py` runs the whole GR00T fine-tuning workflow hands-off and
**resumable**: wait for a free H100 on the remote Pegasus box, fine-tune, pull the
verified checkpoint home, switch Wi-Fi to the lab LAN, push the checkpoint to the
**asus-4090** sim box, switch back to the internet, and write an offline HTML report.

```
IDLE → GPU_WAIT → TRAINING → DOWNLOADING → DEPLOYING → REPORTING → DONE
```

Sim validation is **run manually** on asus-4090 after deploy (see below).

## Layout

The automation code lives in the **`pipeline/`** package (modules + `dashboard.html` +
vendored `chart.umd.min.js`); the reused tools (`pegasus.py`, `run_finetune.py`,
`wifi_switch.py`) stay at the repo root, as do the runtime artifacts (`config.yaml`,
`pipeline_state.json`, `logs/`, `report_*.html`). Two thin root entry points dispatch in:
`run_pipeline.py` (orchestrator) and `run_dashboard.py` (live UI).

## Quick start

```powershell
# one-time
pip install -r requirements-pipeline.txt
$env:PEGASUS_PASSWORD = 'eksncl#...'         # Pegasus (training server) password
copy config.example.yaml config.yaml         # then edit paths / asus-4090 password

python run_pipeline.py --reset             # start fresh from IDLE
python run_pipeline.py --resume            # continue after a disconnect (default)
python run_pipeline.py --status            # show current state, do nothing
python run_pipeline.py --profile n1d5      # use the N1.5 (conda) profile instead of N1.7

python run_dashboard.py                    # live web dashboard at http://localhost:8770
```

The pipeline persists progress in `pipeline_state.json` after every stage. Kill it, lose
Wi-Fi, reboot — re-run `--resume` and it continues from the interrupted stage. Training
keeps running on Pegasus regardless (launched detached with `setsid`+`nohup`).

## Design (reuses existing tools — does not reinvent)

| Stage | Module | Built on |
|---|---|---|
| GPU_WAIT | `gpu_monitor.py` | `pegasus.sh` → `nvidia-smi` (idle = util<10% AND mem<5 GB) |
| TRAINING | `training_monitor.py` | `run_finetune.start/monitor` (detached, log-tail → `logs/metrics.jsonl`) |
| DOWNLOADING | `downloader.py` | `run_finetune.download` (resumable + sha256-verified) |
| DEPLOYING | `deployer.py` | `wifi_switch` + ping + **paramiko** SFTP/unzip to asus-4090 |
| REPORTING | `report_generator.py` | offline HTML + inlined Chart.js loss curve |

Pegasus has **no SSH** — it is the Jupyter HTTP/WebSocket box driven by `pegasus.py`.
asus-4090 **does** use SSH, via paramiko (password auth, no sshpass needed).

## Two training profiles

`config.yaml` defines `profiles.n1d5` (conda) and `profiles.n1d7` (uv); `active_profile`
(or `--profile`) selects one. Each has a `train_cmd_template` with `{gpu}`,
`{dataset_path}`, `{output_dir}`, `{max_steps}` placeholders. **Confirm the N1.7 template
matches your run** — it is a best-effort default (`uv run python
gr00t/experiment/launch_finetune.py …`).

## Configuration & secrets

- `config.yaml` is **gitignored** (it holds the asus-4090 LAN password). Commit only
  `config.example.yaml`.
- Pegasus password: `PEGASUS_PASSWORD` env var. asus-4090 password: `config.yaml`
  (`asus4090.password`) or `ASUS4090_PASSWORD` env.
- Per-module logs in `logs/` (gitignored). Reports `report_<timestamp>.html` at repo root.

## Manual sim validation (on asus-4090, after deploy)

```bash
ssh asus@192.168.32.185
cd /home/asus/Gits/IsaacLab-GR00T
python launch_isaac_policy.py     # policy server + IsaacLab client; ~50–100 pick-and-place eps
```

To automate later, re-insert a `SIMULATING` stage in `pipeline_state.ORDER` and implement
`sim_validator.run` (parse `success_rate: X.XX`); the report already has a slot for it.

## Verify it works

```powershell
python -m unittest discover -s tests -p "test_*.py"   # 52 unit tests (logic, parsers, orchestration)
python run_pipeline.py --status                       # state machine reads cleanly
```

## Live progress dashboard

`python run_dashboard.py` serves a read-only page (default `http://localhost:8770`, and
the LAN IP for phones) that auto-refreshes: stage timeline, the active phase's %/rate/ETA,
a live loss chart, and run outputs. It reads `pipeline_state.json` + `logs/progress.json`
+ `logs/metrics.jsonl` — no extra dependencies (stdlib `http.server`).
