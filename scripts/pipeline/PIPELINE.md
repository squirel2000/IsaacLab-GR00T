# Automated Fine-tuning → Deploy Pipeline

Runs the whole GR00T fine-tuning workflow hands-off and **resumable**: wait for a free
H100 on the remote Pegasus box, fine-tune, pull the verified checkpoint home, switch
Wi-Fi to the lab LAN, push the checkpoint to the **asus-4090** sim box, switch back to
the internet, and write an offline HTML report.

```
IDLE → GPU_WAIT → TRAINING → DOWNLOADING → DEPLOYING → REPORTING → DONE
```

Sim validation is **run manually** on asus-4090 after deploy (see below).

> Prefer pictures? Open **[`docs/pipeline_architecture.html`](./docs/pipeline_architecture.html)**
> in a browser for an illustrated walkthrough of every script and how they connect.

## One entry point

`gr00t_pipeline.py` is the single CLI for everything:

```powershell
# one-time setup
pip install -r requirements-pipeline.txt
copy scripts\pipeline\config.example.yaml scripts\pipeline\config.yaml   # edit paths/passwords

python gr00t_pipeline.py run                  # resume (or start) the full pipeline  ← default
python gr00t_pipeline.py run --reset          # wipe state and run from IDLE
python gr00t_pipeline.py run --profile n1d5   # use the N1.5 (conda) profile instead of N1.7
python gr00t_pipeline.py status               # print state and exit (no network)
python gr00t_pipeline.py stop                 # terminate the detached training run
python gr00t_pipeline.py dashboard            # read-only live web UI at http://localhost:8770
python gr00t_pipeline.py finetune [run|monitor|watch|download|status|stop|selftest]
```

The pipeline persists progress in `pipeline_state.json` after **every** stage. Kill it,
lose Wi-Fi, reboot — re-run `run` and it continues from the interrupted stage. Training
keeps running on Pegasus regardless (launched detached with `setsid`+`nohup`), and
`run --reset` is the only thing that discards progress.

### `run`/`status`/`stop`/`dashboard`/`finetune` are not new code

`gr00t_pipeline.py` is a thin dispatcher — one shared implementation per behavior, nothing
reimplemented:

| Subcommand | Delegates to |
|---|---|
| `run` / `status` / `stop` | `pipeline_runner.run_cli` / `show_status` (core) |
| `dashboard` | `dashboard.serve` (web) |
| `finetune …` | `run_finetune.main` (scripts/common — also runnable standalone) |

`gr00t_pipeline.py` (repo root) is the **single** entry point; the old `run_pipeline.py` /
`run_dashboard.py` shims were removed (their `run`/`dashboard` subcommands replace them).

> `finetune` runs the **fine-tune only** (no download/deploy/report) using
> `run_finetune.py`'s own `CONFIG` block — a lower-level manual tool
> (`python scripts/common/run_finetune.py …`), separate from the orchestrated `run` which is
> driven by `config.yaml`.

## Layout

```
scripts/
  common/      pegasus.py · run_finetune.py · wifi_switch.py        (shared low-level tools)
  pipeline/
    core/      pipeline_runner · pipeline_state/config/progress/paths/logging/retry · net_util
    stages/    gpu_monitor · training_monitor · downloader · deployer · report_generator · sim_validator
    web/       dashboard.py · dashboard.html · vendor/chart.umd.min.js
    config/    config.yaml (gitignored) · config.example.yaml
    PIPELINE.md
  eval/        closed-loop IsaacSim eval (folded into the pipeline in a later phase)
```

Generated runtime artifacts (`pipeline_state.json`, `logs/`, `report_*.html`) stay at the
**repo root**. The only root entry point is `gr00t_pipeline.py`.

## Design (reuses existing tools — does not reinvent)

| Stage | Module | Built on |
|---|---|---|
| GPU_WAIT | `gpu_monitor.py` | `pegasus.sh` → `nvidia-smi` (idle = util<10% AND mem<5 GB; `gpu_priority` orders candidates) |
| TRAINING | `training_monitor.py` | `run_finetune.start/monitor` (detached, log-tail → `logs/metrics.jsonl`) |
| DOWNLOADING | `downloader.py` | `run_finetune.download` (resumable + sha256-verified) |
| DEPLOYING | `deployer.py` | `net_util` Wi-Fi switch + ping + **paramiko** SFTP/unzip to asus-4090 |
| REPORTING | `report_generator.py` | offline HTML + inlined Chart.js loss curve |

Pegasus has **no SSH** — it is the Jupyter HTTP/WebSocket box driven by `pegasus.py`.
asus-4090 **does** use SSH, via paramiko (password auth, no sshpass needed).

## Resume & stop semantics

- **Resume**: `run` (the default) re-reads `pipeline_state.json` and re-enters
  `state.current`. For TRAINING it re-attaches to the live server run if it is still
  `RUNNING`; otherwise it relaunches, and the HF Trainer auto-resumes from the latest
  `checkpoint-*` in `output_dir` (so a killed/rebooted job loses no progress).
- **Stop**: `stop` (or the dashboard's Stop button) kills the detached server job by its
  `output_dir` pattern in the process argv, then marks the stage `stopped` via
  `PipelineState.stop()` so `status` reflects reality. `run` afterwards resumes from the
  last checkpoint; `run --reset` starts over.

## Two training profiles

`config.yaml` defines `profiles.n1d5` (conda) and `profiles.n1d7` (uv); `active_profile`
(or `--profile`) selects one. Each has a `train_cmd_template` with `{gpu}`,
`{dataset_path}`, `{output_dir}`, `{max_steps}` placeholders. The active N1.7 template
drives the repo's own `examples/Openarm_LinkerHandO6/finetune_openarm_o6.sh` via env
overrides (`MODE=right_only NUM_GPUS=1 USE_WANDB=1 … DATALOADER_NUM_WORKERS=4
EVAL_STRATEGY=steps …`).

## Configuration & secrets

- `config.yaml` is **gitignored** (it holds the asus-4090 LAN password, the Pegasus
  password, and your per-user wandb key). Commit only `config.example.yaml`.
- **wandb**: `wandb.api_key` in `config.yaml` is injected into the training job's
  *environment* at launch (`WANDB_API_KEY`) — never committed, never echoed into logs.
  Each user's gitignored config holds their own key → their own wandb account.
- Per-module logs in `logs/` (gitignored). Reports `report_<timestamp>.html` at repo root.

## Network switching (Wi-Fi)

The Pegasus stages need the internet; DEPLOYING needs the OMAP lab LAN. `net_util` handles
the hand-off around a Windows quirk: a manual `netsh wlan connect` to the enterprise SSID
(`CJ86GJI4_5G`) needs elevation, and `netsh` can't even read the current SSID without
Location access. So:

- To go **online** (`ensure_external` / `ensure_reachable`): if `ping 8.8.8.8` already
  succeeds, do nothing; otherwise just `netsh wlan disconnect` the LAN and let CJ
  auto-reconnect at the system level — confirmed by ping, **not** by `netsh`'s SSID read.
- To reach the **LAN** (`switch_wifi('omap')`): connect to the OMAP profile (retries, since
  the profile can be briefly "not available" right after dropping the other network).

This is why the final "switch back to CJ" never throws even though `netsh connect` would:
connectivity is restored by disconnect-and-auto-reconnect, and verified by ping.

## Manual sim validation (on asus-4090, after deploy)

```bash
ssh asus@192.168.32.185
cd /home/asus/Gits/IsaacLab-GR00T
python launch_isaac_policy.py     # policy server + IsaacLab client; ~50–100 pick-and-place eps
```

To automate later, re-insert a `SIMULATING` stage in `pipeline_state.ORDER` and implement
`sim_validator.run` (parse `success_rate: X.XX`); the report already has a slot for it.

## Live progress dashboard

`python gr00t_pipeline.py dashboard` serves a read-only page
(default `http://localhost:8770`, and the LAN IP for phones) that auto-refreshes: stage
timeline, the active phase's %/rate/ETA, live loss / eval-loss / lr / grad-norm charts, a
Stop button, and run outputs. It reads `pipeline_state.json` + `logs/progress.json` +
`logs/metrics.jsonl` — no extra dependencies (stdlib `http.server`). Metrics are re-parsed
only when the file changes (mtime+size cache), so many polling clients stay cheap.

## Verify it works

```powershell
python -m unittest discover -s tests -p "test_*.py"   # 60 unit tests (logic, parsers, orchestration)
python gr00t_pipeline.py status                       # state machine reads cleanly
```

## Proposal: running the loop on the H100 instead of the laptop

**Question.** The laptop may sleep/disconnect, forcing a `run` re-attach. Can the whole
loop live on the H100 so you only open the dashboard to watch?

**What is already H100-side and disconnect-proof.** Training itself runs *on* Pegasus,
detached (`setsid`+`nohup`); GPU_WAIT/TRAINING/DOWNLOADING only *observe* it over HTTP.
Losing the laptop never interrupts training — `run` just re-attaches. So the part that
matters most is already safe.

**Why the loop cannot fully move to the H100.** DEPLOYING must (a) reach the asus-4090 over
the **OMAP lab LAN**, and (b) switch the **laptop's** Wi-Fi. Pegasus is on a different
network and cannot route to the OMAP LAN, nor can it control the laptop's adapter. So the
final deploy + Wi-Fi hand-off is intrinsically a laptop-side action.

**Recommended split (best of both):**

1. **Keep the orchestrator on the laptop** — it is the only host that can both reach
   Pegasus (internet) *and* the asus-4090 (OMAP LAN) and switch between them. This is
   already fully resumable; a disconnect costs one `run` to re-attach, nothing more.
2. **Make disconnects a non-event for the long phase.** GPU_WAIT + TRAINING is where the
   laptop would idle for hours. Run *that prefix* unattended on Pegasus and let the laptop
   join only for DOWNLOADING→DEPLOYING→REPORTING:
   - On Pegasus, the wrapper already writes `status` + `run.log`; nothing extra is needed
     for training to survive.
   - Optionally schedule `gr00t_pipeline.py run` on the laptop (Task Scheduler, "run
     whether logged on or not") so a wake/reboot auto-re-attaches without you typing
     anything. The state machine makes repeated `run` calls idempotent.
3. **Watch from anywhere** with the dashboard. To monitor while the laptop is asleep, run
   a second read-only `dashboard` on a host that stays up (or port-forward Pegasus's
   `run.log`); it only needs read access to the same status/log files.

Net: training is already immortal on the H100; the deploy/Wi-Fi tail is unavoidably
laptop-local; and a scheduled `run` turns "I had to retype `--resume`" into an automatic
re-attach. Moving the *whole* loop to the H100 would trade that small annoyance for losing
the ability to deploy at all.
