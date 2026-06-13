# scripts/eval — GR00T / StarVLA closed-loop eval

Everything for running policies in IsaacSim and scoring task success. **The only entry
point is `run_eval.py`**; all variables live in `eval_config.yaml`. Results (logs, charts,
videos) go to `output/`, never here.

```
scripts/eval/
├── run_eval.py            # ★ the only command: eval missing runs, then compare + chart
├── eval_config.yaml       # all variables (eval plan + global knobs + env)
├── gr00t_infer_agent.py   # IsaacSim client (launched by run_eval.py)
├── compare_runs.py        # log aggregation + success-rate chart (called by run_eval.py; also standalone)
├── policy_configs/        # per-backend server/connection spec (gr00t_n15/16/17, starvla)
├── utils/                 # factory, adapters, joint_mapper, saver, filter, manifest
└── analysis/              # training-curve plots (make_loss_svg.py, make_curve_svg.py)
```

> The underlying `IsaacLab/`, `Isaac-GR00T/`, `Isaac-GR00T_n1d7/` checkouts stay in sync with
> NVIDIA upstream; the eval integration/switching settings live in this repo
> (`eval_config.yaml` + `policy_configs/`).

---

## 1. Run it

```bash
cd /home/asus/Gits/IsaacLab-GR00T
python3 scripts/eval/run_eval.py
```

That single command:

1. For each run in `eval_config.yaml`: if a complete log already exists (≥ `target` episodes),
   **reuse it**; otherwise **eval it** (launch server + IsaacSim client).
2. Print a success / timeout / unsafe comparison across all runs.
3. Write a success-rate comparison chart to `output/analysis/eval_results.svg`.

So re-running with no new checkpoints just re-prints the table and re-draws the chart. To
(re)eval a run, add it to `eval_config.yaml` or delete its `*_combined_episodes.log`.

Eval is **crash-resilient**: the server stays up and the IsaacSim client relaunches (Omniverse
PhysX occasionally crashes at an episode reset), accumulating each attempt's
`Episode N finished …` lines into `<tag>_combined_episodes.log` — the source of truth (the
run_manifest JSON only covers one attempt and isn't written on a crash).

For a long batch, background it:

```bash
nohup python3 scripts/eval/run_eval.py > output/eval/logs/orchestrator.log 2>&1 &
```

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--config <yaml>` | `scripts/eval/eval_config.yaml` | the eval plan |
| `--target <N>` | `100` (from config) | episodes to collect per run |
| `--headless` / `--no-headless` | `true` (from config) | run IsaacSim with / without a GUI window (cameras still render offscreen when headless; `--no-headless` shows the sim on display `:0`) |

Everything else (filter, multitask, pov_list, save_video, timeouts, max_attempts) is set in
`eval_config.yaml`, not on the CLI.

---

## 2. Change what runs — `eval_config.yaml`

- **`defaults:`** global knobs — `target`, `headless`, `pov_list` (options: `head`, `wrist_R`,
  `wrist_L`), `multitask`, `filter`, `save_video`, timeouts.
- **`env:`** integration settings — `isaaclab_conda_env`, `conda_sh` (leave empty to auto-detect
  from `CONDA_EXE`/`PATH`), and the `display` env applied only for `--no-headless` runs.
- **`runs:`** the eval plan. Each entry points at a `policy_configs/*.json` (the per-backend
  server spec) plus a `save_id`; `checkpoint:` overrides that JSON's `model_path` (a parent dir
  auto-resolves to its latest `checkpoint-*`).
- **Per-backend server config** (model path, port, embodiment, `server_repo`/`server_venv`,
  `client_pythonpath`) lives in `policy_configs/*.json` — one file per backend, no duplication.

**Add a checkpoint:** append a `runs:` entry pointing at the right backend JSON with a
`checkpoint:` and `save_id:`. Next `run_eval.py` evals only the new one (others are reused).

---

## 3. Re-analyse existing logs without eval — `compare_runs.py`

`run_eval.py` already does this (it reuses complete logs). To only re-print + re-chart:

```bash
python3 scripts/eval/compare_runs.py              # uses the headless logdir (config default)
python3 scripts/eval/compare_runs.py --no-headless # uses the windowed logdir
```

---

## 4. Training-curve plots — `analysis/`

Separate from eval results (these read each checkpoint's `trainer_state.json`):

```bash
# train loss (log) + grad norm + hardcoded success bars
python3 scripts/eval/analysis/make_loss_svg.py output/analysis/n16_vs_n17/n16_vs_n17_loss.svg
# train loss + grad norm for any single checkpoint
python3 scripts/eval/analysis/make_curve_svg.py <checkpoint>/trainer_state.json [out.svg]
```

---

## 5. Where results go

| path | contents |
|---|---|
| `output/eval/logs/` , `output/eval/logs_headless/` | server / client / `*_combined_episodes.log` |
| `output/analysis/eval_results.svg` | success-rate comparison chart (auto-written by run_eval.py) |
| `IsaacLab/output/infer_record/<save_id>/` | per-run videos + `run_manifest.json` |
| `output/eval/n16_vs_n17_300k_report.md` | the written-up findings |

> run_eval.py truncates a run's combined log only when it actually re-evals that run. To keep
> old evidence around, copy the logdir first: `cp -r output/eval/logs output/eval/logs.bak`.

---

## 6. Troubleshooting

| symptom | check |
|---|---|
| `SERVER NOT READY` | `<tag>_server.log`; checkpoint exists; N1.7 uses `Isaac-GR00T_n1d7/.venv`, N1.6 uses conda `env_gr00t` |
| client can't import `utils.*` | run_eval.py runs `gr00t_infer_agent.py` by absolute path, so its `from utils.*` resolves to `scripts/eval/utils/` |
| N1.7 client wire-format error | that run's `policy_configs/gr00t_n17_*.json` needs `client_pythonpath: Isaac-GR00T_n1d7` |
| black screen in `--no-headless` | needs a graphical session on `:0`; otherwise use headless (cameras still render offscreen, `enable_cameras=True`) |
| `conda.sh not found` | set `env.conda_sh` in `eval_config.yaml` |
