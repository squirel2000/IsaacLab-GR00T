# scripts/eval — GR00T / StarVLA closed-loop eval

Run policies in IsaacSim and score task success. **`run_eval.py` is the only command**; all
variables live in `configs/eval_config.yaml` (self-documented with comments). Results go to
`output/`, never here.

```
scripts/eval/
├── run_eval.py            # ★ the only command: top-level scheduler (eval missing runs, then call analysis)
├── gr00t_infer_agent.py   # IsaacSim client (launched by run_eval.py)
├── configs/               # eval_config.yaml (the plan + knobs) + per-backend JSONs
├── analysis/              # results: aggregate.py, compare_runs.py (table), make_loss_svg.py (figure), make_curve_svg.py
└── utils/                 # factory, adapters, joint_mapper, saver, filter, manifest
```

## Run it

```bash
cd /home/asus/Gits/IsaacLab-GR00T
python3 scripts/eval/run_eval.py              # or --target 50 for a quick pass
```

For each run in `configs/eval_config.yaml`: reuse its log if it already has `target` episodes,
otherwise eval it (crash-resilient — server stays up, client relaunches until `target` is reached).
Then print a comparison and write `output/analysis/eval_results.svg` (success bars + final train
loss). So re-running with no new checkpoints just re-summarises; add a run (or delete its log) to
make it (re)eval. Background a long batch: `nohup python3 scripts/eval/run_eval.py &`.

To change anything else (which checkpoints, episode count, `headless`, cameras, video, timeouts),
edit `configs/eval_config.yaml` — every field has a comment.

| flag | default | meaning |
|---|---|---|
| `--config <yaml>` | `configs/eval_config.yaml` | the eval plan |
| `--target <N>` | `100` (from config) | episodes per run |

## Where results go

| path | contents |
|---|---|
| `output/eval/logs/` | server / client / `*_combined_episodes.log` (source of truth; reused by tag) |
| `output/analysis/eval_results.svg` | success-vs-loss comparison chart (auto) |
| `IsaacLab/output/infer_record/<save_id>/` | per-run videos + `run_manifest.json` |

## Troubleshooting

| symptom | check |
|---|---|
| `SERVER NOT READY` | `<tag>_server.log`; checkpoint exists; N1.7 uses `Isaac-GR00T_n1d7/.venv`, N1.6 uses conda `env_gr00t` |
| client can't import `utils.*` | run_eval.py runs the agent by absolute path, so `from utils.*` resolves to `scripts/eval/utils/` |
| N1.7 client wire-format error | that backend JSON needs `client_pythonpath: Isaac-GR00T_n1d7` |
| black screen with `headless: false` | needs a graphical session on `:0`; else keep headless (cameras still render offscreen) |
| `conda.sh not found` | set `env.conda_sh` in `configs/eval_config.yaml` |
