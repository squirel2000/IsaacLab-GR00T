# agents/evalbot/harness — GR00T / RLDX-1 / StarVLA closed-loop eval

Run policies in IsaacSim and score task success. **`run_eval.py` is the only command**; all
variables live in `configs/eval_config.yaml` (self-documented with comments). Results go to
`agents/evalbot/var/`, never here.

```
agents/evalbot/harness/
├── run_eval.py            # ★ the only command: top-level scheduler (eval missing runs, then call analysis)
├── gr00t_infer_agent.py   # IsaacSim client (launched by run_eval.py)
├── pegasus_deploy.py      # ★ Pegasus only — upload this harness there + verify the remote env
├── pegasus_launch.py      # ★ Pegasus only — launch run_eval.py there, detached, on an idle GPU
├── pegasus_watch.py       # ★ Pegasus only — poll a launched run until it finishes
├── configs/               # eval_config.yaml (the plan + knobs) + per-backend JSONs
│   └── pegasus/           # path-adjusted copies of the above for Pegasus's layout — see PEGASUS.md
├── analysis/              # results: aggregate.py, compare_runs.py (table), make_loss_svg.py (figure), make_curve_svg.py
└── utils/                 # factory, adapters, joint_mapper, saver, filter, manifest
```

## Run it locally

```bash
python agents/evalbot/harness/run_eval.py              # or --target 50 for a quick pass
```

For each run in `configs/eval_config.yaml`: reuse its log if it already has `target` episodes,
otherwise eval it (crash-resilient — server stays up, client relaunches until `target` is reached).
Then print a comparison and write `agents/evalbot/var/analysis/eval_results.svg` (success bars +
final train loss). So re-running with no new checkpoints just re-summarises; add a run (or delete
its log) to make it (re)eval.

To change anything else (which checkpoints, episode count, `headless`, cameras, video, timeouts),
edit `configs/eval_config.yaml` — every field has a comment.

| flag | default | meaning |
|---|---|---|
| `--config <yaml>` | `configs/eval_config.yaml` | the eval plan |
| `--target <N>` | `100` (from config) | episodes per run |

## Run it on Pegasus

**Full walkthrough, environment inventory, and the four bugs this surfaced: [PEGASUS.md](PEGASUS.md).**
Short version, from a machine that already has `agents/tools/common/pegasus.py` working (any
checkout of this repo, once you have the Pegasus password — see that module):

```bash
python agents/evalbot/harness/pegasus_deploy.py --config configs/pegasus/eval_config_phase1_comparison.yaml   # once, or whenever this harness changes
python agents/evalbot/harness/pegasus_launch.py --config configs/pegasus/eval_config_phase1_comparison.yaml   # picks an idle GPU, launches detached
python agents/evalbot/harness/pegasus_watch.py  --log <path pegasus_launch.py printed>                        # optional — blocks until done
```

Write your own `configs/pegasus/<name>.yaml` for a different set of runs — copy an existing one
in that directory; PEGASUS.md explains which fields need Pegasus-specific values and why.

## Where results go

| path | contents |
|---|---|
| `agents/evalbot/var/eval/logs/` (local) or `<logdir>` from the config (Pegasus) | server / client / `*_combined_episodes.log` (source of truth; reused by tag) |
| `agents/evalbot/var/analysis/eval_results.svg` | success-vs-loss comparison chart (auto) |
| `<isaaclab_repo>/output/infer_record/<save_id>/` | per-run videos + `run_manifest.json` |

## Troubleshooting

| symptom | check |
|---|---|
| `SERVER NOT READY` | `<tag>_server.log`; checkpoint exists; N1.7/RLDX-1 use their own `.venv`, older backends use conda |
| client can't import `utils.*` | run_eval.py runs the agent by absolute path, so `from utils.*` resolves to this dir's own `utils/` |
| N1.7 client wire-format error | that backend JSON needs `client_pythonpath: <path to the matching GR00T checkout>` |
| black screen with `headless: false` | needs a graphical session on `:0`; else keep headless (cameras still render offscreen) |
| `conda.sh not found` | set `env.conda_sh`, or (Pegasus and similar boxes with no working conda binary) `env.isaaclab_conda_prefix` — see PEGASUS.md |
| a client that finished all its episodes never advances past "client attempt N" | Kit's `simulation_app.close()` can hang after a fully successful run, not just after an exception — this is what `gr00t_infer_agent.py`'s `os._exit(0)` wrapper at the bottom of the file is for; if you're running a fork without that fix, port it over |
