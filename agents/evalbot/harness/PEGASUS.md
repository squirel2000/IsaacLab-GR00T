# Running evalbot on Pegasus

This is the "why" half of running IsaacSim closed-loop eval on the Pegasus 2×H100 box.
`pegasus_deploy.py` / `pegasus_launch.py` / `pegasus_watch.py` are the "make it true" half —
read this once, then those three scripts are all you need day to day.

## Prerequisites

Anyone with a checkout of this repo and the Pegasus password can do this — nothing here is
tied to one machine or one person's setup. You need:

- `agents/tools/common/pegasus.py` working (`pegasus.connect()` — see that module's own docs
  for how the password is supplied; it is *not* committed to the repo).
- Nothing else installed locally. All three `pegasus_*.py` scripts only talk to Pegasus over
  the same Jupyter HTTP/WebSocket channel every other Pegasus tool in this repo uses — there is
  no SSH to this box, unlike the asus-4090 sim box.

## What's already on Pegasus (as of 2026-08-19)

| Thing | Path | Notes |
|---|---|---|
| IsaacSim + IsaacLab | `/data/VLA/tingying/envs/env_isaaclab` (conda env) | isaacsim 5.1.0.0, isaaclab 0.54.3, installed *editable* against `/data/VLA/tingying/IsaacLab-GR00T/IsaacLab`. Extension versions match this repo's `IsaacLab` checkout exactly (isaaclab_tasks 0.11.14, isaaclab_assets 0.2.4, isaaclab_rl 0.5.0, isaaclab_mimic 1.0.16). |
| GR00T N1.7 | `/data/VLA/tingying/IsaacLab-GR00T/Isaac-GR00T_n1d7` | its own `.venv` (uv) |
| RLDX-1 | `/data/VLA/tingying/RLDX-1` | its own `.venv` (uv). **Not** under `IsaacLab-GR00T/` — a sibling checkout, one level up. |
| checkpoints | `/data/VLA/tingying/IsaacLab-GR00T/artifacts/checkpoints/gr00t/...` (GR00T), `/data/VLA/tingying/artifacts/rldx1/...` (RLDX-1) | |
| evalbot harness | `/data/VLA/tingying/IsaacLab-GR00T/agents/evalbot/` | uploaded by `pegasus_deploy.py`; not part of any git checkout on that box |
| workspace root marker | `/data/VLA/tingying/IsaacLab-GR00T/workspace.yaml` | see "Path layout" below |

`env_isaaclab` also now has `pandas`, `opencv-python-headless`, `matplotlib`, `msgpack`,
`msgpack-numpy`, `pyzmq`, `pyarrow` installed (added 2026-08-19/21 — this harness had simply
never been run through that specific env before, so nothing beyond IsaacLab's own deps had
ever been needed). `pegasus_deploy.py` checks for these every time — including an actual
`pandas.DataFrame.to_parquet()` call, not just `import pandas` — and installs anything
missing, so this table will drift and that step won't.

## Path layout: Pegasus is flat, this repo is not

Locally, VLA engines live under `engines/{sim,vla}/...` (e.g. `engines/vla/Isaac-GR00T_n1d7`,
`engines/sim/IsaacLab`). **Pegasus predates that restructure** and was never migrated: on
Pegasus, the same checkouts sit directly under `IsaacLab-GR00T/` with no `engines/` level at
all (`IsaacLab-GR00T/Isaac-GR00T_n1d7`, `IsaacLab-GR00T/IsaacLab`), and RLDX-1 isn't under
`IsaacLab-GR00T/` at all — it's a sibling checkout (`/data/VLA/tingying/RLDX-1`).

`run_eval.py` resolves every relative config path (`server_repo`, `client_pythonpath`,
`isaaclab_repo`, checkpoint paths) against a `ROOT` found by walking up from `run_eval.py`'s own
location until a `workspace.yaml` marker file is found (same mechanism `agents/tools/common/*`
uses elsewhere in this repo). Locally that marker is this repo's own root. On Pegasus there is
no such file naturally, so `pegasus_deploy.py` creates one at `IsaacLab-GR00T/workspace.yaml` —
after that, every config under `configs/pegasus/` just needs paths written relative to
`IsaacLab-GR00T/` (or, for RLDX-1, an absolute path, since it's outside that tree entirely).

**This is why `configs/pegasus/` exists as a separate copy of each config/JSON rather than one
shared file with environment-specific overrides**: the *values* differ (`Isaac-GR00T_n1d7` vs
`engines/vla/Isaac-GR00T_n1d7`), not just which environment is active. Write a new Pegasus
config by copying the nearest existing one in that directory and adjusting checkpoint paths —
don't try to reuse the main `configs/*.json` files with path overrides bolted on.

## Five bugs this surfaced, all fixed at the source

None of these are Pegasus-only workarounds patched over the top — all five are fixed inside
`run_eval.py` / `gr00t_infer_agent.py` / `pegasus_deploy.py` themselves, so a future box
(including the 4090) gets the fix automatically rather than needing to rediscover it.

1. **No `"rldx"` branch in `server_cmd()`/`server_workdir_activate()`.** Any RLDX-1 run would
   have silently tried to launch `gr00t.eval.run_gr00t_server` — the wrong module. Added,
   using the exact flags verified in the RLDX-1 serving smoke test (see the openspec change's
   task 6.5).
2. **`main()` called `find_conda_sh()` unconditionally**, which raises `SystemExit` if no
   working conda is found. Pegasus's `env_isaaclab` has no `bin/activate` and there is no
   working `conda` binary on the box at all — `conda/` there is a package cache only. Every run
   would fail before any of them started. Fixed with `isaaclab_activate()`: set
   `env.isaaclab_conda_prefix` to an explicit env directory and it takes priority over
   `conda activate NAME`; `conda_sh` lookup is now lazy, only needed by the (unused-on-Pegasus)
   non-venv server path.
3. **Missing Python packages** in `env_isaaclab` (see the table above) — this harness had
   never been run through that env before. `pegasus_deploy.py` checks and installs.
4. **`simulation_app.close()` hangs indefinitely on this box** — not only after an unhandled
   exception (that pattern is already documented elsewhere in this project as a known Kit
   quirk), but *after a fully successful, clean run too*. Confirmed directly: a pre-fix run
   completed both its episodes (readable straight from the raw client log) but the scheduler's
   own bookkeeping never advanced, because `subprocess.run()` was blocked waiting on a process
   that had already done everything it needed to (every result is flushed to disk —
   `finalize_manifest`, `write_manifest`, `simulation_note.txt`, and this process's own
   unbuffered stdout — *before* `env.close()`/`simulation_app.close()` are ever called). Fixed:
   the very end of `gr00t_infer_agent.py` now runs `simulation_app.close()` in a daemon thread
   with a 15s join, then calls `os._exit(0)` unconditionally. Without this, a successful
   100-episode attempt would burn the *entire* `client_timeout_s` (14,400s / 4h in the main
   config) on top of its real runtime, for every single attempt.
5. **A missing dependency that "passed" the dependency check.** `pandas` imports fine with
   zero parquet engines installed — `pyarrow` and `fastparquet` are optional, checked lazily
   only when `DataFrame.to_parquet()` actually runs. `EpisodeDataSaver._save_episode_parquet()`
   only runs when `save_video: true`, which the very first smoke test happened to have off, so
   the gap survived that check. Found the expensive way: the real 100-episode sweep burned
   ~44 of its first 48 hours retrying the identical failure, because the resulting exception
   hit the *same* Kit-teardown hang as finding 4, on the path that fix didn't cover (mid-`main()`,
   not the final clean-exit path) — each attempt hung for the full 4h timeout before being
   killed and retried into the same bug. Fixed three ways: installed `pyarrow`; wrapped the
   per-episode save call in a `try/except` (a save failure must not cost the *remaining*
   episodes); wrapped `main()` itself so any future exception there also reaches the fast-exit
   path instead of relying on the timeout to notice. `pegasus_deploy.py`'s check now calls
   `to_parquet()` for real, not just `import pandas` — and does so without piping through
   `tail` first, since `cmd | tail; echo $?` reads `tail`'s exit code, not `cmd`'s, which had
   made an earlier version of that same check silently pass regardless of the real result.
   **Lesson for the next gap on a new box:** verifying an import succeeds is not the same as
   verifying the feature that harness needs from it works — test with the real configuration's
   flags on, not a cheaper stand-in, before trusting a long unattended run to it.

Also needed, not a bug but easy to miss: **`OMNI_KIT_ACCEPT_EULA=YES`** in the environment —
headless Kit otherwise blocks on an interactive EULA prompt on its very first run and dies on
EOF. `run_eval.py`'s `main()` sets this unconditionally now.

## Walkthrough

```bash
# 1. One-time (or whenever the harness changes locally) — upload + verify.
python agents/evalbot/harness/pegasus_deploy.py

# 2. Launch a config. Picks an idle GPU (util<10%, mem<5GB — same rule as every other
#    training/eval launcher in this repo) and refuses rather than queuing if nothing
#    qualifies. Runs detached (setsid+nohup+disown), so it survives you disconnecting.
python agents/evalbot/harness/pegasus_launch.py \
    --config configs/pegasus/eval_config_phase1_comparison.yaml

# 3. Optional: block here until it's done (hours, for a real 100-episode sweep), printing
#    each run's own "DONE --" line as it lands rather than the full episode-by-episode log.
python agents/evalbot/harness/pegasus_watch.py --log <path printed by step 2>
```

Every run in a config's `runs:` list that already has `target` episodes logged is skipped and
reused — so re-running the same config after adding a new entry only evaluates the new one.

## Writing a new Pegasus config

Copy the nearest existing file under `configs/pegasus/` and change:
- `runs[].checkpoint` — absolute path, or relative to `IsaacLab-GR00T/` (not `engines/...`).
- `runs[].config` — the per-backend JSON filename, in the *same* `configs/pegasus/` directory.
- `defaults.logdir` — give each config its own logdir (`/data/VLA/tingying/pegasus_runs/evalbot_<name>/logs`) so concurrent sweeps don't collide.

For a **new RLDX-1 checkpoint**: reuse `configs/pegasus/rldx1_openarm_o6.json` as-is (its
`model_path` is just a default; `checkpoint:` in the YAML run entry overrides it) — just point
a new run entry's `checkpoint:` at the new checkpoint, with `policy: rldx`.

For a **new GR00T N1.7 checkpoint**: same idea with `configs/pegasus/gr00t_n17_openarm_o6.json`.

## Known limits / not yet exercised

- `jobs > 1` (parallel runs) has not been tried on Pegasus. The main config's own comments
  suggest 2-4 is plausible on an 80GB H100 (vs. 1 on a 24GB 4090) — worth trying once a
  sequential sweep has proven reliable, since three ~2-2.5h sequential runs is the current
  baseline cost for a 100-episode-each comparison.
- `save_video: true` was left on for the first full run; watch `df -h /data` if running many
  models at 100 episodes each — this box has run low on disk before (see the openspec change's
  task 6.6 for the full incident and cleanup).
- Nothing here manages disk cleanup automatically. Old `pegasus_runs/evalbot_*/` directories
  and superseded checkpoints are the user's/operator's call to prune, same as the training side.
