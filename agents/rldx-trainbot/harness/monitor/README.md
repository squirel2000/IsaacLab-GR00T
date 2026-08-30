# Training monitor — one page, any model

A local, disposable viewer for training running on Pegasus. It reads the training logs over
`pegasus.py`, parses them with vla-trainbot's own metric extractor, and writes a single static
HTML file you open in a browser. It does not affect training in any way — training runs
detached on Pegasus (`setsid nohup`), independent of whether this poller is running.

## Files

| File | Role |
| --- | --- |
| `runs.py` | **Edit this to add/change a run.** A list of `RunSpec` — label, log path, output dir, max steps, GPUs, batch, action space, notes. Everything else reads from here. |
| `poller.py` | The process you run. Loops: fetch logs → parse → write `index.html`. |
| `remote_tail.py` | Runs *on Pegasus* (uploaded automatically). Filters each log to the ~200 lines the parser needs and reports GPU occupancy, so one round trip stays small and reliable. |
| `training_bridge.py` | Imports vla-trainbot's `training_monitor.extract_metrics` so RLDX-1 and GR00T logs parse with the exact same regexes — no duplicate, driftable copy. |
| `render.py` | Builds the HTML: tabs, sparklines, ETA. No model-specific logic — reads only the fields `runs.py` defines. |
| `training_poller.py` | **Superseded, do not run.** An earlier prototype, kept only because deleting it wasn't asked for. `poller.py` replaced it. |
| `eval_poller.py` | A separate, unrelated poller for the Phase-0 LIBERO **eval** sweep (`eval_supervisor.sh`'s state), not a training run — writes `status.json` into the same `--outdir`. Promoted from `tmp/dashboard_poller.py` (openspec task 2.5). |
| `eval_refresh_once.py` | One-shot version of `eval_poller.py`, for when the sweep has already finished and a polling loop would be pointless. Promoted from `tmp/refresh_once.py`. |

## Running it

```bash
cd D:\Gits\IsaacLab-GR00T
python agents/rldx-trainbot/harness/monitor/poller.py --outdir tmp/dashboard --interval 120
```

Then open `tmp/dashboard/index.html` in a browser (the file refreshes itself via
`<meta http-equiv=refresh>` every `--interval` seconds — no separate web server, no
`localhost:8787`, just a file on disk you reload or leave open with auto-refresh).

* `--outdir` — where `index.html` and `runs.json` land. Default `tmp/dashboard`.
* `--interval` — seconds between polls **and** the page's self-refresh. Default 90.
* `--once` — poll a single time and exit (useful for a quick check without leaving it running).

Leave it running in one terminal per training session; `Ctrl+C` any time — it only reads,
never writes anything that training depends on. **Don't run it at the same time as another
script that also opens a Pegasus session** (e.g. a manual `pegasus.py` call, or
`launch_training.py`) — logins that fire back-to-back have triggered a temporary lockout before.

## Adding a run

Add a `RunSpec` to the `RUNS` list in `runs.py`:

```python
RunSpec(
    id="my_new_run_30k",             # unique; used as the HTML tab id
    label="Model X · variant Y",     # shown on the tab and panel header
    family="Model X",                # groups tabs under one label (see below)
    model="org/checkpoint-id",
    log="/data/VLA/.../train.log",   # path on the training host
    output="/data/VLA/.../output",   # checkpoint dir, for the "output:" line
    max_steps=30000,
    gpus=1,
    effective_batch=64,
    action_space="13 (right_arm, right_hand)",
    tuning="LoRA r16/a32",
    notes="whatever context future-you will want",
)
```

Nothing in `poller.py` or `render.py` needs to change — both are already model-agnostic. This is
also how the monitor stayed useful across RLDX-1 and GR00T N1.7 without a rewrite.

## Reading the page

* **Overview tab**: one row per run (state, step, loss, estimated finish time) plus a **Box
  state** panel showing what is actually on GPU0/GPU1 right now — training-adjacent processes
  included, so "what's running where" doesn't need a fresh Pegasus login to answer.
* **Family label** (e.g. `RLDX-1`, `GR00T N1.7`): a grouping label above its runs' tabs, **not
  itself clickable** — it has no panel of its own. The buttons after it (bimanual, right_only,
  ...) are the real tabs. Hover it for a tooltip confirming this.
* **Per-run tab**: step/loss/grad-norm/lr, progress bar, elapsed/remaining/estimated-finish
  (from tqdm's own remaining-time estimate — it accounts for the whole run, not just this page's
  polling window), three small charts (loss, grad norm, learning rate — the same small-multiples
  layout W&B uses per run), and the raw log tail.
* **W&B**: not wired up. `WANDB_MODE=disabled` is set for these runs deliberately, to avoid
  needing an API key or network egress approval on the headless box. This page is therefore the
  only live view of them. Say if you'd like W&B enabled instead/also — it needs a key placed on
  Pegasus and is an outward network call, so it's an explicit choice rather than a default.
* Judge whether the page itself is alive by its **refresh timestamp** in the header, not by
  whether the poller's console is still printing — the harness can reap captured stdout while
  the process keeps working.

## Design note: why this lives here and not in each model's agent

Both `poller.py` and `render.py` are intentionally ignorant of "RLDX-1" or "GR00T" — they only
know the `RunSpec` shape. Adding a third model's training run is a data change in `runs.py`, not
a new dashboard. GPU selection and box-wide rules (what counts as idle, how batches split
across GPU counts) are a separate, even-more-general layer — see
`agents/tools/common/README.md`.
