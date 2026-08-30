# `agents/tools/common` — shared mechanisms for the Pegasus H100 box

Anything that talks to the **Pegasus 2×H100 training box** belongs here, not inside a single
model's agent. Every model we train there hits the same machine, the same shared-tenancy
problems and the same transport quirks, so those rules are written once and imported.

| Module | Owns |
| --- | --- |
| `pegasus.py` | The transport: login, `sh()` / `run()` over Jupyter HTTP + WebSocket, file upload/download |
| `h100.py` | GPU selection, launch preflight, checkpoint verification, batch splitting |
| `run_finetune.py` | Legacy one-shot fine-tune driver (kept for reference) |

**Scope**: these rules are about *this shared box*. A local RTX 4090 has one GPU and one user,
so the idle-selection dance does not apply there — but `plan_batch` and `verify_checkpoint` still
do, and `h100.py` imports nothing GPU-count-specific.

## Who uses it

```
agents/tools/common/h100.py                                 <- single implementation
├── agents/vla-trainbot/harness/stages/gpu_monitor.py       <- thin wrapper: adds this
│                                                              pipeline's retry policy, logger,
│                                                              config shape, state recording
├── agents/rldx-trainbot/harness/launch_training.py         <- direct import (training)
└── agents/evalbot/harness/pegasus_launch.py                <- direct import (closed-loop eval —
                                                                see agents/evalbot/harness/
                                                                PEGASUS.md for that workflow)
```

`gpu_monitor.py` re-exports the shared names, so existing imports and
`agents/vla-trainbot/tests/test_gpu_monitor.py` keep working against one implementation.

## The five rules `h100.py` encodes

Each one is here because ignoring it cost us a run.

### 1. The box is shared — ask which card is free, never assume device 0

Colleagues launch jobs from their own checkouts under the same account (one trains GR00T from
`/data/VLA/Isaac-GR00T-n1d7`). Two of our stages died with CUDA OOM because they hardcoded
`CUDA_VISIBLE_DEVICES=0` while another user held **70.4 GB on GPU 0** and GPU 1 sat idle.

```python
import h100
info = h100.preflight(s, need_gb=60)      # raises TimeoutError if nothing fits
gpu = info["gpu"]
```

### 2. "Idle" means utilisation **and** memory are both low

A card at 0 % utilisation can still be holding 70 GB — a model loaded and waiting. Testing
utilisation alone will happily pick it. `pick_idle_gpu` requires both
(`util < 10 %` **and** `used < 5 GB`); `pick_gpu_with_free_memory(gpus, need_gb)` is the stricter
test to use when the run's appetite is known and large.

Preference order lives in **config**, not in the function: `gpu_priority: [1, 0]` (try GPU 1,
fall back to GPU 0). With no priority given, the pure function picks the lowest index.

### 3. Absence of a process is **not** success

An OOM crash and a completed 30 000-step run look identical to a process scan. A chain that
treated "nothing is running" as "the stage finished" once printed `CHAIN_COMPLETE` having
trained nothing at all. Every stage must prove itself by artefact:

```python
if h100.verify_checkpoint(s, output_dir, step=30000) is None:
    raise RuntimeError(h100.last_error(s, log_path))   # name a cause, don't just fail
```

### 4. A process scan must match every entrypoint, and must not match itself

`TRAINING_ENTRYPOINTS` covers `launch_train.py`, `launch_finetune[_asus].py`,
`gr00t_finetune.py` and `torchrun`. An earlier version omitted `launch_finetune.py`, so a
colleague's job was invisible and we launched on top of it.

Separately: `pegasus.sh` passes the whole command to `bash -lc`, so **`pgrep -f <pattern>` matches
the sending shell** when the pattern appears in the command. That has twice reported phantom
jobs and once killed our own shell. Upload a script and run it by path when the pattern is
long, and keep the pattern out of the command text.

### 5. Hold the effective batch fixed; change only the split

Both trainers compute per-device batch as `global_batch // num_gpus`, then apply gradient
accumulation on top — so `effective = global_batch × accum`. Comparisons are only valid when
the effective batch matches across runs, and the split is free to vary to fit the hardware:

```python
global_batch, accum = h100.plan_batch(effective_batch=64, num_gpus=1, per_device_cap=16)
# -> (16, 4)      ; 2 GPUs with cap 32 -> (64, 1)
```

One GPU needs a smaller per-device cap because **DeepSpeed ZeRO only engages with 2+ ranks**
(`experiment.py:180` gates it on `num_gpus > 1 and not use_ddp`), so a single card carries the
whole optimiser state: ~72 GB versus ~36.5 GB/card on two.

**Don't assume a second card buys throughput here.** These are **H100 PCIe with no NVLink**, so
DDP all-reduce cost tracks trainable-parameter count over a comparatively slow interconnect —
which predicts that a LoRA run (RLDX-1: ~26.6 M trainable, ~53 MB bf16 grads/step) should scale
better than a fully-tuned head (N1.7: ~1 GB/step). Measured at fixed effective batch 64, the
opposite happened: RLDX-1 **0.98×** (no gain), N1.7 **1.10×**. At realistic step counts the fixed
per-step synchronisation overhead dominates the payload size, so **neither adaptation style is
worth a second H100 at this batch size** — verify before spending a card, don't reason from
gradient volume alone. Full measurement: `openspec/changes/add-rldx1-cansorting-eval/tasks.md` §6.6.

## Transport quirks `pegasus.py` now handles

* **Jupyter drops large iopub output silently.** The data-rate limit (~1 MB/s) discards stream
  messages and logs the warning on the *server*, so a single `print(r.stdout)` of a few hundred
  KB arrived as an **empty string with exit code 0**. Callers read that as "the command produced
  nothing". `sh()` now emits in paced 16 KB chunks and verifies the received length against a
  server-announced count, raising on a short read. When you only need a summary of something
  large, filter **on the host** anyway (see `monitor/remote_tail.py`).
* **A timeout used to look like success.** The RC sentinel is printed last, so a run that hit the
  deadline returned partial output with `rc=0`. `_exec` now raises `TimeoutError`.
* **A single upload above ~64 MB fails outright.** `PUT /api/contents/<path>` has an
  undocumented server-side ceiling somewhere north of 64 MB — worse for binary files, since
  base64 inflates the body ~33%. Found uploading an IsaacLab tarball, originally hand-patched
  per-caller by chunking client-side and reassembling with `cat` on the server. That's now
  built into `put_file`/`put` themselves (`PUT_CHUNK_SIZE`, 48 MiB): files above the ceiling
  are split, each chunk uploaded with its own retry+relogin, reassembled server-side, and
  size-verified before the temp dir is removed. Every caller gets this for free — nothing to
  opt into.
* **Windows console encoding.** stdout/stderr are reconfigured to UTF-8; remote progress bars
  used to crash the client on cp950.
* **Don't rapid-fire logins.** Back-to-back sessions can trigger a temporary 401 lockout. Reuse
  one session, and pause a running poller before launching something else.
* **The kernel-WebSocket ack for a detached `setsid nohup` launch can stall indefinitely** — the
  job is already running, but the call never returns, so a launcher that waits on the ack looks
  hung and can be killed while its job survives. Fire the launch in a daemon thread and confirm
  it started by polling the job's own status file / log over the Contents API instead of waiting
  on the ack. (Contents-API reads keep working even when kernel creation itself is failing —
  which is also how a disk-full incident stayed diagnosable.)
* **Inlining a script — via `python -c <repr(script)>` or `run --file` — breaks if the script's
  own quoting nests badly inside the `bash -lc "..."` wrapper** (e.g. an apostrophe in a comment),
  and puts the script's whole body on the remote command line, where any `pgrep -f` inside it
  matches the wrapper carrying its own source (rule 4 below). Use **`run_script(s, path)`** /
  `pegasus.py run --script <file>` — it uploads and runs by path, keeping the command line to
  `bash <path>`. Both failure modes disappear rather than needing the escaping to be right.
* **`put()` on a directory is recursive but NOT resumable.** The download direction has
  `download_resumable`; the upload direction has no counterpart, so a re-run re-uploads every
  file. For a large mirror, index the remote side first
  (`find <dir> -printf '%s %P\n'`) and skip files whose size already matches. Single-file
  uploads above ~48 MiB are chunked and size-verified (see above), but that is per-file
  integrity, not whole-directory resume.
* **A PID that shows up in `nvidia-smi --query-compute-apps` can be completely invisible to
  `ps`/`/proc/<pid>` from this session** — `stat /proc/<pid>` returns "No such file or
  directory" (not "Permission denied"), which means a different PID namespace (e.g. a
  colleague's own container on the same shared box), not a permission wall that `sudo` could
  cross. The NVIDIA driver enumerates GPU clients below/outside all containers, so `nvidia-smi`
  still sees the memory and PID; nothing in this session can identify the owning script or user
  beyond that. Don't spend time trying to `sudo`/`ptrace` around it — it's a hard namespace
  boundary, not a bug.

## Known gotchas setting up IsaacLab/IsaacSim on a fresh box

Found once, the expensive way, while getting `env_isaaclab` running on Pegasus. None of these
are Pegasus-specific quirks — they'll recur on any fresh Linux box with no root and an H100/RTX
GPU, so check here before re-diagnosing from scratch.

* **IsaacSim's RTX renderer needs a Vulkan ICD, and a fresh box often hasn't got one.** If
  there's no `nvidia_icd.json` under any standard `/usr/share/vulkan/icd.d`-style path (and that
  directory is typically root-owned, so you can't just add one), Kit's startup table falls back to
  `llvmpipe` — software rendering. NVIDIA's Vulkan implementation is actually baked into
  `libGLX_nvidia.so.0`, which *is* present, so the fix is a hand-written ICD manifest pointing at
  that `.so` with `VK_ICD_FILENAMES`/`VK_DRIVER_FILES` set to it — no root required.
  **Check before applying it:** run `vulkan_icd_probe.py` (via `pegasus.py run --script`), which
  enumerates devices with system ICDs alone and then with the hand-written manifest. As of
  2026-08-30 Pegasus's *system* ICDs already enumerate both H100s, so the workaround is currently
  unnecessary there — it was needed earlier, and the probe is how you tell which situation you're
  in rather than assuming.
* **`isaaclab.sh --install` needs `egl_probe`, which needs EGL/GL dev headers that usually
  aren't there.** `conda install -c conda-forge libglvnd-devel mesalib libglu` (no sudo), then
  install with **`--no-build-isolation`** — otherwise `egl_probe`'s build sees an isolated venv
  that doesn't have the env's own cmake/gcc, and fails anyway even with the headers present.
* **`isaaclab.sh --install` silently pulls an incompatible cu130 torch build**, overwriting
  whatever torch/torchvision was pinned beforehand. Force-reinstall torch/torchvision from the
  cu128 wheel index (`--index-url https://download.pytorch.org/whl/cu128`) **after** the
  installer runs, not just before. The combination that actually works on this box:
  `isaacsim[all,extscache]==5.1.0` (from `https://pypi.nvidia.com`) with `torch==2.7.0` /
  `torchvision==0.22.0` from the cu128 index, in a `conda create -p` env under `/data`.
* **LIBERO's per-project `config.yaml` needs exactly five keys**, or tasks die on
  "`<task>.bddl` does not exist": `assets`, `bddl_files`, `benchmark_root`, `datasets`,
  `init_states` — all pointing into `<checkout>/libero/libero` (with `datasets` at `../datasets`).
  See the isolation rule below for *where* to put that file.
* **Headless Kit blocks on an interactive EULA prompt and dies on EOF** unless
  `OMNI_KIT_ACCEPT_EULA=YES` is set.
* **Conda envs must be created with `-p <path under /data>`, never `-n`** — the default env
  location lands on a nearly-full overlay filesystem on this box.
* **`/tmp` on Pegasus is only ~1 GB, and Kit unpacks its shader caches into it** — a headless
  IsaacSim run dies on a full filesystem unless `TMPDIR`, `TMP`, `TEMP`, `OMNI_CACHE_ROOT` and
  `XDG_CACHE_HOME` are *all* redirected under `/data`. Same root cause as the conda `-p` rule
  above: almost nothing outside `/data` has room on this box.
* **Import name ≠ pip name** for several deps on this stack, and the non-obvious one bites during
  IsaacLab setup: `pink` → **`pin-pink`** (also `cv2` → `opencv-python`, `PIL` → `pillow`,
  `yaml` → `pyyaml`, `sklearn` → `scikit-learn`).
* **LIBERO resolves all its asset/dataset paths from one shared `$HOME/.libero/config.yaml`.**
  Two projects on the same account using different LIBERO checkouts silently clobber each
  other's config through that one file. Give each project an isolated `HOME` override with its
  own `.libero/config.yaml` rather than sharing the real one.
* **Pegasus has no outbound SSH/gh-auth to GitHub.** Git remotes must use HTTPS there — an SSH
  remote just hangs/times out.
* **robosuite 1.4.0's `mujoco>=2.3.0` pin is too loose** — it happily pulls mujoco 3.11.0, which
  renamed `MjData.qM` and breaks robosuite's `mj_fullM` call. Verified working ceiling:
  `mujoco==3.2.7`. Don't assume a newer mujoco is fine just because the version pin allows it —
  test with a real `gym.make/reset/close` cycle, not just import.

## Checklist for the next model

1. Import `h100`; call `preflight(s, need_gb=...)` and let it refuse rather than queue.
2. Compute the batch split with `plan_batch`, keeping the effective batch equal to the runs you
   intend to compare against.
3. Launch detached (`setsid nohup … > log 2>&1 < /dev/null &`) so a dropped connection cannot
   kill training.
4. Gate the next stage on `verify_checkpoint`, and report `last_error` when it fails.
5. Register the run in `agents/rldx-trainbot/harness/monitor/runs.py` so it shows up on the
   monitor page — see that directory's `README.md`.
