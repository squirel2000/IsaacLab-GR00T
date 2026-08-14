## 1. Local scaffolding

- [x] 1.1 Create `scripts/rl/` with `__init__.py` and a `README.md` runbook stub
- [x] 1.2 Add `scripts/rl/config/phase1.yaml` (our run config: chosen LIBERO suite, default GPU=1, PPO knobs, remote paths, checkpoint id) — values marked TBD where they depend on Open Questions
- [x] 1.3 Add a small remote-exec helper layer in `scripts/rl/remote.py` that wraps `scripts/common/pegasus.py` (`sh`/`run`/`get`) for install/launch/poll/fetch

## 2. Resolve Open Questions against RLinf (verify, don't guess)

- [x] 2.1 Install: `bash requirements/install.sh embodied --model gr00t_n1d7 --env maniskill_libero`; RLinf pinned at commit `43b185d`
- [x] 2.2 Checkpoints: `nvidia/Cosmos-Reason2-2B` (backbone) + `nvidia/GR00T-N1.7-LIBERO` subdir `libero_spatial`
- [x] 2.3 Config `examples/embodiment/config/libero_spatial_ppo_gr00t_n1d7.yaml`: placement `cluster.component_placement`, ckpts under `rollout.model.*` AND `actor.model.*`, budget under `runner.{max_epochs,val_check_interval,save_interval}`; log_path forced by run_embodiment.sh CLI
- [x] 2.4 Resolved values filled into `scripts/rl/config/phase1.yaml`, `launch.py`, and the runbook

## 3. Remote RLinf environment on Pegasus (uv, headless)

- [x] 3.1 Clone RLinf at pinned commit `43b185d` under `/data/VLA/tingying/RLinf` (no submodules)
- [x] 3.2 uv embodied install complete (Python 3.11.14, torch 2.6.0+cu124, LIBERO, ManiSkill, flash-attn)
- [x] 3.3 Headless EGL render confirmed on GPU1 (`MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1` → RENDER_OK 240x320). Teardown EGLError is cosmetic.
- [ ] 3.4 Checkpoints: backbone Cosmos-Reason2-2B is GATED → reuse existing local copy at IsaacLab-GR00T/artifacts/vlm_lora/Cosmos-Reason2-2B-base; GR00T-N1.7-LIBERO/libero_spatial (public) downloading now

## 4. Launcher with cooperative single-GPU selection

- [x] 4.1 `scripts/rl/launch.py`: query `nvidia-smi`, pick an idle GPU (prefer GPU1), abort if none free
- [x] 4.2 Pin single-GPU placement (component_placement = chosen index) + EGL render device; assemble RLinf command via generated derived Hydra config
- [x] 4.3 Start training detached (`setsid nohup`), record remote PID + log path, return immediately

## 5. Smoke test (gate before any long run) — PASSED

- [x] 5.1 Launch a minimal-budget run (max_epochs 3) on GPU1, detached
- [x] 5.2 Confirmed: EGL renders, rollout + PPO actor/critic updates + eval run, `env/success_once` reported, NO OOM — **fits one H100 at ~27 GB (GPU0 idle)**
- [x] 5.3 Workarounds required (all patched into the Pegasus RLinf clone, gated by env vars; see project memory): RLINF_RAY_NUM_CPUS=8 + thread-env caps + dashboard off (cgroup pids.max=2048); in-process vec env `RLINF_LIBERO_INPROCESS=1` (the mujoco env-worker crash was spawn-subprocess-specific); component_placement pinned to GPU1; non-fatal signal_handler list_actors; global_batch_size=16 (divisibility). These belong in the runbook + a clone-patch script.

## 6. Local progress poller

- [x] 6.1 `scripts/rl/poll.py`: periodically fetch the remote log/metrics via `remote.py` and parse step, reward, success-rate
- [x] 6.2 Print a refreshing local progress view; detect and report completion/failure terminal states (tolerant regexes; tighten after smoke shows exact stdout format)

## 7. Phase-1 training run + baseline

- [~] 7.1 Baseline = the run's periodic eval (val_check_interval=10); first eval point is the early reference. NOTE: with total_num_envs=1 the eval is undersampled/noisy, and the base `GR00T-N1.7-LIBERO` ckpt is near-ceiling on libero_spatial, so any "lift" will be weak (expected; flagged to user)
- [x] 7.2 Full PPO run COMPLETE — 100 epochs, checkpoint saved at step 100, GPU1, no errors
- [x] 7.3 Re-ran cleaner (total_num_envs=4, 4 trials/eval). Result = NO lift, slight DECLINE: eval/success_once early-qtr ~0.375 -> late-qtr ~0.125; env/success_once train ~0.43 -> ~0.22. Likely cold-critic (random value head + critic_warmup_steps=0) + minimal/short single-GPU setup + near-ceiling base. Honest, expected-ish.
- [x] 7.4 metrics synced to `artifacts/rl/libero_spatial_ppo_gr00t_n1d7_phase1/metrics.json` (52 scalars; 4-env run)

## 8. HTML analysis report

- [x] 8.1 `scripts/rl/report.py` + `scripts/rl/extract_metrics.py`: self-contained HTML (inline CSS/SVG; no external CDN), TB scalars dumped in-venv → JSON → rendered
- [x] 8.2 Report pipeline validated end-to-end on live run: extract_metrics finds TB events under `<logdir>/tensorboard/` (52 scalars incl. env/success_once, eval/success_once); fixed resumable-fetch clobber
- [x] 8.3 HTML written to `docs/rlinf_gr00t_n17_libero_phase1_report.html` (preliminary; regenerate at run completion for the final trend)

## 9. Verification & close-out

- [x] 9.1 `openspec validate add-rlinf-gr00t-n17-libero-rl --strict` passes
- [x] 9.2 Success criterion — honest outcome: the RL loop is validated end-to-end and, with `critic_warmup_steps=40`, eval success is stable/slightly-positive (~0.50→0.58); WITHOUT warmup it declined. Not a dramatic lift because NVIDIA's base ckpt is near-ceiling on libero_spatial + weak value calibration. Documented in report + README; real lift belongs in Phase 2.
- [x] 9.3 `scripts/rl/README.md` runbook rewritten (real end-to-end steps) + `scripts/rl/patch_rlinf_clone.py` captures the RLinf-clone patches idempotently. All changes uncommitted — commit left to the user.
