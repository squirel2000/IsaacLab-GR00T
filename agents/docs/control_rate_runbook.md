# Control-Rate Tuning — Apply & Verify Runbook

Operational companion to `vla_to_hardware_architecture.md`. Per phase: **edit → build/apply → relaunch → verify → record → analyze**. Changes are committed by the repo owner.

## Repos & branches
| Repo | Branch | Holds |
|------|--------|-------|
| `openarm_ros2` | `develop` | controllers YAML, hardware C++ (rates, LPF) |
| `Isaac-GR00T` | `feature/control_loop_rate` | `gr00t_control_robot.py` (VLA node), `scripts/analysis/` |

## Make a change take effect
The launch loads the controllers YAML from the **installed share path**
(`get_package_share_directory("openarm_bringup")`, see `openarm_o6_bimanual_lpf.launch.py`),
so source edits need a (re)build/install unless the workspace was built with `--symlink-install`.

| Changed | Required to apply | Why |
|---------|-------------------|-----|
| `v10_simple_hardware.hpp` / `.cpp` (C++) | `colcon build --packages-select openarm_hardware` → re-`source install/setup.bash` | compile-time constants |
| `openarm_v10_o6_bimanual_controllers.yaml` | rebuild/reinstall `openarm_bringup` (or `--symlink-install`) | launch reads installed share copy |
| `gr00t_control_robot.py` (Python) | just rerun the node in `env_gr00t` | interpreted, no build |

Then relaunch:
```bash
ros2 launch openarm_bringup openarm_o6_bimanual_lpf.launch.py \
  right_can_interface:=can2 left_can_interface:=can3 \
  right_o6_can_interface:=can0 left_o6_can_interface:=can1 \
  robot_controller:=joint_trajectory_controller
```

## Sanity checks after relaunch
```bash
htop                                                          # update_rate 100->500 raises CM/read-loop CPU ~5x; confirm headroom
ros2 topic hz /joint_states                                   # expect ~500 Hz (was ~100)
ros2 topic hz /left_joint_trajectory_controller/controller_state   # target 100 Hz; phase3 bag measured ~89 Hz
ros2 topic hz /vla/diagnostics                                # action-chunk/inference cadence (~1.3 Hz in these runs)
```

## Record + analyze (before/after each phase)
```bash
# in Isaac-GR00T (env_gr00t), with the controller stack + VLA node running:
python3 scripts/analysis/record_vla_session.py --prefix <tag>      # Ctrl-C to stop
python3 scripts/analysis/analyze_vla_pipeline.py --bag <tag>_<stamp> --out ./<tag>_analysis
# add --csv <pkg_share>/debug_csvs/<date>/debug_*.csv for motor-level plots
```
The **L key** in `gr00t_control_robot.py` also auto-starts/stops `record_vla_session.py`
(own process group; clean SIGINT finalizes the bag) — disable with `--no-rosbag`,
relocate with `--rosbag_dir`. The 500 Hz motor CSV is written separately by the hardware
component on controller deactivate.

**Compare two sessions** (baseline vs optimized — the report figures):
```bash
python3 scripts/analysis/compare_vla_sessions.py \
  --a <baseline_dir> --a-label "baseline (100Hz)" \
  --b <optimized_dir> --b-label "phase0-3 (500Hz)" --out <out_dir>
# → session_comparison.png (timing/jitter/accel/RMSE) + command_chain_comparison.png
#   (500 Hz pos_cmd staircase vs cubic, + velocity PSD) + a printed metrics table.
```
Rate/jitter/command-shape metrics compare cleanly across different motions; absolute
tracking RMSE is motion-dependent — for a strict accuracy A/B record the **same scripted
trajectory** under both configs.

## Phase 0 status (analysis tooling & diagnostics)
| Task | What | Where |
|------|------|-------|
| T0.1 | `analyze_vla_pipeline.py` — latency dashboard, JTC tracking (actual=red, reference=blue, error=green in rad), hardware-CSV, velocity-continuity (arm joint count is data-driven, not hardcoded to 7) | `Isaac-GR00T/scripts/analysis/` |
| T0.2 | `/vla/diagnostics` = `[t_obs, t_gen, latency, cutoff_step, n_steps]` (Float64) + `/vla/action_chunk` (all active limbs) | `gr00t_control_robot.py` |
| T0.3–T0.5 | consolidated as functions inside `analyze_vla_pipeline.py` (no separate `plot_*.py`) | — |
| T0.6 | `ros2 bag` via `record_vla_session.py` (also auto-triggered by the L key) | run on hardware |
| T0.7 | `compare_vla_sessions.py` — two-session diff → `session_comparison.png` + `command_chain_comparison.png` + metrics table | `Isaac-GR00T/scripts/analysis/` |

**Environment:** run all analysis with the **`env_gr00t`** conda env (Python 3.10,
has numpy/pandas/matplotlib/scipy/rosbags) — the system `python3` lacks the stack:
```bash
/home/asus/miniforge3/envs/env_gr00t/bin/python scripts/analysis/analyze_vla_pipeline.py ...
```

**Docs report:** `vla_to_hardware_architecture.html` is now a **standalone single-file
report** — architecture is HTML tables/cards, and the two `compare_vla_sessions.py`
result figures are embedded directly as base64. The old `docs/images/generate_figures.py`
+ `docs/embed_images.py` workflow is no longer used (the matplotlib architecture diagrams
were dropped); those files are now orphaned and can be deleted.

## Phase 1 status (applied 2026-05-25, batched for net effect)
| Task | Change | File |
|------|--------|------|
| T1.1 | `update_rate` 100→500 | controllers YAML |
| T1.2 | `CONTROL_READ_RATE_HZ` 200→500 | `v10_simple_hardware.hpp` |
| T1.3 | `STATE_FILTER_CUTOFF_HZ` 50→100 | `v10_simple_hardware.hpp` |
| T1.4 | `state_publish_rate` 50→100 (×4) | controllers YAML |
| T1.5 | joint-state spin `1/control_hz`→`1/200` | `gr00t_control_robot.py` |

Because T1.1–T1.5 landed together, before/after compares Phase-0 baseline → all of Phase 1 (net effect, not per-change attribution).

## Phase 2 status (applied 2026-05-25)
| Task | Change | File |
|------|--------|------|
| T2.1 | finite-difference velocities in `send_joint_trajectory()` → JTC cubic-spline (C1) instead of linear | `gr00t_control_robot.py` |
| T2.2 | record + analyze the velocity-continuity panel before/after | run on hardware |
| T2.3 | **skipped** — quintic/acceleration amplifies VLA action noise; cubic (velocity-only) is the better trade-off | — |
| T2.4 | `cmd_filter_cutoff_hz` left at 10 Hz; it is a **launch argument**, so isolate T2.1 first, then try `cmd_filter_cutoff_hz:=15` (or `20`) at launch | no code change |

To experiment with the command-side LPF without rebuilding:
```bash
ros2 launch openarm_bringup openarm_o6_bimanual_lpf.launch.py ... cmd_filter_cutoff_hz:=15
```

## Phase 3 status (applied 2026-05-25)
| Task | Change | File |
|------|--------|------|
| T3.1 | in-memory ring buffer (`ArmDebugSample`) records every `arm_control_loop` iteration (500 Hz) with **zero file I/O** | `v10_simple_hardware.{hpp,cpp}` |
| T3.2 | batch flush to CSV on deactivate (`flush_debug_ring_to_csv()`, after the arm thread is joined) | `v10_simple_hardware.cpp` |
| T3.3 | **skipped** — optional on-demand ROS2 dump service (awkward to host on a hardware component; not needed for the core win) | — |
| T3.4 | **no-op** — CSV columns are unchanged, so `analyze_vla_pipeline.py --csv` works as-is | — |

Notes / operational limits:
- **Ring buffer** keeps only the most recent `DEBUG_RING_SECONDS` (default **120 s**) → ~31 MB/arm pre-allocated (≈62 MB for both). A longer session keeps only the **last 120 s**; raise `DEBUG_RING_SECONDS` in the hpp (memory scales linearly) for longer captures.
- **Flush happens only on a clean `on_deactivate()`** (controller_manager deactivate / clean `ros2_control_node` shutdown). A `kill -9`, crash, or power loss writes **nothing** — confirm the `"Flushed N debug samples…"` log on stop. **So: record episodes ≤120 s and stop cleanly.**
- A crash-proof / unbounded logger would need a dedicated background flush thread (deferred by choice; the current design keeps the 500 Hz loop I/O-free).
- Same output path as before: `<install>/openarm_hardware/share/openarm_hardware/debug_csvs/YYYYMMDD/debug_<arm>_<stamp>.csv`, written once on deactivate.
- Requires a C++ rebuild: `colcon build --packages-select openarm_hardware`.

## Measured results (baseline 100 Hz → phase 0–3 500 Hz, 2026-05-25)
Two hardware sessions (`artifacts/outputs/vla_record_original` vs `vla_record_p3`),
compared with `compare_vla_sessions.py`. Different motions, so read by category:

| Metric | baseline | phase0-3 | comparable? |
|--------|----------|----------|-------------|
| `/joint_states` rate | 100 Hz | **500 Hz** | yes (config) |
| `/joint_states` Δt p99 / max | 10.4 / 12.9 ms | **2.2 / 4.5 ms** | yes |
| JTC `controller_state` rate | 38 Hz | **89 Hz** | yes (target 50→100; ~11% realtime-publisher skips) |
| JTC `controller_state` Δt std | 4.85 ms | **0.97 ms** | yes |
| motor setpoint change rate | 78 Hz | **395 Hz** | yes — baseline = 5-step staircase (100→500), cubic ≈ per-cycle |
| commanded `\|accel\|` p99 | 278 rad/s² | **11.4 rad/s²** | yes — ~24× fewer spikes (Phase 2) |
| cmd-velocity energy >40 Hz | ~67% | ~24% | yes — 100 Hz staircase peak removed |
| tracking RMSE (active window) | ~10–12 mrad | ~10–12 mrad | motion-dependent → parity, no regression |
| VLA inference latency (mean) | 151 ms | 145 ms | GPU-bound, not a target |

**Why control "feels" the same:** motion is VLA-paced (new chunk ~1/s, dispatched at
30 Hz) and the motor PD loop was already 500 Hz in both — so gross tracking of slow
motion is unchanged. The wins are timing regularity + command smoothness. **Phase 3 is
invisible by design** (both CSVs are identical 500 Hz × 120 s; it only moves logging I/O
out of the hot loop). htop: load avg ~1.4→2.0 on a many-core box = not overloaded; the
phase-3 CSV shows the 500 Hz loop holds 2.000 ± 0.013 ms (max 3 ms) = no I/O stalls.

Full write-up with both figures: `docs/vla_to_hardware_architecture.html` (Results section).
