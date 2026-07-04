# VLA → ROS2 JTC → Hardware Control Architecture

> **Document scope:** Complete code-flow analysis of the GR00T VLA inference pipeline, from image/state observation to motor commands, with update-rate analysis, identified side-effects, and recommendations for logging & analysis.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Update Rate Map](#2-update-rate-map)
3. [Detailed Code Flow](#3-detailed-code-flow)
   - [3.1 VLA Inference Loop (gr00t_control_robot.py)](#31-vla-inference-loop)
   - [3.2 Control Loop & JTC Dispatch](#32-control-loop--jtc-dispatch)
   - [3.3 Joint Trajectory Controller (JTC)](#33-joint-trajectory-controller-jtc)
   - [3.4 ros2_control Hardware Interface (v10_simple_hardware)](#34-ros2_control-hardware-interface)
   - [3.5 Motor-Level Threads](#35-motor-level-threads)
4. [Rate Mismatch Analysis & Side-Effects](#4-rate-mismatch-analysis--side-effects)
5. [Recommended Rate Changes](#5-recommended-rate-changes)
6. [Logging Strategy](#6-logging-strategy)
   - [6.1 What to Log](#61-what-to-log)
   - [6.2 Recommended Architecture (Multi-rate Bag + CSV)](#62-recommended-architecture-multi-rate-bag--csv)
   - [6.3 Implementation Plan](#63-implementation-plan)
7. [Data Analysis & Visualization](#7-data-analysis--visualization)
8. [Velocity & Acceleration in JTC Points](#8-velocity--acceleration-in-jtc-points)

---

## 1. System Overview

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              GR00T VLA Control System                                │
│                                                                                      │
│  ┌─────────────┐  camera+state   ┌───────────────┐  action chunk   ┌─────────────┐  │
│  │   Cameras   │────────────────►│  GR00T Server │────────────────►│ inference_  │  │
│  │  (30 fps)   │                 │  (inference)  │   (async)       │   loop()    │  │
│  └─────────────┘                 └───────────────┘                 │  ~variable  │  │
│                                                                     └──────┬──────┘  │
│                                                                            │latest_  │
│                                                                            │action   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │         │
│  │                    control_loop()  @ 30 Hz                           │◄─┘         │
│  │  1. Consume latest_action (anti-stale: drop if >300ms old)           │            │
│  │  2. Compute latency cutoff_step                                       │            │
│  │  3. Apply LPF to action sequence                                      │            │
│  │  4. call send_command() → 4× FollowJointTrajectory goals             │            │
│  └──────────────────────────────┬───────────────────────────────────────┘            │
│                                 │  ROS2 Action (async)                               │
│  ┌──────────────────────────────▼───────────────────────────────────────┐            │
│  │          JointTrajectoryController (JTC)  @ 100 Hz (controller_manager)│          │
│  │  - Receives trajectory (position-only, no vel/acc)                   │            │
│  │  - Interpolates between waypoints at 100 Hz                          │            │
│  │  - Writes pos_commands_[] via CommandInterface                        │            │
│  └──────────────────────────────┬───────────────────────────────────────┘            │
│                                 │  ros2_control Hardware Interface                   │
│  ┌──────────────────────────────▼───────────────────────────────────────┐            │
│  │              v10_simple_hardware  write() @ 100 Hz                   │            │
│  │  - Copies JTC output to arm_pos_cmd_buffer_                          │            │
│  └────┬─────────────────────────────────────────────────────────────────┘            │
│       │                                                                               │
│  ┌────▼───────────────────┐         ┌─────────────────────────────────┐             │
│  │  arm_control_loop()    │         │       state_read_loop()         │             │
│  │    @ 500 Hz            │         │         @ 200 Hz                │             │
│  │  Reads cmd buffer      │         │  CAN recv + LPF(50 Hz cutoff)  │             │
│  │  Gravity + friction    │         │  Updates arm_pos_state_buffer_  │             │
│  │  MIT control → CAN-FD  │         └────────────────────────────────┘             │
│  │  ─────────────────     │                                                          │
│  │  leap_control_loop()   │                                                          │
│  │    @ 500 Hz            │                                                          │
│  └────────────────────────┘                                                          │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐            │
│  │  joint_state_broadcaster  → /joint_states @ hardware update rate     │            │
│  │  gr00t_control_robot.py subscribe_joint_state_loop() @ 30 Hz (spin) │            │
│  └──────────────────────────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Update Rate Map

| Layer | Component | Current Rate | Source / Location |
|-------|-----------|-------------|-------------------|
| **Perception** | Camera capture | 30 fps | `--fps 30` arg |
| **VLA Inference** | GR00T server call | ~variable (network bound) | `inference_loop()` |
| **VLA Observation publish** | `joint_state_callback` spin rate | 30 Hz | `rclpy.spin_once` in `subscribe_joint_state_loop`, interval = `1/control_hz` |
| **Control dispatch** | `control_loop()` sleep | 30 Hz | `time.sleep(1.0 / self.control_hz)` (default `control_hz=30`) |
| **Action chunk step interval** | Time-from-start spacing in trajectory | 1/30 s = 33 ms | `dt = 1.0 / control_hz` in `send_joint_trajectory()` |
| **JTC update (controller_manager)** | JTC interpolation + write | **100 Hz** | `update_rate: 100` in `ros2_controllers.yaml` |
| **Hardware Interface read()** | Called by controller_manager | 100 Hz | Same as `update_rate` |
| **Hardware Interface write()** | Called by controller_manager | 100 Hz | Same as `update_rate` |
| **Motor command (arm)** | `arm_control_loop()` CAN-FD MIT | **500 Hz** | `CONTROL_WRITE_RATE_HZ = 500.0` in header |
| **Motor command (LEAP)** | `leap_control_loop()` serial | **500 Hz** | `const auto loop_period = microseconds(2000)` |
| **State read from motors** | `state_read_loop()` CAN recv | **200 Hz** | `CONTROL_READ_RATE_HZ = 200.0` in header |
| **LPF on arm states** | Inside `state_read_loop()` | 200 Hz | Cutoff = 50 Hz (`STATE_FILTER_CUTOFF_HZ = 50.0`) |
| **JointStateBroadcaster publish** | `/joint_states` topic | 100 Hz | Follows controller_manager update rate |
| **VLA state observation** | Python spin sees `/joint_states` | ≤30 Hz | `spin_once` called at 30 Hz intervals |

---

## 3. Detailed Code Flow

### 3.1 VLA Inference Loop

**File:** [`gr00t_control_robot.py`](../Isaac-GR00T/scripts/sim2real/gr00t_control_robot.py) — `inference_loop()` (line 599)

```
Thread: inference_thread (daemon)
─────────────────────────────────────────────────────────────────
1. Check paused / blocking mode → sleep 10ms if not ready
2. Grab latest camera frames from each camera thread
3. Lock joint_state_lock → copy joint_state dict
   (joint_state comes from /joint_states via joint_state_callback)
4. format_to_gr00t():
   - joints_to_array() for each limb → np.array per group
   - Pack into dict: state.left_arm, state.right_arm, state.left_hand, state.right_hand
   - Pack camera frames as (1, H, W, C) arrays
   - Pack language annotation
5. t_obs = time.time()
6. gr00t_action = policy_client.get_action(gr00t_obs)
   ← BLOCKING RPC call to GR00T server (ZMQ / gRPC)
   ← Returns action_horizon steps of joint targets (e.g. 16 steps × N joints)
7. Store {action, t_obs, t_gen} → self.latest_action (protected by action_lock)
8. Loop immediately (no sleep) → next inference starts right away
```

**Key detail:** The inference thread runs as fast as the server allows (no sleep). The action horizon is typically 16 steps at 30 Hz spacing = 533 ms of planned trajectory per chunk.

---

### 3.2 Control Loop & JTC Dispatch

**File:** [`gr00t_control_robot.py`](../Isaac-GR00T/scripts/sim2real/gr00t_control_robot.py) — `control_loop()` (line 654)

```
Thread: control_thread (daemon), sleeps 1/control_hz = 33 ms
─────────────────────────────────────────────────────────────────
1. Check paused → sleep 10ms if paused
2. Consume self.latest_action (with action_lock)
3. Compute latency = now - t_obs
4. Anti-stale: if latency > 300ms → skip (data too old)
5. cutoff_step = min(int(round(latency / dt)), 15)  ← drop stale steps
6. delay = max(0, latency - cutoff_step * dt)       ← fractional timing alignment
7. For each limb: process_action_seq():
   - Slice seq[cutoff_step:] to skip past steps
   - Apply LowPassFilter (alpha=0.3) across chunk (smoothing)
8. send_command() → for each limb → send_joint_trajectory()
```

**`send_joint_trajectory()`** (line 362):
```python
for i, positions in enumerate(positions_seq):
    point = JointTrajectoryPoint()
    point.positions = positions.tolist()   # ← position ONLY, no velocity/acceleration
    t = (i + 1) * dt + delay
    if self.first_step:
        t += 1.0  # extra 1-second buffer on first step
    point.time_from_start = Duration(t)
    goal_msg.trajectory.points.append(point)

client.send_goal_async(goal_msg)           # ← fire-and-forget (async)
```

**⚠ No velocity or acceleration fields are set in JointTrajectoryPoint.**  
The JTC must infer them via its internal spline interpolation using only position + time_from_start. This degrades trajectory smoothness.

---

### 3.3 Joint Trajectory Controller (JTC)

**Config:** [`ros2_controllers.yaml`](../openarm_ros2/openarm_bimanual_moveit_config/config/ros2_controllers.yaml)

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz — JTC runs at this rate

left_joint_trajectory_controller:
  command_interfaces: [position]        # ← position only
  state_interfaces:  [position, velocity]
```

The JTC operates at **100 Hz**. When it receives a trajectory with:
- 16 waypoints spaced at 33 ms each (30 Hz spacing)
- No velocity/acceleration hints

It performs **cubic spline interpolation** (ros2_control default) to generate smooth intermediate positions at 100 Hz. However, because:
1. The VLA issues a new trajectory every 33 ms
2. JTC may not complete the previous trajectory before the new one arrives
3. JTC **preempts** the current goal when a new goal arrives

This creates a **saw-tooth velocity profile** at the joint level unless the new trajectory starts smoothly from the current position.

---

### 3.4 ros2_control Hardware Interface

**File:** [`v10_simple_hardware.cpp`](../openarm_ros2/openarm_hardware/src/v10_simple_hardware.cpp)

The `read()` and `write()` methods are called synchronously by the controller_manager at **100 Hz**:

```
controller_manager update cycle (100 Hz):
  1. hardware_interface::read()   → copies arm_pos_state_buffer_ → pos_states_[]
                                    (state_read_loop at 200 Hz fills the buffer)
  2. JTC compute()                → interpolates trajectory → updates pos_commands_[]
  3. hardware_interface::write()  → copies pos_commands_[] → arm_pos_cmd_buffer_
                                    (arm_control_loop at 500 Hz reads the buffer)
```

**Double-buffer pattern:**
- `write()` is called at 100 Hz, but the actual motor send happens at **500 Hz** in `arm_control_loop()`
- Between consecutive `write()` calls (10 ms), `arm_control_loop()` sends the **same** command **5 times** (at 2 ms intervals)
- This means the motor sees a command update only every 10 ms (100 Hz effective), but the motor control loop maintains 500 Hz timing for determinism

---

### 3.5 Motor-Level Threads

| Thread | Rate | Description |
|--------|------|-------------|
| `arm_control_loop()` | 500 Hz | Reads `arm_pos_cmd_buffer_`, applies gravity + friction compensation, sends MIT control via CAN-FD |
| `leap_control_loop()` | 500 Hz | Reads `leap_pos_cmd_buffer_`, converts to Dynamixel ticks, sends via RS-485 |
| `state_read_loop()` | 200 Hz | CAN recv + gripper read; applies 50 Hz LPF to positions; updates `arm_pos_state_buffer_` |
| `o6_control_loop()` | 60 Hz | O6 hand CAN control |

**MIT Control parameters** (from `parameters.yaml` / header defaults):
```
arm_params = {kp[i], kd[i], pos_cmd[i], vel_cmd[i]=0, tau_ff[i]=gravity+friction}
```
Note: `vel_cmd` fed to the motor MIT controller is always 0 (from command buffer), even though the hardware exports a velocity interface. This means the motor's internal velocity tracking is open-loop.

---

## 4. Rate Mismatch Analysis & Side-Effects

### 4.1 VLA → JTC: 30 Hz chunk dispatch to 100 Hz controller

| Issue | Detail |
|-------|--------|
| **Trajectory preemption** | A new 16-step trajectory (533 ms) arrives every ~33 ms. JTC is executing step 1–2 before it's replaced. The JTC preempts in-progress trajectories. |
| **Velocity discontinuity** | New trajectory begins at position extrapolated by JTC, not at `current_position`. Without `vel` hints, JTC starts each new chunk at zero velocity, causing a jerk at every 33 ms boundary. |
| **Stale action drop** | `cutoff_step` logic clips leading steps that were "scheduled in the past", but this is position-only — there's no velocity continuity guarantee at the cut point. |
| **Latency accumulation** | Inference latency varies (50–300ms). `cutoff_step` compensates but the number of executed steps shrinks unpredictably. |

### 4.2 JTC → Hardware: 100 Hz → 500 Hz write redundancy

| Issue | Detail |
|-------|--------|
| **Effective motor update = 100 Hz** | `write()` updates the buffer at 100 Hz. The 500 Hz loop just re-sends the same command 5× between updates. From the motor perspective, new setpoints arrive at 100 Hz, not 500 Hz. |
| **Vel_cmd always 0** | MIT controller receives `vel_cmd=0` while the joint is moving. This creates a velocity error that the hardware kd term partially compensates, but it's not optimal. |

### 4.3 State Read → JTC Feedback: 200 Hz → 100 Hz

| Issue | Detail |
|-------|--------|
| **No downsampling mismatch** | The 200 Hz state read feeds a double-buffer. The 100 Hz `read()` snaps the latest value — this is fine. No aliasing, but 50% of state updates are "wasted". |
| **LPF cutoff vs. sample rate** | Hardware LPF: 50 Hz cutoff at 200 Hz sample rate → alpha ≈ 0.61 (mild smoothing). Python LPF on VLA actions: alpha=0.3 (heavy smoothing). |

### 4.4 State Publish → VLA: 100 Hz published → 30 Hz observed

| Issue | Detail |
|-------|--------|
| **70% of state updates unseen by VLA** | `joint_state_callback` is called ~30 times/sec (spin_once at 30 Hz period). The VLA sees joint states that are up to 33 ms old. |
| **State-action temporal alignment** | VLA uses state at `t_obs`. Inference takes 50–200ms. By the time the action is dispatched, the robot state is 50–250 ms ahead of the observed state. This is the main source of **temporal misalignment**. |

---

## 5. Recommended Rate Changes

### Goal: Align rates for smoother, more reactive control

```
Current:  VLA 30Hz → JTC 100Hz → hw_interface 100Hz → motor_write 500Hz ← motor_read 200Hz
Target:   VLA 30Hz → JTC 500Hz → hw_interface 500Hz → motor_write 500Hz ← motor_read 500Hz
```

### 5.1 Increase JTC / controller_manager to 500 Hz

**File:** `ros2_controllers.yaml`
```yaml
controller_manager:
  ros__parameters:
    update_rate: 500  # Hz  ← was 100
```
**Effect:** JTC interpolation runs at 500 Hz, matching the motor command rate. The hardware interface `read()`/`write()` also run at 500 Hz, eliminating the 5× redundant command repetition. The motor now truly gets 500 Hz updates.

**Risk:** CPU load increases 5×. Verify with `htop` after change.

### 5.2 Increase joint_state_broadcaster publish rate

Add to `ros2_controllers.yaml`:
```yaml
joint_state_broadcaster:
  ros__parameters:
    publish_rate: 500.0  # Hz  ← default follows update_rate, explicit override
```

### 5.3 Increase VLA joint state subscription spin rate

**File:** `gr00t_control_robot.py` — `subscribe_joint_state_loop()` (line 749)

```python
# Current:
rclpy.spin_once(self, timeout_sec=1/self.control_hz)  # 33 ms

# Proposed: dedicated high-rate spin
rclpy.spin_once(self, timeout_sec=1/200)  # 5 ms → catches joint states at 200 Hz
```

Or add dedicated publisher in `gr00t_control_robot.py` that queries state at higher rate.

### 5.4 Add velocity & acceleration to JointTrajectoryPoint

See [Section 8](#8-velocity--acceleration-in-jtc-points) for implementation details.

### 5.5 Summary of Rate Changes

| Component | Current | Recommended | File |
|-----------|---------|-------------|------|
| controller_manager update_rate | 100 Hz | **500 Hz** | `ros2_controllers.yaml` |
| joint_state_broadcaster | 100 Hz | **500 Hz** | `ros2_controllers.yaml` |
| Hardware interface read()/write() | 100 Hz | **500 Hz** | (automatic with above) |
| VLA spin_once interval | 33 ms | **5 ms (200 Hz)** | `gr00t_control_robot.py` line 751 |
| JTC trajectory waypoint spacing | 33 ms (30 Hz) | unchanged 33 ms | (VLA cadence is fixed at 30 Hz) |

---

## 6. Logging Strategy

### 6.1 What to Log

| Signal | Source | Rate | Purpose |
|--------|--------|------|---------|
| **VLA action chunk** | `gr00t_control_robot.py` control_loop | 30 Hz (per chunk) | What the VLA wants |
| **JTC setpoint** | JTC `/left_joint_trajectory_controller/controller_state` or custom publisher | 100→500 Hz | What JTC is commanding |
| **Motor command (pos/vel/tau)** | `arm_control_loop()` in hardware driver | 500 Hz | What goes to motors |
| **Motor state (pos/vel/tau raw)** | `state_read_loop()` before LPF | 200 Hz | Raw sensor feedback |
| **Motor state (LPF'd)** | `state_read_loop()` after LPF | 200 Hz | Filtered feedback |
| **Joint states published** | `/joint_states` topic | 100→500 Hz | What ROS2 sees |
| **VLA observed state** | At `t_obs` in `inference_loop()` | ~inference rate | What VLA used |
| **Latency metrics** | `t_obs`, `t_gen`, dispatch time | per chunk | Pipeline timing |

### 6.2 Recommended Architecture (Multi-rate Bag + CSV)

The problem: signals live at 30 Hz, 100 Hz, 200 Hz, and 500 Hz. A single-rate logger wastes bandwidth or misses fast signals.

**Best solution: Two-tier logging**

```
┌─────────────────────────────────────────────────────────────┐
│  Tier 1: ROS2 Bag (rosbag2) — Multi-rate, lossless          │
│                                                              │
│  Topics to record:                                           │
│    /joint_states              (100→500 Hz) ← state feedback │
│    /left_joint_trajectory_controller/controller_state (100→500 Hz) ← JTC setpoints │
│    /right_joint_trajectory_controller/controller_state                              │
│    /vla/action_chunk          (30 Hz, custom topic)         │
│    /vla/diagnostics           (30 Hz, timing + latency)     │
│    /diagnostics               (health monitor)              │
│                                                              │
│  Command: ros2 bag record -o run_YYYYMMDD /joint_states \   │
│    /left_joint_trajectory_controller/controller_state /vla/action_chunk ... │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Tier 2: Hardware CSV (already exists)  — 500 Hz motor-level │
│                                                              │
│  Location: <package_share>/debug_csvs/YYYYMMDD/             │
│  Files: debug_left.csv, debug_right.csv, debug_leap_*.csv   │
│  Content: timestamp, joint_id, pos_cmd, vel_cmd, tau_cmd,   │
│           pos_state, vel_state, tau_state, pos_error,       │
│           gravity_comp, friction_comp, feedforward_tau       │
│  Current sample rate: every 100 iters = 5 Hz (too slow!)   │
│  ← Reduce to every 5 iters = 100 Hz for diagnostics         │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Implementation Plan

#### A. Add VLA diagnostic publisher to `gr00t_control_robot.py`

```python
# In __init__:
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Time as TimeMsg
import rclpy.time

self.action_pub = self.create_publisher(Float64MultiArray, '/vla/action_chunk', 10)
self.diag_pub   = self.create_publisher(Float64MultiArray, '/vla/diagnostics', 10)

# In control_loop(), after processing action:
msg = Float64MultiArray()
msg.data = right_arm_seq.flatten().tolist()  # or all limbs concatenated
self.action_pub.publish(msg)

diag = Float64MultiArray()
diag.data = [t_obs, t_gen, latency, float(cutoff_step), float(len(right_arm_seq))]
self.diag_pub.publish(diag)
```

#### B. Subscribe to JTC state feedback

The JTC publishes `control_msgs/msg/JointTrajectoryControllerState` on:
- `/left_joint_trajectory_controller/controller_state`
- `/right_joint_trajectory_controller/controller_state`

These contain: `reference`, `feedback`, `error`, `output` positions — extremely useful for comparing "what JTC wants" vs "what hardware reports".

```bash
# Record all relevant topics:
ros2 bag record -o vla_run_$(date +%Y%m%d_%H%M%S) \
  /joint_states \
  /left_joint_trajectory_controller/controller_state \
  /right_joint_trajectory_controller/controller_state \
  /vla/action_chunk \
  /vla/diagnostics
```

#### C. Increase hardware CSV logging rate

In `v10_simple_hardware.cpp`, line 1286:
```cpp
// Current: every 100 iterations = 5 Hz
if (csv_sample_count_ % 100 == 0) {

// Change to: every 5 iterations = 100 Hz
if (csv_sample_count_ % 5 == 0) {
```

Or use a time-based gate to avoid miscount when the loop runs at different rates.

#### D. Synchronized timestamp

All data sources must use the **same clock reference**:
- ROS2 topics: use `rclcpp::Clock` (either `RCL_SYSTEM_TIME` or `RCL_ROS_TIME`)
- Hardware CSV: use `steady_clock::now()` and cross-reference with ROS2 `/clock`
- Python: use `self.get_clock().now().nanoseconds` for consistency with ROS2 bag

Add a shared `session_start_ns` logged at startup in both Python and C++ for offset alignment during analysis.

---

## 7. Data Analysis & Visualization

### 7.1 Recommended Tools

| Tool | Use Case |
|------|----------|
| **Foxglove Studio** | Best for multi-rate ROS2 bag playback. Time-synchronized panels for all topics. Free. |
| **PlotJuggler** | Fast interactive joint-by-joint time series. Supports ROS2 bag + CSV. |
| **Python (pandas + matplotlib)** | Custom offline analysis scripts. Best for statistical metrics. |
| **rqt_plot** | Quick live monitoring during experiments (limited to ~10 joints comfortably) |

### 7.2 Suggested Analysis Scripts

**Script 1: `scripts/analysis/plot_pipeline_latency.py`**
- Load `/vla/diagnostics` from bag
- Plot: t_gen - t_obs (inference time), dispatch_time - t_gen (control loop delay)
- Show histogram of inference latencies

**Script 2: `scripts/analysis/plot_joint_tracking.py`**
- Load `/joint_states` (feedback) + `/left_joint_trajectory_controller/controller_state` (JTC reference)
- Per-joint overlay: reference vs feedback vs VLA action
- Compute RMSE tracking error per joint
- Highlight discontinuities at chunk boundaries (every 33 ms)

**Script 3: `scripts/analysis/plot_hardware_csv.py`**
- Load `debug_left.csv` and `debug_right.csv`
- Per-joint subplots: pos_cmd, pos_state, pos_error, gravity_comp, friction_comp
- Identify motor saturation (large torque errors)

### 7.3 Suggested Layout for Multi-Joint Visualization

With 7+7+11=25 joints (or 7+16=23 for LEAP), plotting all in one figure is unwieldy.

**Recommended layout:**
```
Figure 1 — Left Arm (7 joints)
  Subplots: J1–J7, each showing:
    - reference (JTC target, blue)
    - feedback (actual, orange)
    - vla_cmd (step-wise, green)
    - pos_error (red, secondary axis)

Figure 2 — Right Arm (same layout)

Figure 3 — Hands (grid: 4×4 for LEAP or 2×3 for O6)

Figure 4 — Timing/Latency Dashboard
    - Inference latency histogram
    - Control loop timing jitter
    - State staleness (age of joint_state at t_obs)
```

**Key metric to watch:**  
- **Position tracking error** at chunk boundaries (should approach 0 after adding velocity)
- **Velocity continuity** at 33 ms boundaries (current: discontinuous; post-fix: smooth)
- **Inference latency distribution** (target: P99 < 200ms to avoid stale drops)

### 7.4 Foxglove Studio Quick Setup

1. Start `foxglove_bridge` alongside your control launch:
   ```bash
   ros2 launch foxglove_bridge foxglove_bridge_launch.xml
   ```
2. Open `ws://localhost:8765` in Foxglove Studio
3. Add panels:
   - **Plot panel**: `/joint_states/position[0..6]` (left arm)
   - **Plot panel**: `/left_joint_trajectory_controller/controller_state/reference/positions[0..6]`
   - **Plot panel**: `/vla/diagnostics/data` (latency metrics)
   - **3D panel**: robot URDF + joint state for live visualization

---

## 8. Velocity & Acceleration in JTC Points

### Problem

Currently, `send_joint_trajectory()` sends **position-only** `JointTrajectoryPoint`s:

```python
point = JointTrajectoryPoint()
point.positions = positions.tolist()
# point.velocities = []        ← NOT SET
# point.accelerations = []     ← NOT SET
```

The JTC then uses a **cubic spline** (LSPB or quintic, depending on config) to infer velocities, but it assumes **zero velocity at each waypoint** since no hints are given. This causes:
- Velocity discontinuities at chunk boundaries
- Jerky motion between 30 Hz action steps
- Slower convergence to target positions

### Solution: Compute Finite-Difference Velocities

```python
def send_joint_trajectory(self, name, joint_names, positions_seq, control_hz, start_stamp, delay=0.0):
    ...
    dt = 1.0 / control_hz
    n_steps = len(positions_seq)
    
    # Compute velocities via central finite differences
    velocities_seq = []
    for i in range(n_steps):
        if i == 0:
            # Forward difference for first point
            if n_steps > 1:
                v = (positions_seq[1] - positions_seq[0]) / dt
            else:
                v = np.zeros_like(positions_seq[0])
        elif i == n_steps - 1:
            # Backward difference for last point — zero velocity at end of chunk
            v = np.zeros_like(positions_seq[0])
        else:
            # Central difference
            v = (positions_seq[i+1] - positions_seq[i-1]) / (2 * dt)
        velocities_seq.append(v)
    
    for i, (positions, velocities) in enumerate(zip(positions_seq, velocities_seq)):
        point = JointTrajectoryPoint()
        point.positions = positions.tolist()
        point.velocities = velocities.tolist()     # ← ADD THIS
        # Optionally add accelerations (second derivative):
        # point.accelerations = accelerations.tolist()
        t = (i + 1) * dt + delay
        if self.first_step:
            t += 1.0
        point.time_from_start.sec = int(t)
        point.time_from_start.nanosec = int((t % 1.0) * 1e9)
        goal_msg.trajectory.points.append(point)
```

**Why zero velocity at the last waypoint?**  
The VLA issues overlapping chunks. The next chunk's first velocity will be non-zero, providing continuity. Setting the last point to zero creates a brief deceleration — better than a velocity discontinuity.

**Alternative: Use the current joint state as the initial velocity hint**  
```python
# Get current velocity from joint_state or from derivative of last two positions:
if self.joint_state is not None:
    current_vel = self._get_current_velocity()  # Needs state history
    # Prepend a time=0 point at current state:
    first_point = JointTrajectoryPoint()
    first_point.positions = current_positions
    first_point.velocities = current_vel
    first_point.time_from_start = Duration(delay)
    goal_msg.trajectory.points.insert(0, first_point)
```

### Acceleration (Optional but recommended for fast joints)

```python
# Accelerations via second finite difference:
for i in range(n_steps):
    if 0 < i < n_steps - 1:
        a = (positions_seq[i+1] - 2*positions_seq[i] + positions_seq[i-1]) / (dt**2)
    else:
        a = np.zeros_like(positions_seq[0])
    accelerations_seq.append(a)
```

Acceleration hints allow the JTC to use a **quintic spline** instead of cubic, achieving C2 continuity (smooth velocity AND acceleration profiles).

To enable quintic interpolation in JTC config:
```yaml
left_joint_trajectory_controller:
  ros__parameters:
    # ... existing config ...
    interpolation_method: splines  # default, supports vel+acc hints
```

---

## Appendix: Key File Locations

| File | Purpose |
|------|---------|
| [`gr00t_control_robot.py`](../Isaac-GR00T/scripts/sim2real/gr00t_control_robot.py) | Main VLA control node |
| [`ros2_controllers.yaml`](../openarm_ros2/openarm_bimanual_moveit_config/config/ros2_controllers.yaml) | JTC / controller_manager rates |
| [`v10_simple_hardware.cpp`](../openarm_ros2/openarm_hardware/src/v10_simple_hardware.cpp) | Hardware interface + motor threads |
| [`v10_simple_hardware.hpp`](../openarm_ros2/openarm_hardware/include/openarm_hardware/v10_simple_hardware.hpp) | Rate constants (CONTROL_WRITE_RATE_HZ, CONTROL_READ_RATE_HZ) |
| [`filter.py`](../Isaac-GR00T/scripts/sim2real/utils/filter.py) | Python LPF applied to action chunks |
| `<pkg_share>/debug_csvs/YYYYMMDD/debug_*.csv` | Hardware-level motor logs |
