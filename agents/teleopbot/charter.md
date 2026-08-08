# teleopbot — charter

## Mission

OpenArm teleoperation across **two input modalities**, producing demonstration
episodes for VLA training (LeRobot v2 datasets):

- **Physical leader-follower** (`openarm_teleop`) — a leader arm drives the follower
  1:1 (bilateral/unilateral modes, gravity compensation).
- **Wearable / XR** (`gallop_teleop`) — MANUS gloves, PICO XR trackers, or UDCAP
  gloves drive the arm (EE-delta) and hand (direct joint angles); no physical leader
  arm required.

## Engines used (resolved via workspace.yaml keys)

- `openarm_teleop` — leader-follower teleop implementation (own record/replay tools)
- `gallop_teleop` — wearable/XR teleop input (glove + tracker drivers only; the
  arm's EE-delta output needs `openarm_ros2`'s IK to become joint commands — see
  Boundaries)
- `openarm_ros2` — the underlying control chain (shared with rosbot); also hosts
  the EE-delta → IK → joint-trajectory bridge that `gallop_teleop`'s arm path relies
  on (`docs/DATASET_COLLECTION_LAYERS.md`, `scripts/gripper/pico_vr_bridge.py` +
  `placo_ik_online_profiler_ws_mesh.py` — topic-compatible with `gallop_teleop`'s
  `/ee_delta/{side}`, not yet confirmed wired together)
- `vla_control` — its `vla_control_tools` package (`data_collector_node.py` +
  `collector_dashboard.py`) is the actual recorder: subscribes `/joint_states`
  (state) + `/joint_actions` (action), opens cameras directly (raw RGB, not a ROS
  topic), and writes LeRobot v2.0 episodes on stop/quit — run alongside whichever
  teleop modality is active

## Boundaries

- The control chain's health is **rosbot**'s responsibility; teleopbot operates on
  top of it.
- `gallop_teleop` is **input-only**: it does not record state/action/camera data and
  has no dataset-collection UI. Recording and its dashboard live in **`vla_control`**
  (see above and its own charter/README).
- Dataset post-processing/metadata generation belongs to **vla-trainbot**'s data
  stage (`data_generation.sh`, IsaacLab `gr00t_script` utilities).

## Handoffs

Collected demonstrations → `datasets/` (LeRobot v2 layout) → **vla-trainbot**
fine-tuning.
