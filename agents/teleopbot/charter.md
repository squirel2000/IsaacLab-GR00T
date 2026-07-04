# teleopbot — charter

## Mission

OpenArm teleoperation: a human operator (leader/follower arms or other interfaces)
drives the arm and dexterous hand, producing demonstration episodes for VLA training
(LeRobot v2 datasets).

## Engines used (resolved via workspace.yaml keys)

- `openarm_teleop` — the teleoperation implementation
- `openarm_ros2` — the underlying control chain (shared with rosbot)

## Boundaries

- The control chain's health is **rosbot**'s responsibility; teleopbot operates on
  top of it.
- Dataset post-processing/metadata generation belongs to **vla-trainbot**'s data
  stage (`data_generation.sh`, IsaacLab `gr00t_script` utilities).

## Handoffs

Collected demonstrations → `datasets/` (LeRobot v2 layout) → **vla-trainbot**
fine-tuning.
