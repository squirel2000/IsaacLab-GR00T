# rosbot — charter

## Mission

The ROS2 side of OpenArm: sim/real bring-up, RViz visualization, the
JointTrajectoryController control chain, and health checks on `/joint_states` and
diagnostics topics. This agent is the hardware-side counterpart of the sim→hardware
path (agentbot's `vla/backends/hardware.py`).

## Engines used (resolved via workspace.yaml keys)

- `openarm_ros2` — OpenArm's ROS2 packages (description / controllers / launch)

## Boundaries

- Launch commands and package details follow the engine repo's own README — this
  charter pins the role, not the commands.
- Teleoperation is **teleopbot**'s role; rosbot guarantees the control chain
  underneath it.

## Handoffs

- Provides the control interface (JointTrajectoryController) that **agentbot**'s
  hardware backend will drive.
- Supports **teleopbot** during real-robot data collection (healthy control chain).
