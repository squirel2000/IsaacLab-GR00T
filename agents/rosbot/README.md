# rosbot — runbook

Brings up the OpenArm ROS2 stack (simulation or real robot), RViz visualization, and
the JointTrajectoryController chain.

## Prerequisites

- A sourced ROS2 environment and a built colcon workspace containing
  `engines/robot/openarm_ros2` (see that repo's README for the authoritative build
  and launch instructions).

## Launch

Follow [engines/robot/openarm_ros2/README.md](../../engines/robot/openarm_ros2/README.md)
— typical flow is `colcon build` then `ros2 launch openarm_… <launch-file>` for
sim/RViz or the real arm. This runbook intentionally defers command details to the
engine repo so they cannot drift.

## Health checks

- `ros2 topic echo /joint_states` streams at the expected rate
- Controller states: `ros2 control list_controllers` shows the
  JointTrajectoryController active
- RViz displays the arm model tracking commanded motion

## Handoffs

A healthy control chain is the prerequisite for **teleopbot** (data collection) and
for **agentbot**'s future hardware backend (`vla/backends/hardware.py`, which bridges
VLA action chunks → JointTrajectoryController via
`engines/vla/Isaac-GR00T/scripts/sim2real/gr00t_control_robot.py`).
