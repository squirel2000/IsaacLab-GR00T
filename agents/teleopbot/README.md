# teleopbot — runbook

Teleoperates the OpenArm (arm + dexterous hand) to collect demonstration episodes for
VLA fine-tuning, via either of two input modalities.

## Prerequisites

- **rosbot** bring-up completed (control chain healthy — see
  [agents/rosbot/README.md](../rosbot/README.md))
- One of the two engines built per its own README:
  `engines/robot/openarm_teleop/` (leader-follower) or
  `engines/robot/gallop_teleop/` (wearable/XR)

## Operate

**Leader-follower** (physical leader arm): follow
[engines/robot/openarm_teleop/README.md](../../engines/robot/openarm_teleop/README.md)
for device setup and the `script/launch_*.sh` variants (bilateral/unilateral,
gravity comp, with/without LeapHand or O6 hand).

**Wearable/XR** (MANUS glove / PICO tracker / UDCAP glove — no leader arm): follow
[engines/robot/gallop_teleop/README.md](../../engines/robot/gallop_teleop/README.md).
Build once (`colcon build`), then:

```bash
ros2 launch gallop_bringup teleop.launch.py       # PICO arm + MANUS hand (default combo)
ros2 service call /set_teleop_arm_enabled  std_srvs/srv/SetBool "{data: true}"
ros2 service call /set_teleop_hand_enabled std_srvs/srv/SetBool "{data: true}"
```

`gallop_teleop` only publishes the hand's final joint angles
(`forward_position_controller`) and the arm's relative EE-pose deltas
(`/ee_delta/{side}`) — the arm needs `openarm_ros2`'s IK bridge
(`placo_ik_online_profiler_ws_mesh.py`) subscribed to turn deltas into joint
commands; it does not record data or provide a collection UI itself.

Keyboard-teleop utilities for the sim also exist in
`engines/sim/IsaacLab/scripts/gr00t_script/` (e.g. `teleop_keyboard_agent_2hands.py`).

## Recording demonstrations

State/action/camera recording and its dashboard are **not** part of either teleop
engine — they live in `vla_control`'s `vla_control_tools` package. Run it alongside
whichever teleop modality is active:

```bash
cd engines/vla/vla_control
colcon build --packages-select vla_control_tools && source install/setup.bash
ros2 run vla_control_tools data_collector \
    --pov_list head wrist_L wrist_R --dataset_dir <path> \
    --robot_type openarm_linkerhand_o6 --task_list "pick the can" "place on the plate"
```

Opens a PySide6 dashboard (camera preview, stats, trajectory plot, dataset browse);
keyboard shortcuts **O** start/stop, **R** discard + reset pose, **T** toggle teleop,
**M** next task, **Q** quit. It subscribes `/joint_states` (state) and
`/joint_actions` (action), and opens cameras directly (raw RGB frames via
`CameraFactory` — ZED SDK or OpenCV, not a ROS topic). Writes are event-triggered
(on stop / quit), not on a fixed wall-clock schedule — see
[engines/vla/vla_control/src/vla_control_tools/README.md](../../engines/vla/vla_control/src/vla_control_tools/README.md)
for full CLI/dashboard details, and
[agents/docs/vla_control_data_collector_flow.png](../docs/vla_control_data_collector_flow.png)
(editable: `.drawio` alongside it) for the full pipeline diagram.

## Output convention

Demonstrations land in `datasets/<dataset_name>/` in LeRobot v2 layout
(`data/chunk-*/episode_*.parquet`, `videos/chunk-*/…/episode_*.mp4`, `meta/`).
Generate GR00T metadata (`info.json`, `episodes.jsonl`) with
`engines/sim/IsaacLab/scripts/gr00t_script/utils/generate_dataset_meta.py`
(run from the IsaacLab dir with `--robot_type … --dataset_root …`).

## Handoffs

New datasets → **vla-trainbot** (fine-tune) and **vlm-trainbot** (VQA generation
uses the same LeRobot videos).
