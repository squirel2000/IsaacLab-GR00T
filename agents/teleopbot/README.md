# teleopbot — runbook

Teleoperates the OpenArm (arm + dexterous hand) to collect demonstration episodes for
VLA fine-tuning.

## Prerequisites

- **rosbot** bring-up completed (control chain healthy — see
  [agents/rosbot/README.md](../rosbot/README.md))
- `engines/robot/openarm_teleop` built per its own README

## Operate

Follow [engines/robot/openarm_teleop/README.md](../../engines/robot/openarm_teleop/README.md)
for the authoritative device setup and start commands. Keyboard-teleop utilities for
the sim also exist in `engines/sim/IsaacLab/scripts/gr00t_script/` (e.g.
`teleop_keyboard_agent_2hands.py`).

## Output convention

Demonstrations land in `datasets/<dataset_name>/` in LeRobot v2 layout
(`data/chunk-*/episode_*.parquet`, `videos/chunk-*/…/episode_*.mp4`, `meta/`).
Generate GR00T metadata (`info.json`, `episodes.jsonl`) with
`engines/sim/IsaacLab/scripts/gr00t_script/utils/generate_dataset_meta.py`
(run from the IsaacLab dir with `--robot_type … --dataset_root …`).

## Handoffs

New datasets → **vla-trainbot** (fine-tune) and **vlm-trainbot** (VQA generation
uses the same LeRobot videos).
