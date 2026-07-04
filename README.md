# IsaacLab-GR00T

Workspace (super-repo) for training, evaluating, and orchestrating VLA models
(GR00T N1.x / starVLA) on the OpenArm robot in IsaacLab — as pinned engine repos
driven by role-based agents. Locations are defined in [workspace.yaml](workspace.yaml)
(the single source of truth; scripts resolve paths through it — never hardcode).

## Layout — three planes

- [agents/](agents/) — **control plane**: every action launches from an agent here.
  See [agents/README.md](agents/README.md) for the agent model and the full table.
  Also hosts shared [agents/tools/](agents/tools/) and workspace docs
  [agents/docs/](agents/docs/).
- [engines/](engines/) — **execution plane**: the heavy repos (gitlinks), grouped by
  domain: [vla/](engines/vla/) (`Isaac-GR00T`, `Isaac-GR00T_n1d7`, `starVLA`),
  [vlm/](engines/vlm/) (`Isaac-GR00T-VLM`), [sim/](engines/sim/) (`IsaacLab`),
  [robot/](engines/robot/) (`openarm_ros2`, `openarm_teleop`).
- `datasets/`, `artifacts/` — **data plane**: shared assets across backends.

## Quickstart — one entry point per agent

| Agent | Launch |
|---|---|
| [agentbot](agents/agentbot/) (orchestrator) | `cd agents/agentbot && uv run python -m agentbot.stack up` |
| [vla-trainbot](agents/vla-trainbot/) | `python agents/vla-trainbot/trainbot.py run` |
| [evalbot](agents/evalbot/) | `python agents/evalbot/harness/run_eval.py` |
| [vlm-trainbot](agents/vlm-trainbot/) | see its [runbook](agents/vlm-trainbot/README.md) |
| [rosbot](agents/rosbot/) / [teleopbot](agents/teleopbot/) | see their runbooks |

Each agent directory has a `README.md` (runbook), `charter.md` (role), and
`config.yaml`; runtime outputs land in its `var/` (gitignored).

## Shared storage conventions

```
datasets/<dataset_name>/                    # LeRobot v2 datasets
artifacts/checkpoints/gr00t/<run_name>/     # fine-tuned checkpoints (gr00t | starvla)
artifacts/checkpoints/starvla/<run_name>/
```

Override the roots with `VLA_DATASETS_ROOT` and `VLA_CHECKPOINTS_ROOT`. Engine-internal
demo data, sim assets, and rollout logs stay inside their owning repos.
