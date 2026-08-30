# agents/ — the control plane (one agent per role)

Every action in this workspace is launched by an agent in this directory. An agent is a
role: a `README.md` (runbook), a `charter.md` (mission and boundaries), a `config.yaml`
(machine-readable settings), an optional `harness/` (the code it executes), and a `var/`
(runtime outputs, gitignored). Heavy repos are **engines** (`../engines/`, shared by
multiple agents) — agents reference them through the root [`workspace.yaml`](../workspace.yaml)
keys, never hardcoded paths.

| Agent | Role | Shape |
|---|---|---|
| [`agentbot/`](agentbot/) | **Orchestrator** — Brain→Skill→VLA execution; wakes/monitors the whole sim/real stack via `python -m agentbot.stack` | executable package (own repo) |
| [`vla-trainbot/`](vla-trainbot/) | VLA model training (GR00T N1.x / starVLA) — automated H100 train→eval→deploy | charter + harness |
| [`vlm-trainbot/`](vlm-trainbot/) | VLM training (Cosmos-R2 VQA LoRA → merge → swap) | charter + runbook |
| [`rldx-trainbot/`](rldx-trainbot/) | Research evaluation of RLWRLD RLDX-1 vs the GR00T N1.7 baseline (**non-commercial weights** — see its charter) | charter + harness |
| [`evalbot/`](evalbot/) | Post-training closed-loop evaluation in IsaacLab | charter + harness |
| [`rosbot/`](rosbot/) | OpenArm ROS2 bring-up (sim / RViz / controllers) | charter + runbook |
| [`teleopbot/`](teleopbot/) | Teleoperation + demonstration data collection | charter + runbook |

Shared here as well: [`tools/`](tools/) (cross-agent utilities: `workspace_paths.py`,
`common/pegasus.py`, `common/run_finetune.py`, `data_downloader/`) and [`docs/`](docs/)
(workspace-level documentation).

## Upgrade paths (per agent, when needed)

1. **agentbot-managed service** — turn an agent into a process the orchestrator wakes and
   monitors (like the stack supervisor manages the VLM/GR00T/sim services); events go on
   the redis bus, status shows on the dashboard. No LLM involved.
2. **Claude Code subagent** — drop a definition in `.claude/agents/<name>.md` (system
   prompt ≈ the charter) so an LLM operator can be dispatched to drive the harness.

The charters are written to double as the spec for both forms.
