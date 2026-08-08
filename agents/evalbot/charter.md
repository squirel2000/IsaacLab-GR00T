# evalbot — charter

## Mission

Closed-loop evaluation of trained VLA checkpoints in IsaacLab: launch the matching
policy server (GR00T N1.x / StarVLA) → run the IsaacSim client to collect episodes →
compute success rates → produce comparison charts and analysis.

## Engines used (resolved via workspace.yaml keys)

- `isaaclab` — the IsaacSim client (`env_isaaclab` conda env)
- `isaac_gr00t` / `isaac_gr00t_n1d7` / `starvla` — per-backend policy servers

## Owns

- `harness/` — `run_eval.py` (single entry), `configs/` (eval plan + per-backend
  specs), `utils/` (client adapters, joint mapping, filters), `analysis/`
  (aggregation, comparison charts), `gr00t_infer_agent.py` (the IsaacLab client)

## Boundaries

- Training belongs to **vla-trainbot**; this agent only consumes checkpoints from
  `artifacts/checkpoints/`.
- **agentbot**'s runtime reuses this harness (`gr00t_infer_agent.py` + configs +
  utils), resolved through its `vla.eval_harness` setting — keep the client
  interface stable or update agentbot in lockstep.
- The IsaacLab repo keeps its own independent `scripts/gr00t_script/gr00t_infer_agent.py`
  (older interface, `--record_episode`; used by its `run_simulation.sh` in
  standalone `$HOME` checkouts, maintained by multiple contributors). It is NOT
  this harness — in this workspace, `harness/gr00t_infer_agent.py` is authoritative
  and the two are intentionally left to evolve separately.

## Notes

- On the RTX 4090 keep `jobs: 1` (one eval ≈ 12–17 GB VRAM); on an H100 2–4 parallel
  jobs are fine.

## Handoffs

- Success rates + charts → `var/analysis/` → reports, model-selection decisions
- Verified checkpoints → **agentbot** checkpoint registry
