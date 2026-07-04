# evalbot — runbook

Evaluates trained VLA checkpoints closed-loop in IsaacLab and compares them. The
harness is crash-resilient: the policy server stays up while the IsaacSim client
relaunches and accumulates episodes until the target count is reached.

## Prerequisites

- `env_isaaclab` conda env (IsaacSim client) and the backend env for the policy
  server (`Isaac-GR00T_n1d7/.venv` for N1.7, `env_gr00t` for N1.5/1.6, `starVLA` env)
- Checkpoints under `artifacts/checkpoints/gr00t/` (or `starvla/`)

## Launch

```bash
python agents/evalbot/harness/run_eval.py               # eval every run in the plan, then compare + chart
python agents/evalbot/harness/run_eval.py --target 50   # fewer episodes per run
```

- The eval plan (which checkpoints, episodes, headless, cameras, video) lives in
  [harness/configs/eval_config.yaml](harness/configs/eval_config.yaml).
- Per-backend server/connection specs: `harness/configs/gr00t_n1?_openarm_o6.json`,
  `starvla_openarm_o6.json` (`server_repo` / `client_pythonpath` are
  workspace-root-relative).
- Re-running with no new checkpoints just re-summarises existing logs.

## Outputs

- Per-run episode logs → `var/eval/logs/` (existing complete logs are reused)
- Success-rate comparison chart → `var/analysis/eval_results.svg`

## GPU budget

Keep `jobs: 1` on the RTX 4090 (each run = 1 IsaacSim + 1 policy server ≈ 12–17 GB).

## Consumers

**agentbot** reuses this harness at runtime (`gr00t_infer_agent.py`, configs, utils)
via its `vla.eval_harness` setting; **vla-trainbot**'s EVAL stage runs this harness on
the remote H100 checkout.
