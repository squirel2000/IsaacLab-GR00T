"""Simulation validation — INTENTIONALLY a stub (run manually for now).

Per the agreed scope, the pipeline stops after delivering the checkpoint to asus-4090;
the user launches sim validation manually there. On asus-4090 the flow is roughly:

    cd /home/asus/Gits/IsaacLab-GR00T
    python launch_isaac_policy.py            # starts GR00T policy server + IsaacLab client
    # the robot repeats the pick-and-place task ~50-100x and accumulates a success rate

To re-enable automation later, re-insert a SIMULATING stage between DEPLOYING and
REPORTING in ``pipeline_state.ORDER`` and implement :func:`run` to SSH into asus-4090,
launch the sim command, parse a ``success_rate: X.XX`` line, and
``state.record_output("sim_result", {...})``. The report already has a slot for it.
"""
from __future__ import annotations


def run(config: dict, state):  # pragma: no cover - not wired into the active flow
    raise NotImplementedError(
        "sim validation is run manually on asus-4090; SIMULATING is not in the pipeline. "
        "See this module's docstring to re-enable it."
    )
