# rldx1-integration

## ADDED Requirements

### Requirement: RLDX-1 is a tracked engine with a workspace key

The repository SHALL track RLDX-1 at `engines/vla/RLDX-1`, pinned to the commit verified in Phase 0, and SHALL expose it through `workspace.yaml` and `paths.env` like every other engine. No script may hardcode the path.

It SHALL be tracked the same way as the existing engines: as a gitlink (index mode `160000`) with **no** `.gitmodules` entry. This repo tracks all nine of its engines that way; introducing a `.gitmodules` entry for one engine alone would make the tree inconsistent.

#### Scenario: Path resolves through the workspace root marker

- **WHEN** a script sources `paths.env` from anywhere inside the workspace
- **THEN** `WS_RLDX1` resolves to the absolute path of `engines/vla/RLDX-1`
- **AND** `workspace.yaml` contains a `rldx1:` key whose value is the repo-relative path

#### Scenario: Engine is pinned to the verified commit

- **WHEN** the engine directory is inspected
- **THEN** `git -C engines/vla/RLDX-1 rev-parse HEAD` reports `cf67c31`, the commit Phase 0 was verified against
- **AND** the entry appears in the index with mode `160000`, matching the other engines

### Requirement: The LIBERO evaluation environment is repairable by one documented command

Upstream's `setup_libero.sh` reports success while leaving an environment in which the documented rollout client cannot start. The repository SHALL provide an idempotent repair script that makes the environment executable, and SHALL document, per fix, the failure it prevents.

#### Scenario: Repair script makes the rollout client importable

- **WHEN** the repair script is run against a freshly built LIBERO sim venv
- **THEN** `rldx/eval/rollout_policy.py --help` exits successfully instead of raising `ModuleNotFoundError`
- **AND** the script is safe to re-run with no further effect

#### Scenario: Each fix names the defect it addresses

- **WHEN** a reader opens the repair script
- **THEN** it states for each of the five fixes the upstream cause and the exact error string that appears without it
- **AND** the required `mujoco` ceiling is recorded as `3.2.7` with the reason `MjData.qM` was removed upstream

#### Scenario: Server readiness does not depend on unavailable tooling

- **WHEN** the harness waits for a policy server to accept connections on a host without `iproute2` or `net-tools`
- **THEN** readiness is determined by reading the kernel socket table for a listening socket
- **AND** the harness does not report the server as dead while it is in fact listening

#### Scenario: LIBERO asset paths do not collide with other projects

- **WHEN** the harness runs on a machine where another project has written a `~/.libero/config.yaml`
- **THEN** the harness directs LIBERO at its own configuration via `LIBERO_CONFIG_PATH`
- **AND** the shared configuration file is left unmodified

### Requirement: Evaluation runs survive interruption without redoing finished work

The evaluation supervisor SHALL record per-task completion, resume by re-running only unfinished tasks, keep the policy server alive across failures, and release GPU memory when it finishes.

#### Scenario: Restart re-runs only unfinished tasks

- **WHEN** the supervisor is re-invoked after some tasks have completed
- **THEN** it skips every task with a completion marker
- **AND** it does not load the policy checkpoint at all if no task remains

#### Scenario: Policy server death does not abort the sweep

- **WHEN** the policy server exits while tasks remain
- **THEN** the supervisor restarts it, up to a bounded number of attempts, and continues
- **AND** it reports the restart count in its status output

#### Scenario: Supervisor releases the GPU when the sweep ends

- **WHEN** the last task completes
- **THEN** the supervisor terminates the policy server, writes a terminal status, and exits
- **AND** it does not block waiting on the server process it launched itself

#### Scenario: Configured parallelism is honoured

- **WHEN** the supervisor is configured to run N rollout tasks concurrently
- **THEN** N rollout tasks run concurrently
- **AND** the policy server job does not consume one of the N slots

### Requirement: Non-commercial weights are kept out of the product checkpoint path

RLDX-1 weights and every checkpoint derived from them carry a non-commercial, share-alike licence. The repository SHALL store them only under a research-labelled path and SHALL NOT place them in the checkpoint directory used for product work.

#### Scenario: Derived checkpoints land under the research path

- **WHEN** an RLDX-1 fine-tune produces a checkpoint
- **THEN** it is written under `artifacts/rldx1/`
- **AND** nothing derived from RLDX-1 appears under `artifacts/checkpoints/gr00t/`

#### Scenario: The licence constraint is discoverable

- **WHEN** a team member reads the agent charter for this work
- **THEN** it states that the weights are non-commercial with share-alike, that derivatives inherit those terms, and that the work is a research comparison only

### Requirement: The Phase-0 result is reproducible from the repository

The promoted harness SHALL reproduce the Phase-0 LIBERO figure from a repository checkout, so that the number is verifiable rather than anecdotal.

#### Scenario: Tally reproduces the recorded result

- **WHEN** the promoted tally is run against the Phase-0 output directory
- **THEN** it reports a suite mean of 97.47% over 1100 episodes
- **AND** it reports per-suite rates of 97.50 / 99.00 / 99.00 / 94.40 for spatial / object / goal / long
