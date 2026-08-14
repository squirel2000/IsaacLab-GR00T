## ADDED Requirements

### Requirement: Cooperative single-GPU selection
The launcher SHALL run training on exactly one GPU, never occupying both H100s, and SHALL prefer an unused GPU, defaulting to GPU index 1.

#### Scenario: default to GPU1 when free
- **WHEN** the launcher starts and GPU1 is idle (no significant compute/memory in use)
- **THEN** it pins the run to GPU1 via `CUDA_VISIBLE_DEVICES` and the matching EGL render device

#### Scenario: fall back to another free GPU
- **WHEN** GPU1 is busy but another GPU is idle
- **THEN** the launcher selects the idle GPU and reports which index it chose

#### Scenario: refuse to run when no GPU is free
- **WHEN** all GPUs are busy
- **THEN** the launcher aborts with a clear message and does not start training, so other users are not disrupted

### Requirement: Detached launch surviving disconnects
The launcher SHALL start training as a detached background process on Pegasus so that a dropped Jupyter/WebSocket connection does not terminate the run.

#### Scenario: training survives a closed session
- **WHEN** training is launched and the local control session disconnects
- **THEN** the remote training process keeps running under `nohup` and continues writing to its log file

### Requirement: Local progress polling
The system SHALL display RL improvement and progress on the local terminal by polling the remote training log, without keeping a long-lived blocking connection.

#### Scenario: progress is surfaced locally
- **WHEN** the operator runs the poller while training is active
- **THEN** it fetches the latest remote log/metrics via `scripts/common/pegasus.py` and prints current step, reward trend, and success-rate trend, refreshing on an interval

#### Scenario: poller reports terminal states
- **WHEN** the remote run has finished or died
- **THEN** the poller detects this from the log/exit marker and reports completion or failure rather than polling forever
