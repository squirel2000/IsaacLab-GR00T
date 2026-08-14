## ADDED Requirements

### Requirement: Headless RLinf environment on Pegasus
The system SHALL provision an RLinf embodied environment on the Pegasus Jupyter host using `uv` (not Docker), configured for headless GPU rendering of LIBERO via NVIDIA EGL.

#### Scenario: uv-based install completes without Docker
- **WHEN** the setup runbook is executed on Pegasus
- **THEN** RLinf is cloned at a pinned commit and its embodied dependencies (LIBERO/robosuite/MuJoCo) are installed into a `uv`-managed Python 3.11 environment without requiring `docker`

#### Scenario: headless rendering is configured
- **WHEN** an RLinf rollout renders LIBERO camera observations on Pegasus
- **THEN** `MUJOCO_GL=egl` is set and rendering uses the NVIDIA EGL device for the selected GPU, producing image observations with no X display present

### Requirement: GR00T-N1.7 PPO training bootstrapped from the NVIDIA LIBERO checkpoint
The system SHALL run RLinf PPO (actor-critic, flow-matching action head) on GR00T-N1.7, initialized from NVIDIA's official `GR00T-N1.7-LIBERO` checkpoint, on exactly one LIBERO suite.

#### Scenario: training starts from the official checkpoint
- **WHEN** a training run is launched
- **THEN** the policy weights are initialized from the downloaded NVIDIA `GR00T-N1.7-LIBERO` checkpoint and the configured LIBERO suite (e.g. Goal or Spatial) is the only suite active

#### Scenario: PPO updates execute end-to-end
- **WHEN** the training loop runs
- **THEN** rollout collection, advantage estimation, and at least one PPO optimizer update complete without error, and per-step metrics (reward, success rate, KL, loss) are written to a log readable from local

### Requirement: Single-GPU memory fit verified by a smoke test
The system SHALL verify that the full GR00T-N1.7 + PPO loop fits on a single 80 GB H100 via a short smoke test before any long run, and SHALL provide documented fallbacks if it does not fit.

#### Scenario: smoke test confirms the loop fits one GPU
- **WHEN** the smoke test runs with a minimal step/rollout budget on one GPU
- **THEN** it completes at least one PPO update without out-of-memory, and the peak GPU memory is recorded

#### Scenario: out-of-memory triggers documented fallback
- **WHEN** the smoke test reports an out-of-memory error
- **THEN** the run halts and the runbook directs the operator to reduce batch size, rollout length, or parallel-env count (and/or enable gradient checkpointing / FSDP offload) before retrying

### Requirement: Measurable success-rate lift on one LIBERO suite
RL training SHALL produce a clearly measurable improvement in task success rate over the base checkpoint on the chosen LIBERO suite; the change MUST NOT be considered complete otherwise.

#### Scenario: post-RL success rate exceeds the baseline
- **WHEN** the base `GR00T-N1.7-LIBERO` checkpoint and the post-RL checkpoint are both evaluated on the chosen suite under identical settings
- **THEN** the post-RL success rate exceeds the baseline by a margin larger than evaluation noise, and both numbers are recorded for the report
