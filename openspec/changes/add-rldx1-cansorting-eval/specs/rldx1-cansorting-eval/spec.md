# rldx1-cansorting-eval

## ADDED Requirements

### Requirement: Success rate is derived from the per-episode results record

Reported success rates SHALL be computed from `simulation_results.csv`, which holds one row per `(env_idx, episode_idx)` with an explicit success flag. Counting rollout video files SHALL NOT be used, because episodes that exhaust the step limit are not recorded to video and the rollout overshoots its per-environment episode target.

#### Scenario: Rate comes from the results record

- **WHEN** a success rate is reported for any run
- **THEN** it is computed from the per-episode rows of `simulation_results.csv`
- **AND** rows sharing an `(env_idx, episode_idx)` key are counted once

#### Scenario: Video-derived counts are rejected

- **WHEN** the number of rollout video files disagrees with the number of result rows
- **THEN** the result rows are authoritative and the discrepancy is reported rather than silently reconciled

#### Scenario: An independent derivation corroborates the number

- **WHEN** a run completes
- **THEN** the reported rate is compared against the per-task success rate the rollout itself prints
- **AND** a disagreement beyond 0.1 percentage points is surfaced as a defect rather than averaged away

### Requirement: Reported aggregates state their weighting

Any aggregate across tasks or suites SHALL name the weighting used. Episode-weighted and task-weighted aggregates differ whenever episode counts differ, and the difference has previously been large enough to invert a verdict.

#### Scenario: Aggregate names its weighting

- **WHEN** an aggregate success rate is reported
- **THEN** the report states whether it is task-weighted or episode-weighted
- **AND** where the two differ, both are shown

### Requirement: RLDX-1 serves the existing can-sorting task through the existing harness

RLDX-1 SHALL be evaluated on `Isaac-Can-Sorting-OpenArm-DexHand-v0` through the same `run_eval.py` path, task definition, episode budget, action filtering and multitask prompt switching as the GR00T runs. Only the wire protocol translation may differ.

#### Scenario: RLDX-1 is selectable as a policy backend

- **WHEN** an evaluation run names the RLDX-1 policy configuration
- **THEN** `run_eval.py` starts an RLDX-1 policy server and drives the task without changes to the task or the client
- **AND** the run appears in the standard comparison output alongside the GR00T runs

#### Scenario: Observation and action contracts are translated

- **WHEN** the client sends an observation
- **THEN** the adapter presents it as RLDX-1 expects: video as `(batch, timestep, height, width, 3)` uint8 and state as float32, keyed by modality
- **AND** returned action chunks are mapped onto the **26-dimensional** bimanual OpenArm action layout — `left_arm` 0:7, `right_arm` 7:14, `left_hand` 14:20, `right_hand` 20:26

#### Scenario: Adapter is validated before closed-loop use

- **WHEN** the adapter is exercised against recorded episodes from the dataset
- **THEN** action shapes and value ranges are confirmed plausible against the recorded actions
- **AND** closed-loop evaluation does not begin until this passes

### Requirement: The comparison is symmetric in tuning method

The comparison SHALL include a GR00T N1.7 LoRA fine-tune trained on the same dataset with the same budget as the RLDX-1 LoRA, so that a difference in outcome is attributable to the model rather than the tuning method. The existing N1.7 full fine-tune SHALL be reported as an upper-bound reference, not as the comparison.

#### Scenario: Three runs are reported together

- **WHEN** the Phase-1 comparison is reported
- **THEN** it contains RLDX-1 LoRA, GR00T N1.7 LoRA, and GR00T N1.7 full fine-tune
- **AND** the two LoRA runs used the same dataset, episode budget and evaluation task

#### Scenario: Asymmetry is labelled where it remains

- **WHEN** the full fine-tune is included in a table with the LoRA runs
- **THEN** it is marked as a different tuning budget rather than presented as a like-for-like comparison

### Requirement: Both fine-tunes train on the same prepared dataset

Training SHALL use `OpenArm_O6_CanSorting_MultiTask_Sim_Dataset_0403` — 2000 episodes, 882,270 frames, 30 fps, **26-dimensional** bimanual state and action, one head camera at 480×640, two task prompts (orange plate / green plate) — with camera count unchanged, so results stay comparable with the existing head-camera GR00T baselines.

The dataset SHALL NOT be upgraded to LeRobot v2.1. RLDX-1's loader performs no `codebase_version` check and reads `meta/stats.json`, which is the v2.0 convention; v2.1 replaces that file with per-episode `episodes_stats.jsonl`, so converting would remove the file the loader asserts on. What the dataset is actually missing is `meta/stats.json` itself.

#### Scenario: Required metadata is present

- **WHEN** the dataset is prepared for training
- **THEN** `meta/` contains `info.json`, `episodes.jsonl`, `tasks.jsonl`, `modality.json` and a generated `meta/stats.json`
- **AND** episode count is 2000, frame count is 882,270, and the state and action layout is 26-dimensional with groups `left_arm` 0:7, `right_arm` 7:14, `left_hand` 14:20, `right_hand` 20:26

#### Scenario: Statistics are generated with upstream's own tool

- **WHEN** `meta/stats.json` is generated
- **THEN** it is produced by `rldx/data/stats.py`, which the loader's assertion names, rather than hand-written
- **AND** it carries `mean`, `std`, `min`, `max`, `q01` and `q99` per field

#### Scenario: Camera configuration is unchanged

- **WHEN** either model is trained for this comparison
- **THEN** it consumes the single head camera present in the dataset
- **AND** no wrist camera is introduced, because the baselines are head-only

### Requirement: Peak GPU memory is measured per component

The evaluation SHALL record peak GPU memory for the policy server and for the simulator separately, so that feasibility on a 24 GB GPU can be determined without repeating the run.

#### Scenario: Per-component peaks are recorded

- **WHEN** an evaluation run completes
- **THEN** peak GPU memory is reported for the policy server alone and the simulator alone, as well as combined
- **AND** the report states whether the combined peak would fit within 24 GB

### Requirement: Throughput is measured before the full budget is committed

Because the evaluation host has no ray-tracing cores, rendering throughput SHALL be measured on a short run before the full three-way sweep is launched.

#### Scenario: Smoke run precedes the full sweep

- **WHEN** a new evaluation host or policy backend is used for the first time
- **THEN** a short run of at most five episodes is completed first
- **AND** measured per-episode wall-clock is used to confirm the full sweep is affordable before it starts

### Requirement: The comparison reports data quality, not just a winner

The Phase-1 report SHALL state, for every run, the episode count actually completed, any task that fell short of its target, server restart counts, and the tuning method — so a reader can judge whether the numbers support the conclusion.

#### Scenario: Report exposes run integrity

- **WHEN** the Phase-1 comparison is published
- **THEN** each run lists completed episodes against the target, any shortfall, and server restart count
- **AND** a run that did not reach its episode target is not presented as comparable without that being stated
