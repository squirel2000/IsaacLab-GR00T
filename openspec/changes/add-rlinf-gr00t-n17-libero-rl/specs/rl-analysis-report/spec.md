## ADDED Requirements

### Requirement: Complete standalone HTML analysis report
The system SHALL generate a single self-contained HTML report summarizing the RL run, consistent with the existing `docs/*.html` report style in the repository.

#### Scenario: report is generated after a run
- **WHEN** the report generator is run against a completed (or in-progress) RL run's metrics and logs
- **THEN** it writes one standalone HTML file under `docs/` that opens offline with no external network dependencies

#### Scenario: report contains the required sections
- **WHEN** the HTML report is opened
- **THEN** it shows: run configuration (suite, GPU, checkpoint, PPO hyperparameters), the before/after success rate, training curves (reward, success rate, KL/loss), GPU/resource usage, and a clear pass/fail verdict against the success criterion

#### Scenario: report states verdict honestly
- **WHEN** the post-RL success rate does not exceed the baseline beyond noise
- **THEN** the report's verdict is reported as NOT met (no overstated success), including the observed numbers
