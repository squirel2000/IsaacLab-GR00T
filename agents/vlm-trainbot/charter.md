# vlm-trainbot — charter

## Mission

Fine-tune the VLM backbone of GR00T N1.7 (Cosmos-Reason2-2B) on auto-generated VQA:
generate VQA from a LeRobot dataset → LoRA fine-tune → merge into a standalone VLM
(**Product A**, deployable as agentbot's Brain endpoint) → swap back into the VLA
(**Product B**) → evaluate accuracy by question type.

## Engines used (resolved via workspace.yaml keys)

- `isaac_gr00t_vlm` — the whole flow lives there (`src/vlm_lora/*`; settings
  centralized in `configs/default.yaml`)
- `isaac_gr00t_n1d7` — swap target (baseline VLA checkpoint)

## Owns

- The runbook (README.md) that drives the engine's 5-step flow. A thin
  `harness/run_vqa_flow.py` driver is planned (see
  `engines/vlm/Isaac-GR00T-VLM/docs/plans/2026-06-23-vqa-6phase-qwen3vl8b-4090.md`) —
  it will be added when that plan executes.

## Boundaries

- The engine repo stays independent — it is simultaneously the *training* codebase
  and the *runtime* Brain endpoint (served by agentbot's stack supervisor).
- GPU budget: teacher VLM on the RTX 4090 must be the dense 8B
  (`Qwen/Qwen3-VL-8B-Instruct`, ~17 GB); the 30B-A3B MoE does not fit.

## Handoffs

- Product A → **agentbot** Brain (`stack.brain_model`, `vlm.base_url`)
- Product B → **evalbot** (open-loop / closed-loop comparison vs baseline VLA)
