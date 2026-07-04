# vlm-trainbot — runbook

Fine-tunes GR00T N1.7's VLM backbone (Cosmos-Reason2-2B) on auto-generated VQA, then
produces two artifacts: **Product A** (standalone VLM — agentbot's Brain endpoint) and
**Product B** (the VLA with the tuned backbone swapped in). All implementation lives in
the engine repo `engines/vlm/Isaac-GR00T-VLM/` — this runbook drives it.

## Prerequisites

```bash
cd engines/vlm/Isaac-GR00T-VLM
uv sync                          # builds .venv (torch cu128 wheels)
```

- Base model `nvidia/Cosmos-Reason2-2B` cached in the local HF hub (gated — resolved
  offline via `resolve_model_path`).
- Teacher on the RTX 4090: use the dense `Qwen/Qwen3-VL-8B-Instruct` (~17 GB bf16).
  The default 30B-A3B MoE (~62 GB) does NOT fit — always pass `--teacher-model`.
- Dataset (LeRobot v2): `datasets/OpenArm_CanSorting_MultiTask_Sim_dataset_O6_0403`.

## The 5-step flow (run from `engines/vlm/Isaac-GR00T-VLM/`)

```bash
DS=../../../datasets/OpenArm_CanSorting_MultiTask_Sim_dataset_O6_0403
GR=../../../artifacts/checkpoints/gr00t

# 1) Generate VQA (teacher VLM + templates) from LeRobot videos
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 uv run python -m vlm_lora.gen_vqa_from_lerobot \
  --dataset-path "$DS" --out-dir artifacts/vqa --num-episodes 100 \
  --teacher-model Qwen/Qwen3-VL-8B-Instruct

# 2) LoRA fine-tune (frozen 2B base + r=16 adapters; fits 24 GB)
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 uv run python -m vlm_lora.train_vlm_lora \
  --dataset-path artifacts/vqa/data.train.jsonl --image-root artifacts/vqa \
  --output-dir artifacts/cosmos_r2_lora

# 3) Merge adapter into the base -> Product A (standalone VLM)
uv run python -m vlm_lora.merge_lora --adapter-dir artifacts/cosmos_r2_lora \
  --out-dir "$GR/lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged"

# 4) Swap Product A into a baseline VLA -> Product B (pure safetensors, key intersection)
uv run python -m vlm_lora.swap_backbone \
  --vla-ckpt "$GR/N1_7_fft_0614_150k_lr1e4_no_tune_visual" \
  --merged-vlm "$GR/lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged" \
  --out-dir "$GR/swapped_checkpoints/N1_7_cosmosR2lora_swapped"

# 5) Evaluate VQA accuracy by question type (7 types)
uv run python -m vlm_lora.eval_vlm_vqa \
  --model-dir "$GR/lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged" \
  --val-jsonl artifacts/vqa/data.val.jsonl --image-root artifacts/vqa \
  --out-json artifacts/eval/acc.json
```

All knobs are documented in the engine's
[configs/default.yaml](../../engines/vlm/Isaac-GR00T-VLM/configs/default.yaml); CLI
flags override. Serve Product A as agentbot's Brain:
`bash examples/run_vlm_server.sh` (or let the stack supervisor launch it).

## Pending plan

The 6-phase VQA re-pipeline (phase-aware frames + templates covering all 7 question
types, Qwen3-VL-8B teacher on the 4090) is specified in
`engines/vlm/Isaac-GR00T-VLM/docs/plans/2026-06-23-vqa-6phase-qwen3vl8b-4090.md`.
When executed, a thin `harness/run_vqa_flow.py` driver will be added here.

## Handoffs

Product A → **agentbot** (`stack.brain_model`); Product B → **evalbot** for
open-loop/closed-loop comparison against the baseline VLA.
