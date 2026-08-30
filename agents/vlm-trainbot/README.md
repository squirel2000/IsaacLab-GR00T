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
flags override.

## The tool-calling variant — this is what agentbot's Brain actually runs

The 5 steps above train the VLM to answer VQA in free text. A **second, parallel** flow trains it
to emit OpenAI-style **tool calls** instead. Both LoRAs can coexist (every toolcall artifact path
is a `_toolcall` sibling), but note the asymmetry: **the toolcall variant produces a Product-A-shaped
artifact only — there is no `swap_backbone`/Product B for it.** It exists to be *served*, not
swapped into the VLA.

| Stage | VQA flow (above) | Tool-calling flow |
| --- | --- | --- |
| data gen | `vlm_lora.gen_vqa_from_lerobot` — **needs a teacher VLM on GPU**, hours | `vlm_lora.gen_toolcall_data` — **templates only, no teacher, CPU, seconds** |
| train | step 2, unchanged | *identical module & LoRA config*, `--max-steps 1500 --save-steps 500` |
| merge | step 3, unchanged | *identical*, into `lora_tuned_vlm_toolcall/` |
| swap | step 4 → Product B | **n/a — no swap** |
| eval | `eval_vlm_vqa` (accuracy × 7 question types) | `eval_toolcall` (valid-parse, skills-in-vocab, name-sequence, args-exact) |
| serve | — | `examples/run_vlm_server.sh` → `:8000` |

```bash
DS=/data/VLA/datasets/OpenArm_CanSorting_MultiTask_dataset_O6_0403
TC=artifacts/vlm_lora/toolcall

# 1) Template-derived labels — no teacher model, no GPU
HF_HUB_OFFLINE=1 uv run python -m vlm_lora.gen_toolcall_data \
  --dataset-path "$DS" --out-dir "$TC" --num-episodes 100 --frames-per-episode 2

# 2) Train + 3) merge: same modules as the VQA flow. Pick the GPU with h100.preflight(),
#    do NOT hardcode a device (see agents/tools/common/README.md rule 1).
uv run python -m vlm_lora.train_vlm_lora \
  --dataset-path "$TC/data.train.jsonl" --image-root "$TC" \
  --output-dir artifacts/vlm_lora/cosmos_r2_toolcall_lora --max-steps 1500 --save-steps 500
uv run python -m vlm_lora.merge_lora --adapter-dir artifacts/vlm_lora/cosmos_r2_toolcall_lora \
  --out-dir "$GR/lora_tuned_vlm_toolcall/Cosmos-Reason2-2B-toolcall-merged"

# 4) Serve as the Brain (same script as any variant — only VLM_MODEL_DIR changes)
VLM_MODEL_DIR="$GR/lora_tuned_vlm_toolcall/Cosmos-Reason2-2B-toolcall-merged" \
  HF_HUB_OFFLINE=1 bash examples/run_vlm_server.sh      # OpenAI-compatible, :8000
```

Endpoint `POST /v1/chat/completions` (+ `GET /health`) takes standard `{messages, tools}` with
multimodal `image_url`. The model only ever emits **text** shaped
`<tool_call>{"name":…,"arguments":…}</tool_call>` (Qwen/Hermes convention — base Cosmos-Reason2-2B
has no tool-calling chat template, so the fine-tune and the server agree on this format by
choice); `serve/toolcall.py` parses it into OpenAI `tool_calls` at the edge. **The parsing layer
is load-bearing** — a raw `generate` bypassing `serve/` returns the unparsed tagged string.
`eval_toolcall` scores through that same parser, so training and serving formats cannot drift.

**Why the tuned model, and not the base one, is the Brain:** untuned/VQA-tuned Cosmos over-decomposes
a sorting request into `pick`/`place`/`home` — skeleton skills with `sim_runnable=False` and no sim
task behind them, which used to stall the orchestrator until a 30-minute timeout. The toolcall LoRA
reliably emits a single `sort_can` instead. (The orchestrator now auto-skips skeleton skills, but the
LoRA is what makes the plan correct in the first place.)

### Traps worth knowing before re-running this

- **The currently-deployed toolcall weights came from `checkpoint-500`, not the full 1500 steps** —
  a separate manual merge of `<adapter>/checkpoint-500` was what actually produced the shipped model.
  Treat "1500 steps" as the *intended* recipe, not the provenance of the artifact on disk; re-run
  and re-merge if you need the step count to be true.
- **`gen_toolcall_data` has a silent label default.** `template_calls()` regex-matches `\borange\b` /
  `\bgreen\b` in each episode's task string and falls through to `{"target_color": "orange"}` with no
  warning. On a dataset whose task text omits the colour word this yields 100% orange labels and a
  model that trains cleanly while having learned a constant. Spot-check the generated JSONL.
- **Split-leak trap.** Generation writes `data.jsonl` (all rows), `data.train.jsonl` and
  `data.val.jsonl` (10%). Train on `data.train.jsonl` — the earlier scripts *logged* `wc -l
  data.jsonl` while *training* on the train split, so copying a log line into a command leaks val.
- **`HF_HUB_OFFLINE=1` is mandatory, not cosmetic** — the base model is gated and resolved from the
  local hub cache by `hf_utils.resolve_model_path`. If you invoke `.venv/bin/python` directly instead
  of `uv run`, `PYTHONPATH=<engine>/src` is mandatory too.

Full spec: the engine's
[integration plan, Phase D](../../engines/vlm/Isaac-GR00T-VLM/docs/plans/2026-06-21-vlm-brain-agentbot-integration.md).
Wiring on the agentbot side (`backend: gr00t-vlm`, `base_url: http://localhost:8000/v1`,
`Gr00tVLMClient` → `SkillCall`) belongs to agentbot:
[agentbot/docs/USING_VLM_BRAIN.md](../agentbot/docs/USING_VLM_BRAIN.md). Port neighbours in that
stack: GR00T policy server `:5555`, agentbot dashboard/API `:8780`.

## Pending plan

The 6-phase VQA re-pipeline (phase-aware frames + templates covering all 7 question
types, Qwen3-VL-8B teacher on the 4090) is specified in
`engines/vlm/Isaac-GR00T-VLM/docs/plans/2026-06-23-vqa-6phase-qwen3vl8b-4090.md`.
When executed, a thin `harness/run_vqa_flow.py` driver will be added here.

## Handoffs

Product A → **agentbot** (`stack.brain_model`); Product B → **evalbot** for
open-loop/closed-loop comparison against the baseline VLA.

The open-loop half of that comparison is a per-checkpoint invocation from the **N1.7** repo (run it
once for the baseline and once for the swapped checkpoint, then diff the plots):

```bash
CUDA_VISIBLE_DEVICES=0 MODE=right_only EMBODIMENT_TAG=new_embodiment TRAJ_IDS="0" \
  CHECKPOINT_PATH=<ckpt> DATASET_PATH=<ds> SAVE_PLOT_PATH=<out>.jpeg \
  bash examples/Openarm_LinkerHandO6/openloop_eval_openarm_o6.sh
```

**Provenance of the swap baselines** (`N1_7_fft_0614_150k_lr{1e4,5e5}_no_tune_visual`, referenced
as swap targets in step 4 and in the engine's `configs/default.yaml`) — both were trained one-GPU
each with:

```
MODE=right_only NUM_GPUS=1 MAX_STEPS=150000 SAVE_STEPS=5000
GLOBAL_BATCH_SIZE=32 GRADIENT_ACCUMULATION_STEPS=1
WARMUP_RATIO=0.1 WEIGHT_DECAY=1e-5 SAVE_TOTAL_LIMIT=3 DATALOADER_NUM_WORKERS=0
```

differing only in `LEARNING_RATE` (1e-4 vs 5e-5). Recorded here because nothing else in the repo
captures how these two checkpoints were produced.
