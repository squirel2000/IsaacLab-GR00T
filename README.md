# IsaacLab-GR00T

Integration glue for training and evaluating GR00T / StarVLA VLA models inside IsaacLab simulations.

## Automated Fine-tuning → Deploy Pipeline

For a hands-off, resumable end-to-end run (wait for a free H100 → fine-tune on Pegasus →
download the verified checkpoint → switch Wi-Fi → deploy to the asus-4090 sim box →
offline HTML report), use the single entry point `gr00t_pipeline.py`. The pipeline code lives
under `scripts/` (`scripts/pipeline/` + shared `scripts/common/`). See
**[scripts/pipeline/PIPELINE.md](./scripts/pipeline/PIPELINE.md)** for details and
**[docs/pipeline_architecture.html](./docs/pipeline_architecture.html)** for an illustrated walkthrough.

```powershell
pip install -r requirements-pipeline.txt
copy scripts\pipeline\config\config.example.yaml scripts\pipeline\config\config.yaml   # edit paths / passwords (gitignored)
python scripts/gr00t_pipeline.py run                   # resume or start  (run --reset for fresh)
python scripts/gr00t_pipeline.py status                # inspect state without touching anything
python scripts/gr00t_pipeline.py dashboard             # live web dashboard at http://localhost:8770
```

## Repository layout

- [Isaac-GR00T/](Isaac-GR00T/) — upstream NVIDIA GR00T model + finetuning scripts
- [IsaacLab/](IsaacLab/) — IsaacLab + custom scripts under `scripts/gr00t_script/`
- [starVLA/](starVLA/) — StarVLA model server
- [scripts/eval/](scripts/eval/) — closed-loop eval harness; single entry point `run_eval.py` (server + IsaacLab client)
- `datasets/`, `artifacts/checkpoints/` — shared assets across backends

## 1. Collect a dataset

Two paths produce a GR00T-compatible dataset for the G1 "open drawer, pick-and-place mug, pour water" task. Adjust target poses in [IsaacLab/scripts/gr00t_script/utils/constants/](IsaacLab/scripts/gr00t_script/utils/constants/).

Run IsaacLab commands from [IsaacLab/](IsaacLab/). Use `../datasets/...` paths when the output should land in the shared root [datasets/](datasets/) directory.

**Option A — data-collection agent** (99.1% success / 2000 trials):
```bash
cd IsaacLab
./isaaclab.sh -p scripts/gr00t_script/data_collect_agent.py
```

**Option B — IsaacLab mimic pipeline** (98.5% success / 2000 trials):
```bash
# Record 10 demonstrations -> ../datasets/g1_cabinet_pour/g1_pour_dataset.hdf5
cd IsaacLab
./isaaclab.sh -p scripts/gr00t_script/record_g1.py \
  --num_demos 10 \
  --dataset_file ../datasets/g1_cabinet_pour/g1_pour_dataset.hdf5

# Annotate the recorded demonstrations -> g1_pour_annotated.hdf5
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos_g1.py \
  --input_file ../datasets/g1_cabinet_pour/g1_pour_dataset.hdf5 \
  --output_file ../datasets/g1_cabinet_pour/g1_pour_annotated.hdf5

# Generate the full HDF5 dataset
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset_g1.py \
  --input_file ../datasets/g1_cabinet_pour/g1_pour_annotated.hdf5 \
  --output_file ../datasets/g1_cabinet_pour/g1_pour_generated.hdf5

# Convert HDF5 -> Parquet + MP4 for GR00T
python scripts/gr00t_script/convert_hdf5_to_parquet.py \
  --hdf5_path ../datasets/g1_cabinet_pour/g1_pour_generated.hdf5 \
  --output_dataset_dir ../datasets/G1_Inspire_Cabinet_Pour_Dataset
```

## 2. Generate dataset metadata for GR00T

GR00T expects `info.json` and `episodes.jsonl` inside `<dataset>/meta/`. Generate them with:

```bash
cd IsaacLab
python scripts/gr00t_script/utils/generate_dataset_meta.py \
  --robot_type g1_inspire \
  --dataset_root ../datasets/G1_Inspire_Cabinet_Pour_Dataset
```

Requirements:
- Videos at `<dataset>/videos/chunk-<CHUNK_ID>/observation.images.<video_key>/episode-<EPISODE_ID>.mp4`
- `modality.json` and `tasks.jsonl` already present in `meta/`

The script extracts video metadata into `info.json` and one record per episode (index, task description, frame count) into `episodes.jsonl`.

## 3. Finetune

### GR00T N1.5 on RTX 4090

```bash
cd Isaac-GR00T
nohup python scripts/gr00t_finetune.py \
  --dataset_path ../datasets/G1_Inspire_Cabinet_Pour_Dataset \
  --output_dir ../artifacts/checkpoints/gr00t/G1_Inspire_Cabinet_Pour_N1_5_fft_200k \
  --data_config g1_can_pick_and_sort \
  --batch_size 16 --max_steps 200000 --save_steps 10000 \
  --tune_visual --tune_projector --tune_diffusion_model \
  --lora_rank 0 --lora_full_model \
  --dataloader_num_workers 12 --gradient_accumulation_steps 1 \
  --report_to tensorboard --embodiment_tag new_embodiment --video_backend decord \
  >train.log 2>&1 &
```

### GR00T N1.6 on RTX 5090

Visual + projector tuning disabled due to memory constraints; DeepSpeed Stage 3 + bf16 + grad checkpointing keep it in budget.

```bash
nohup .venv/bin/torchrun --nproc_per_node=1 gr00t/experiment/launch_finetune.py \
  --base_model_path "nvidia/GR00T-N1.6-3B" \
  --dataset_path ../datasets/OpenArm_CanSorting_Dataset_MoveBasket_AdjustSteps \
  --output_dir ../artifacts/checkpoints/gr00t/OpenArm_CanSorting_N1_6_fft_200k \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/Openarm_Leaphand/modality_config.py \
  --global-batch-size 1 --num-gpus 1 --max_steps 200000 --save_steps 10000 \
  --save-total-limit 2 --shard-size 1024 \
  --no-tune-visual --no-tune-projector --tune-diffusion-model \
  --dataloader_num_workers 4 --gradient_accumulation_steps 2 --no-use-wandb \
  --gradient-checkpointing --deepspeed-stage 3 --no-fp16 --bf16 \
  >train.log 2>&1 &
```

### Multi-GPU H100 (pin GPU 1)

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/gr00t_finetune.py \
  --dataset-path /data/storage/HumanoidRobot/datasets/G1_CubeStacking_Dataset_3k \
  --output-dir /data/storage/HumanoidRobot/experiments/g1-cube-stacking-3k/N1_5_fft_500k_visual_ds16 \
  --data-config unitree_g1 --embodiment_tag new_embodiment \
  --gpu-id 1 --num-gpus 1 --batch-size 32 --video-backend torchvision_av \
  --max-steps 500000 --save-steps 10000 --eval_steps 10000 \
  --eval-args-trajs 2 --eval-args-max-steps 1300 --dataset-split-ratio "9:1" \
  --dataloader-num-workers 4 --tune_visual --denoising_step 16
```

### Pull a remote checkpoint

```bash
rsync -avz --progress \
  asus@192.168.32.143:/home/asus/Gits/IsaacLab-GR00T/artifacts/checkpoints/gr00t/<run>/ \
  ./artifacts/checkpoints/gr00t/<run>/
```

### Monitor with TensorBoard

Requires `--report_to tensorboard` during training. Open `http://localhost:6006/` after launching:

```bash
tensorboard --logdir ./artifacts/checkpoints/gr00t
```

## 4. Launch policy server + IsaacLab client

### Recommended — single entry point

[scripts/eval/run_eval.py](scripts/eval/run_eval.py) is the only eval entry point. It reads
[scripts/eval/configs/eval_config.yaml](scripts/eval/configs/eval_config.yaml) (the eval plan + global knobs),
evals each checkpoint that lacks a complete log (crash-resilient: server stays up, client relaunches
and accumulates episodes), then prints a comparison and writes a success-rate chart. Re-running with
no new checkpoints just re-summarises existing logs.

```bash
# Eval every checkpoint in eval_config.yaml (default 100 eps each), then compare + chart
python3 scripts/eval/run_eval.py

# Fewer episodes
python3 scripts/eval/run_eval.py --target 50
```

Everything else (which checkpoints, episode count, headless, cameras, video, per-backend server
settings) lives in [scripts/eval/configs/](scripts/eval/configs/) — `eval_config.yaml` (the plan +
knobs) plus the per-backend JSONs. See [scripts/eval/README.md](scripts/eval/README.md) for details.

### Manual launch (debugging / fallback)

```bash
# Terminal A — server
cd Isaac-GR00T && conda activate env_gr00t
python3 ./scripts/inference_service.py --server \
  --model_path ../artifacts/checkpoints/gr00t/<run>/checkpoint-200000/ \
  --embodiment_tag new_embodiment --data_config openarm_leaphand \
  --denoising_steps 4

# Terminal B — client
cd IsaacLab && conda activate env_isaaclab
python3 ./scripts/gr00t_script/gr00t_infer_agent.py \
  --task "Isaac-Can-Sorting-OpenArm-DexHand-v0" \
  --max_eps_num 2000 --save_video \
  --openarm_hand_type leaphand_right --filter
```

## Shared storage layout

Datasets and fine-tuned checkpoints live at the project root so Isaac-GR00T, StarVLA, and future VLA backends share the same assets:

```
datasets/<dataset_name>/
artifacts/checkpoints/gr00t/<run_name>/
artifacts/checkpoints/starvla/<run_name>/
```

Override the roots with `VLA_DATASETS_ROOT` and `VLA_CHECKPOINTS_ROOT`. Upstream demo data, simulator assets, rollout logs, TensorBoard runs, and analysis outputs stay in their owning repositories unless they become shared assets.

## Client-server architecture

See [Isaac-GR00T/gr00t/policy/server_client.py](Isaac-GR00T/gr00t/policy/server_client.py) and [Isaac-GR00T/gr00t/policy/policy.py](Isaac-GR00T/gr00t/policy/policy.py).

```text
+----------------+         +----------------+         +----------------+
|  PolicyClient  | <-----> |  PolicyServer  | <-----> | Policy (Model) |
+----------------+         +----------------+         +----------------+
       | 1. call_endpoint()       |                          |
       |------------------------->|                          |
       |                          | 2. recv + deserialize    |
       |                          | 3. endpoint lookup       |
       |                          | 4. handler(**data)       |
       |                          |------------------------->|
       |                          |                          | 5. get_action()
       |                          |<-------------------------|
       |                          | 6. serialize + send      |
       |<-------------------------|                          |
       | 7. deserialize + return  |                          |
```

- **Endpoint registration:** `@self.register_endpoint("get_action", self.policy.get_action)`
- **Wire format:** MsgPack with custom numpy / class handlers
- **Transport:** ZeroMQ REQ/REP sockets

## Conda env backup / restore

```bash
# Export
conda activate isaaclab
conda env export > isaaclab_environment.yml

# Restore
conda deactivate
conda env remove --name isaaclab
conda env create -f isaaclab_environment.yml -n isaaclab
```
