
# IsaacLab-GR00T
This repository contains the code for the Isaac-GR00T project, which is designed to work with the Gr00t model. The project includes scripts for training, finetuning, and launching client-server interactions.

## IsaacLab

Generate the dataset for Gr00t using the `record_g1.py`, `annotate_demos_g1.py`, and `generate_dataset_g1.py` scripts for G1 that opens the drawer, pick-and-place the mug on the mat, and pour water into the mug.


### Use data collection agent script to collect data as parquet and mp4 files (99.1% success rate for 2000 trials)
```bash
./isaaclab.sh -p scripts/gr00t_script/data_collect_agent.py
```

### Use isaaclab mimic method to record, annotate, and generate dataset for Gr00t (98.5% success rate for 2000 trials)
```bash
# Record 10 demonstrations as ./datasets/g1_cabinet_pour/g1_pour_dataset.hdf5
./isaaclab.sh -p scripts/gr00t_script/record_g1.py --num_demos 10

# Annotate the recorded demonstrations, ./datasets/g1_cabinet_pour/g1_pour_annotated.hdf5
./isaaclab.sh -p scripts/gr00t_script/annotate_demos_g1.py

# Generate the dataset in HDF5 format based on the annotated demonstrations
./isaaclab.sh -p scripts/gr00t_script/generate_dataset_g1.py

# Convert the HDF5 dataset to Parquet format and MP4 videos for Isaac-GR00T
python ./scripts/gr00t_script/convert_hdf5_to_parquet.py
```

## Shared Project Storage

Project-level VLA fine-tuning datasets and fine-tuned checkpoints are stored at the root so Isaac-GR00T, StarVLA, and future VLA backends can reuse the same assets:

```text
datasets/<dataset_name>
artifacts/checkpoints/gr00t/<run_name>
artifacts/checkpoints/starvla/<run_name>
```

You can override these locations with `VLA_DATASETS_ROOT` and `VLA_CHECKPOINTS_ROOT`. Upstream demo data, simulator assets, rollout logs, TensorBoard runs, and analysis outputs should stay in their owning repositories unless they become shared fine-tuning datasets or checkpoints.

## Human Demonstration Collection on IsaacLab

To collect human demonstrations for GR00T on G1 that opens the drawer, pick-and-place the mug on the mat, and pour water into the mug, you can adjust the poses in the following script [constants.py](./IsaacLab/scripts/gr00t_script/utils/constants.py):

## Dataset Collection on IsaacLab

There are two ways to generate the dataset for GR00T on G1 that opens the drawer, pick-and-place the mug on the mat, and pour water into the mug.

1. Option 1: Use data collection agent script to collect data as parquet and mp4 files (99.1% success rate for 2000 trials)
```bash
./isaaclab.sh -p scripts/gr00t_script/data_collect_agent.py
```

2. Option 2: Use isaaclab mimic method to record, annotate, and generate dataset for GR00T (98.5% success rate for 2000 trials)
```bash
# Record 10 demonstrations as ./datasets/g1_cabinet_pour/g1_pour_dataset.hdf5
./isaaclab.sh -p scripts/gr00t_script/record_g1.py --num_demos 10

# Annotate the recorded demonstrations, ./datasets/g1_cabinet_pour/g1_pour_annotated.hdf5
./isaaclab.sh -p scripts/gr00t_script/annotate_demos_g1.py

# Generate the dataset in HDF5 format based on the annotated demonstrations
./isaaclab.sh -p scripts/gr00t_script/generate_dataset_g1.py

# Convert the HDF5 dataset to Parquet format and MP4 videos for Isaac-GR00T
python ./scripts/gr00t_script/convert_hdf5_to_parquet.py
```

## Prepare the metadata for fine-tuning in GR00T

   To prepare your dataset for GR00T, you need to generate metadata files (`info.json`, `episodes.jsonl`) for your robot dataset. Use the script `scripts/utils/generate_dataset_meta.py` for this purpose.

   ### Dataset Structure Requirements
   - Video files should be organized as:
     `<dataset_root>/videos/chunk-<CHUNK_ID>/observation.images.<video_key>/episode-<EPISODE_ID>.mp4`
   - The script extracts metadata from these videos and generates the necessary JSON/JSONL files.
   - It also reads task descriptions from a `tasks.jsonl` file located in the `meta` directory of the dataset.
   - The generated `info.json` contains dataset-wide metadata, while `episodes.jsonl` contains per-episode information such as episode index, task description, and length (number of frames).
   - Ensure the dataset directory structure is correct and that `modality.json` and `tasks.jsonl` files exist in the `meta` directory.

   ### Usage Example
   ```bash
   python scripts/utils/generate_dataset_meta.py <dataset_root>
   ```
   For example:
   ```bash
   python scripts/utils/generate_dataset_meta.py ../datasets/G1_Inspire_Cabinet_Pour_Dataset
   ```
   This will create `info.json` and `episodes.jsonl` files in the `meta` directory of your dataset root.


## Finetune the GR00T VLA Model on Isaac-GR00T

This script is designed to finetune the Gr00t model on a specific dataset. It utilizes the `gr00t_finetune_g1.py` script for the finetuning process.

```bash
cd Isaac-GR00T
# Finetuning the Gr00t_N1.5 model on 4090
nohup python scripts/gr00t_finetune.py \
  --dataset_path ../datasets/G1_Inspire_Cabinet_Pour_Dataset \
  --output_dir ../artifacts/checkpoints/gr00t/G1_Inspire_Cabinet_Pour_Dataset_Checkpoints_N1_5_fft_200k \
  --data_config g1_can_pick_and_sort \
  --batch_size 16 --max_steps 200000 --save_steps 10000 \
  --tune_visual --tune_projector --tune_diffusion_model \
  --lora_rank 0 --lora_full_model \
  --dataloader_num_workers 12 --gradient_accumulation_steps 1 \
  --report_to "tensorboard" --embodiment_tag new_embodiment --video_backend decord  >train.log 2>&1 &

# Fine tuning the Gr00t N1.6 model on RTX 5090 (No visual and projector tuning due to memory constraints)
nohup .venv/bin/torchrun --nproc_per_node=1 gr00t/experiment/launch_finetune.py \
    --base_model_path "nvidia/GR00T-N1.6-3B" \
    --dataset_path ../datasets/OpenArm_CanSorting_Dataset_MoveBasket_AdjustSteps \
    --output_dir ../artifacts/checkpoints/gr00t/OpenArm_CanSorting_Dataset_MoveBasket_AdjustSteps_Checkpoints_N1_6_fft_200k \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/Openarm_Leaphand/modality_config.py \
    --global-batch-size 1 --num-gpus 1 --max_steps 200000 --save_steps 10000 --save-total-limit 2 --shard-size 1024 \
    --no-tune-visual --no-tune-projector --tune-diffusion-model \
    --dataloader_num_workers 4 --gradient_accumulation_steps 2 --no-use-wandb \
    --gradient-checkpointing \
    --deepspeed-stage 3 \
    --no-fp16 \
    --bf16 \
    >train.log 2>&1 &

# Training the Gr00t model on GPU 1, if available (e.g., on server with 2 x H100)
CUDA_VISIBLE_DEVICES=1 python3 scripts/gr00t_finetune.py \
  --dataset-path /data/storage/HumanoidRobot/datasets/G1_CubeStacking_Dataset_3k \
  --output-dir /data/storage/HumanoidRobot/experiments/g1-cube-stacking-3k-checkpoints/new_embodiment/N1_5_fft_500k_visual_ds16 \
  --data-config unitree_g1 --embodiment_tag new_embodiment \
  --gpu-id 1 --num-gpus 1 --batch-size 32 --video-backend torchvision_av \
  --max-steps 500000 --save-steps 10000 --eval_steps 10000 --eval-args-trajs 2 --eval-args-max-steps 1300 --dataset-split-ratio "9:1" --dataloader-num-workers 4 --tune_visual --denoising_step 16
```

Tips: To retrieve the remote folder into your local output directory using rsync, you can use the following command:

```bash
rsync -avz --progress asus@192.168.32.143:/home/asus/Gits/IsaacLab-GR00T/artifacts/checkpoints/gr00t/G1_Inspire_Cabinet_Pour_Dataset_Checkpoints_N1_5_fft_200k ./artifacts/checkpoints/gr00t/
```

## Launch the Client and Server Scripts

To launch the client and server scripts, you can manually run the client and server scripts as follows:

```bash
# Start the server on a new terminal in the "env_gr00t" conda environment
cd Isaac-GR00T && conda deactivate && conda activate env_gr00t
python3 ./scripts/inference_service.py \
  --server \
  --model_path ../artifacts/checkpoints/gr00t/openarm_leaphand_cansorting_N15_fft_200k_dataset_0226_radomization/checkpoint-200000/ \
  --embodiment_tag new_embodiment --data_config openarm_leaphand \
  --denoising_steps 4

python3 ./scripts/inference_service.py   --server   --model_path ../artifacts/checkpoints/gr00t/openarm_leaphand_cansorting_N15_fft_200k_movebasket/openarm_cansorting_N15_fft_200k_visual_ds4_lr1e-4_movebasket/checkpoint-200000/   --embodiment_tag new_embodiment   --data_config openarm_leaphand   --denoising_steps 4


# Start the client on another terminal in the "env_isaaclab" conda environment
cd IsaacLab && conda deactivate && conda activate env_isaaclab
python3 ./scripts/gr00t_script/gr00t_infer_agent.py \
  --task "Isaac-Can-Sorting-OpenArm-DexHand-v0" \
  --save_dir ./outputs/openarm_leaphand_cansorting_N15_fft_200k_movebasket/ \
  --max_eps_num 2000 \
  --save_video \
  --openarm_hand_type "leaphand_right" \
  --filter

 python3 ./scripts/gr00t_script/gr00t_infer_agent.py   --task "Isaac-Can-Sorting-OpenArm-DexHand-v0"   --save_dir ./outputs/openarmeaphando6_cansorting_N15_fft_200k_movebasket/   --max_eps_num 10 --save_video
```

Or you can use the provided `launch_isaac_gr00t.py` script. This script will set up the environment and start the necessary processes.

```bash
python3 ./launch_isaac_gr00t.py \
  --model_path ./output/G1_Inspire_Cabinet_Pour_Dataset_Checkpoints_N1_5_fft/ \
  --task "Isaac-Cabinet-Pour-G1-Abs-v0" 
```

## Check the trainging metrics in Tensorboard

To monitor the training metrics during the finetuning process, you can use TensorBoard. After starting your finetuning script with the `--report_to "tensorboard"` flag, you can launch TensorBoard to visualize the metrics.

```bash
tensorboard --logdir ../artifacts/checkpoints/gr00t
```

Then, open your web browser and navigate to `http://localhost:6006/` to view the TensorBoard dashboard.


## Source Code Analysis and Customization

### Client-Server Interaction

The following diagram illustrates the interaction between the [PolicyClient, PolicyServer](Isaac-GR00T/gr00t/policy/server_client.py), and the [Policy (Model)](Isaac-GR00T/gr00t/policy/policy.py) during the inference process:

```Text
+-------------------+         +-------------------+         +-------------------+
|                   |         |                   |         |                   |
|   PolicyClient    | <-----> |   PolicyServer    | <-----> |   Policy (Model)  |
|                   |         |                   |         |                   |
+-------------------+         +-------------------+         +-------------------+
        |                             |                              |
        | 1. call_endpoint()          |                              |
        |---------------------------->|                              |
        |                             | 2. recv & deserialize        |
        |                             | 3. endpoint lookup           |
        |                             | 4. handler.handler(**data)   |
        |                             |----------------------------->|
        |                             |                              | 5. get_action()
        |                             |                              |    (inference)
        |                             |<-----------------------------|
        |                             | 6. serialize & send result   |
        |<----------------------------|                              |
        | 7. deserialize & return     |                              |
        v                             v                              v
```

Key Syntax and Concepts:
- `Endpoint Registration - @self.register_endpoint("get_action", self.policy.get_action)`: Registers the policy's get_action method as the handler for the "get_action" endpoint.
- `Handler Invocation - handler.handler(**request.get("data", {}))`: Unpacks the data dictionary and calls the handler with named arguments (e.g., observation=..., options=...).
- `Data Serialization:` Uses MsgPack for efficient binary serialization, with custom handling for numpy arrays and custom classes.
- `ZeroMQ Communication:` Uses REQ/REP sockets for request-response messaging between client and server

## Steps to Backup (Export) and Restore the "isaaclab" Environment

Activate and export the environment to a YAML file

```bash
conda activate isaaclab
conda env export > isaaclab_environment.yml
```

Restore (Duplicate) the Environment from Backup

```bash
conda deactivate
conda env remove  --name isaaclab
conda env create -f isaaclab_environment.yml -n isaaclab
```

This will create a new conda environment named "isaaclab" with the same packages and dependencies as the original environment.
