
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
   python scripts/utils/generate_dataset_meta.py datasets/gr00t_collection/G1_Inspire_Cabinet_Pour_Dataset
   ```
   This will create `info.json` and `episodes.jsonl` files in the `meta` directory of your dataset root.


## Finetune the GR00T VLA Model on Isaac-GR00T

This script is designed to finetune the Gr00t model on a specific dataset. It utilizes the `gr00t_finetune_g1.py` script for the finetuning process.

```bash
cd Isaac-GR00T
# Finetuning the Gr00t_N1.5 model on 4090
python scripts/gr00t_finetune.py \
  --dataset_path demo_data/G1_CubeStacking_Dataset \
  --output_dir output/G1_CubeStacking_Dataset_Checkpoints_N1_5_fft \
  --data_config g1_can_pick_and_sort \
  --batch_size 8 --max_steps 100000 --save_steps 10000 \
  --tune_visual --tune_projector --tune_diffusion_model \
  --lora_rank 1024 --lora_full_model \
  --report_to "tensorboard" --embodiment_tag new_embodiment --video_backend torchvision_av

# Training the Gr00t model on GPU 1, if available (e.g., on server with 2 x H100)
CUDA_VISIBLE_DEVICES=1 python3 scripts/gr00t_finetune.py \
  --dataset-path /data/storage/HumanoidRobot/datasets/G1_CubeStacking_Dataset_3k \
  --output-dir /data/storage/HumanoidRobot/experiments/g1-cube-stacking-3k-checkpoints/new_embodiment/N1_5_fft_500k_visual_ds16 \
  --data-config unitree_g1 --embodiment_tag new_embodiment \
  --gpu-id 1 --num-gpus 1 --batch-size 32 --video-backend torchvision_av \
  --max-steps 500000 --save-steps 10000 --eval_steps 10000 --eval-args-trajs 2 --eval-args-max-steps 1300 --dataset-split-ratio "9:1" --dataloader-num-workers 4 --tune_visual --denoising_step 16
```

## Launch the Client and Server Scripts

To launch the client and server scripts, you can manually run the client and server scripts as follows:

```bash
# Start the server on a new terminal in the "env_gr00t" conda environment
cd Isaac-GR00T && conda deactivate && conda activate env_gr00t
python3 ./scripts/inference_service.py \
  --server \
  --model_path ./output/G1_Inspire_Cabinet_Pour_Dataset_Checkpoints_N1_5_fft/ \
  --embodiment_tag new_embodiment \
  --data_config g1_can_pick_and_sort \
  --denoising_steps 4

# Start the client on another terminal in the "env_isaaclab" conda environment
cd IsaacLab && conda deactivate && conda activate env_isaaclab
python3 ./scripts/gr00t_script/gr00t_infer_agent.py \
  --task "Isaac-Cabinet-Pour-G1-Abs-v0" \
  --save_dir ./output/G1_Inspire_Cabinet_Pour_Dataset_Checkpoints_N1_5_fft_lowpass/ \
  --max_eps_num 2000 \
  --filter
```

Or you can use the provided `launch_isaac_gr00t.py` script. This script will set up the environment and start the necessary processes.

```bash
python3 ./launch_isaac_gr00t.py \
  --model_path ./output/G1_Inspire_Cabinet_Pour_Dataset_Checkpoints_N1_5_fft/ \
  --task "Isaac-Cabinet-Pour-G1-Abs-v0" 
```

## Steps to Backup (Export)  and Restore the "isaaclab" Environment

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
