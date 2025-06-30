
# Isaac-GR00T
This repository contains the code for the Isaac-GR00T project, which is designed to work with the Gr00t model. The project includes scripts for training, finetuning, and launching client-server interactions.

## Gr00t Finetune Script

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

To launch the client and server scripts, you can use the provided `launch_isaac_gr00t.sh` script. This script will set up the environment and start the necessary processes.

```bash
./launch_isaac_gr00t.sh
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
