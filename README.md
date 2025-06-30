
# Isaac-GR00T
This repository contains the code for the Isaac-GR00T project, which is designed to work with the Gr00t model. The project includes scripts for training, finetuning, and launching client-server interactions.

## Gr00t Finetune Script

This script is designed to finetune the Gr00t model on a specific dataset. It utilizes the `gr00t_finetune_g1.py` script for the finetuning process.

```bash
cd Isaac-GR00T
# Finetuning the Gr00t model
python scripts/gr00t_finetune_g1.py

# Training the Gr00t model on GPU 1, if available (e.g., on server with 2 x H100)
CUDA_VISIBLE_DEVICES=1 python scripts/gr00t_finetune_g1.py
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
