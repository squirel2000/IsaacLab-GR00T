!/bin/bash

# Mimic Generation Script

# Record the manual-based teleoperation demonstration for the mimic dataset:
./isaaclab.sh -p scripts/gr00t_script/record_g1.py \
  --task Isaac-Cabinet-Pour-G1-Abs-v0 \
  --num_demos 10

# Annotate the subtasks for the manual-based recorded demonstrations:
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos_g1.py \
  --device cuda \
  --task Isaac-Cabinet-Pour-G1-Abs-v0 \
  --input_file ./datasets/g1_dataset.hdf5 \
  --output_file ./datasets/g1_dataset_annotated.hdf5
  # --headless # headless will fluctuate the GPU memory drastically, and even cause OOM errors

# Use Isaac Lab Mimic to generate the mimic dataset:
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset_g1.py \
  --device cuda --enable_cameras --num_envs 1 --generation_num_trials 1000 \
  --input_file ./datasets/g1_dataset_annotated.hdf5 \
  --output_file ./datasets/g1_generated_dataset.hdf5 \
  --headless