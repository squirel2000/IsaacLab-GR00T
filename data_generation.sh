#!/usr/bin/env bash
set -euo pipefail

# Top-level launcher for the IsaacLab mimic dataset pipeline.
# Paths below are relative to IsaacLab after this script changes directory.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="${ROOT_DIR}/IsaacLab"

TASK="${TASK:-Isaac-Cabinet-Pour-G1-Abs-v0}"
NUM_DEMOS="${NUM_DEMOS:-10}"
GENERATION_NUM_TRIALS="${GENERATION_NUM_TRIALS:-1000}"
DEVICE="${DEVICE:-cuda}"
NUM_ENVS="${NUM_ENVS:-1}"

DATASET_DIR="${DATASET_DIR:-../datasets/g1_cabinet_pour}"
RAW_DATASET="${RAW_DATASET:-${DATASET_DIR}/g1_pour_dataset.hdf5}"
ANNOTATED_DATASET="${ANNOTATED_DATASET:-${DATASET_DIR}/g1_pour_annotated.hdf5}"
GENERATED_DATASET="${GENERATED_DATASET:-${DATASET_DIR}/g1_pour_generated.hdf5}"
OUTPUT_DATASET_DIR="${OUTPUT_DATASET_DIR:-../datasets/G1_Inspire_Cabinet_Pour_Dataset}"

cd "${ISAACLAB_DIR}"
mkdir -p "$(dirname "${RAW_DATASET}")"

./isaaclab.sh -p scripts/gr00t_script/record_g1.py \
  --task "${TASK}" \
  --num_demos "${NUM_DEMOS}" \
  --dataset_file "${RAW_DATASET}"

./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos_g1.py \
  --device "${DEVICE}" \
  --task "${TASK}" \
  --input_file "${RAW_DATASET}" \
  --output_file "${ANNOTATED_DATASET}"

./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset_g1.py \
  --device "${DEVICE}" \
  --enable_cameras \
  --num_envs "${NUM_ENVS}" \
  --generation_num_trials "${GENERATION_NUM_TRIALS}" \
  --input_file "${ANNOTATED_DATASET}" \
  --output_file "${GENERATED_DATASET}" \
  --headless

python scripts/gr00t_script/convert_hdf5_to_parquet.py \
  --hdf5_path "${GENERATED_DATASET}" \
  --output_dataset_dir "${OUTPUT_DATASET_DIR}"
