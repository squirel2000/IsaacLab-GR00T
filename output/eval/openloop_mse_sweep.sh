#!/bin/bash
# Offline open-loop MSE sweep over checkpoints -> overfitting curve.
#
# WHY: train loss is a velocity-field (flow-matching) MSE on the TRAINING data and
# can keep dropping while the policy overfits. open_loop_eval.py instead rolls out
# the fully-denoised ACTION and reports action-space MSE vs ground truth. Run it on
# a HELD-OUT dataset (episodes NOT used in training) per checkpoint and the minimum
# of MSE-vs-step is the best checkpoint; a rising MSE while train loss still falls is
# the overfitting onset (the N1.5 phenomenon). Needs NO change to the training loop
# and sidesteps factory.py's "sharded dataset cannot hold out an eval set" assert.
#
# Usage:
#   CKPT_ROOT=/data/VLA/experiments/.../N1_7_fft_retrain \
#   DATASET=/data/.../OpenArm_O6_CanSorting_HELDOUT \
#   TRAJ_IDS="0 1 2 3 4" bash openloop_mse_sweep.sh
#
# To feed the side-by-side comparison figure, point OUT at the analysis CSV the
# generator expects (n16_openloop_mse.csv / n17_openloop_mse.csv), e.g.:
#   OUT=../analysis/n16_vs_n17/n17_openloop_mse.csv CKPT_ROOT=... DATASET=... \
#     bash openloop_mse_sweep.sh
#   python3 ../analysis/n16_vs_n17/make_compare_svg.py   # re-render with panel 2 filled
set -u
REPO=${REPO:-/home/asus/Gits/IsaacLab-GR00T/Isaac-GR00T_n1d7}
CKPT_ROOT=${CKPT_ROOT:?set CKPT_ROOT to the training output dir holding checkpoint-* dirs}
DATASET=${DATASET:?set DATASET to a HELD-OUT lerobot dataset (episodes NOT in training)}
EMB=${EMB:-NEW_EMBODIMENT}
TRAJ_IDS=${TRAJ_IDS:-"0 1 2 3 4"}
STEPS=${STEPS:-1000}
ACTION_HORIZON=${ACTION_HORIZON:-16}
DENOISING=${DENOISING:-16}
OUT=${OUT:-$CKPT_ROOT/openloop_mse_sweep.csv}
PATIENCE=${PATIENCE:-3}   # stop after held-out MSE rises this many checkpoints in a row (0 = sweep all)

echo "step,mse,mae" > "$OUT"
ckpts=$(ls -d "$CKPT_ROOT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
[ -z "$ckpts" ] && { echo "No checkpoint-* dirs under $CKPT_ROOT"; exit 1; }

best_mse=; best_step=; rises=0; prev_mse=
for c in $ckpts; do
  step=$(basename "$c" | sed 's/checkpoint-//')
  log=$(cd "$REPO" && uv run python gr00t/eval/open_loop_eval.py \
        --dataset-path "$DATASET" --model-path "$c" --embodiment-tag "$EMB" \
        --action-horizon "$ACTION_HORIZON" --steps "$STEPS" \
        --denoising-steps "$DENOISING" --traj-ids $TRAJ_IDS 2>&1)
  mse=$(echo "$log" | grep -oE "Average MSE across all trajs: [0-9.eE+-]+" | tail -1 | awk '{print $NF}')
  mae=$(echo "$log" | grep -oE "Average MAE across all trajs: [0-9.eE+-]+" | tail -1 | awk '{print $NF}')
  if [ -z "$mse" ]; then echo "[step $step] eval failed (no MSE line)"; continue; fi
  echo "$step,$mse,$mae" >> "$OUT"
  echo "[step $step] held-out action MSE=$mse MAE=$mae"

  if [ -z "$best_mse" ] || awk "BEGIN{exit !($mse < $best_mse)}"; then best_mse=$mse; best_step=$step; fi
  if [ -n "$prev_mse" ] && awk "BEGIN{exit !($mse > $prev_mse)}"; then rises=$((rises+1)); else rises=0; fi
  prev_mse=$mse
  if [ "$PATIENCE" -gt 0 ] && [ "$rises" -ge "$PATIENCE" ]; then
    echo "OVERFITTING onset: MSE rose $rises checkpoints in a row -> stopping sweep."
    break
  fi
done
echo "BEST: checkpoint-$best_step  held-out MSE=$best_mse   (full curve: $OUT)"
