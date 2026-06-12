#!/bin/bash
# Sequentially eval N1.6 then N1.7 (100 eps each) on Display 0, save results.
set -u
ROOT=/home/asus/Gits/IsaacLab-GR00T
LOGDIR=$ROOT/output/eval/logs
CONDA_SH=/home/asus/miniforge3/etc/profile.d/conda.sh
EPS=100
export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority

log() { echo "[$(date '+%F %T')] $*"; }

wait_ready() { # $1=logfile $2=server_pid $3=timeout_s
  local f="$1" pid="$2" t="${3:-900}" i=0
  while [ "$i" -lt "$t" ]; do
    grep -q "Server is ready and listening" "$f" 2>/dev/null && return 0
    kill -0 "$pid" 2>/dev/null || { log "SERVER PROCESS $pid DIED"; return 2; }
    sleep 4; i=$((i+4))
  done
  return 1
}

run_one() { # tag server_activate server_dir embodiment ckpt ver config save_id client_pp
  local TAG="$1" SACT="$2" SDIR="$3" EMB="$4" CKPT="$5" VER="$6" CFG="$7" SID="$8" CPP="${9:-}"
  log "=== $TAG: launching server ($EMB) ==="
  bash -c "cd '$SDIR' && $SACT && exec python3 -u -m gr00t.eval.run_gr00t_server --model-path '$CKPT' --embodiment-tag '$EMB' --port 5555" \
    > "$LOGDIR/${TAG}_server.log" 2>&1 &
  local SPID=$!
  wait_ready "$LOGDIR/${TAG}_server.log" "$SPID" 900
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    log "=== $TAG: SERVER NOT READY (rc=$rc) -- skipping ==="; kill "$SPID" 2>/dev/null; return 1
  fi
  log "=== $TAG: server ready (pid $SPID); launching client ($EPS eps) ==="
  local PPENV=""
  [ -n "$CPP" ] && PPENV="PYTHONPATH='$CPP' "
  timeout 14400 bash -c "cd '$ROOT/IsaacLab' && source '$CONDA_SH' && conda activate env_isaaclab && ${PPENV}exec python3 -u '$ROOT/gr00t_eval/gr00t_infer_agent.py' \
    --task Isaac-Can-Sorting-OpenArm-DexHand-v0 --policy gr00t --policy_config '$CFG' \
    --host localhost --port 5555 --max_eps_num $EPS --openarm_hand_type linkerhand_o6 \
    --filter --save_dir 'output/infer_record/$SID' --gr00t_ver $VER --save_video" \
    > "$LOGDIR/${TAG}_client.log" 2>&1
  log "=== $TAG: client exited (rc=$?) ==="
  kill "$SPID" 2>/dev/null; sleep 5; pkill -9 -f run_gr00t_server 2>/dev/null; sleep 4
  log "=== $TAG: finished, server stopped ==="
}

log "########## EVAL COMPARE START ##########"

run_one n16 "source '$CONDA_SH' && conda activate env_gr00t" \
  "$ROOT/Isaac-GR00T" "NEW_EMBODIMENT" \
  "$ROOT/artifacts/checkpoints/gr00t/openarm_linkerhando6_multitask_N16_150k_dataset_0403_no_tune_visual/checkpoint-150000" \
  "N1.6" "$ROOT/gr00t_eval/policy_configs/gr00t_n16_openarm_o6.json" \
  "openarm_linkerhando6_multitask_N16_150k_dataset_0403_no_tune_visual_checkpoint-150000" ""

run_one n17 "source '$ROOT/Isaac-GR00T_n1d7/.venv/bin/activate'" \
  "$ROOT/Isaac-GR00T_n1d7" "new_embodiment" \
  "$ROOT/artifacts/checkpoints/gr00t/openarm_linkerhando6_multitask_N17_150k_dataset_0403_no_tune_visual/checkpoint-150000" \
  "N1.7" "$ROOT/gr00t_eval/policy_configs/gr00t_n17_openarm_o6.json" \
  "openarm_linkerhando6_multitask_N17_150k_dataset_0403_no_tune_visual_checkpoint-150000" \
  "$ROOT/Isaac-GR00T_n1d7"

log "########## EVAL COMPARE DONE ##########"
