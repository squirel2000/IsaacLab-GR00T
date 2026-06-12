#!/bin/bash
# Crash-resilient sequential eval of N1.6 then N1.7 on Display 0.
#
# Why v2: the IsaacSim client intermittently dies on an Omniverse PhysX reentrancy
# bug ("carb.tasking Mutex Recursion not allowed") at an episode reset. v1 ran the
# client once, so a single crash aborted the version with partial data. v2 keeps the
# server up and RELAUNCHES the client, accumulating completed-episode outcomes across
# attempts into one combined log until TARGET episodes are collected (or guards trip).
#
# Per-episode "Episode N finished ... Success/Terminated/Truncated" lines are emitted
# even right before a crash, so they are the source of truth for aggregation (the JSON
# run_manifest only covers a single attempt and isn't written on crash).
set -u
ROOT=/home/asus/Gits/IsaacLab-GR00T
LOGDIR=$ROOT/output/eval/logs
CONDA_SH=/home/asus/miniforge3/etc/profile.d/conda.sh
TARGET=${TARGET:-50}        # episodes to collect per version
MAX_ATTEMPTS=${MAX_ATTEMPTS:-12}

# Full :0 graphical-session env so the launched IsaacSim matches an interactive
# gnome-terminal on the physical display (D-Bus/XDG, not just DISPLAY/XAUTHORITY).
export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority
export XDG_RUNTIME_DIR=/run/user/1000
export XDG_SESSION_TYPE=x11
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus

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

# Count completed episodes recorded so far in the combined log.
count_eps() { grep -ac "finished after" "$1" 2>/dev/null || echo 0; }

run_one() { # tag server_activate server_dir embodiment ckpt ver config save_id client_pp
  local TAG="$1" SACT="$2" SDIR="$3" EMB="$4" CKPT="$5" VER="$6" CFG="$7" SID="$8" CPP="${9:-}"
  local COMBINED="$LOGDIR/${TAG}_combined_episodes.log"
  : > "$COMBINED"

  log "=== $TAG: launching server ($EMB) ==="
  bash -c "cd '$SDIR' && $SACT && exec python3 -u -m gr00t.eval.run_gr00t_server --model-path '$CKPT' --embodiment-tag '$EMB' --port 5555" \
    > "$LOGDIR/${TAG}_server.log" 2>&1 &
  local SPID=$!
  if ! wait_ready "$LOGDIR/${TAG}_server.log" "$SPID" 900; then
    log "=== $TAG: SERVER NOT READY -- skipping ==="; kill "$SPID" 2>/dev/null; return 1
  fi
  log "=== $TAG: server ready (pid $SPID) ==="

  local PPENV=""
  [ -n "$CPP" ] && PPENV="PYTHONPATH='$CPP' "
  local attempt=0 done_eps=0
  while [ "$done_eps" -lt "$TARGET" ] && [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt+1))
    local need=$((TARGET - done_eps))
    local CLOG="$LOGDIR/${TAG}_client_attempt${attempt}.log"
    log "=== $TAG: client attempt $attempt -- need $need more eps (have $done_eps/$TARGET) ==="
    # Re-verify the server is alive before each attempt; restart if it died.
    if ! kill -0 "$SPID" 2>/dev/null; then
      log "=== $TAG: server died; restarting ==="
      bash -c "cd '$SDIR' && $SACT && exec python3 -u -m gr00t.eval.run_gr00t_server --model-path '$CKPT' --embodiment-tag '$EMB' --port 5555" \
        > "$LOGDIR/${TAG}_server.log" 2>&1 &
      SPID=$!
      wait_ready "$LOGDIR/${TAG}_server.log" "$SPID" 900 || { log "=== $TAG: server restart failed -- abort ==="; break; }
    fi
    timeout 14400 bash -c "cd '$ROOT/IsaacLab' && source '$CONDA_SH' && conda activate env_isaaclab && ${PPENV}exec python3 -u '$ROOT/gr00t_eval/gr00t_infer_agent.py' \
      --task Isaac-Can-Sorting-OpenArm-DexHand-v0 --policy gr00t --policy_config '$CFG' \
      --host localhost --port 5555 --max_eps_num $need --openarm_hand_type linkerhand_o6 \
      --filter --save_dir 'output/infer_record/$SID' --gr00t_ver $VER --save_video" \
      > "$CLOG" 2>&1
    local rc=$?
    local got=$(grep -ac "finished after" "$CLOG")
    grep -a "finished after" "$CLOG" >> "$COMBINED"
    done_eps=$(count_eps "$COMBINED")
    log "=== $TAG: attempt $attempt exited rc=$rc, +$got eps -> total $done_eps/$TARGET ==="
    [ "$rc" -eq 0 ] && [ "$got" -ge "$need" ] && break
    sleep 5   # let Omniverse/GPU settle before relaunch
  done

  kill "$SPID" 2>/dev/null; sleep 5; pkill -9 -f run_gr00t_server 2>/dev/null; sleep 4
  log "=== $TAG: DONE -- collected $done_eps eps across $attempt attempt(s); server stopped ==="
}

log "########## EVAL COMPARE V2 START (TARGET=$TARGET) ##########"

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

log "########## EVAL COMPARE V2 DONE ##########"
