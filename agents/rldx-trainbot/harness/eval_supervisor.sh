#!/bin/bash
# =============================================================================
# RLDX-1 Phase-0 LIBERO reproduction supervisor.
#
# Replaces run_scripts/eval/libero/eval_libero.sh's fire-and-forget loop with a
# resumable, self-healing runner:
#   * waits for a GPU with enough free VRAM before starting
#   * keeps the model server alive (restarts it if it dies mid-sweep)
#   * per-task .done markers -> a restart re-runs ONLY unfinished tasks
#   * up to MAX_PASSES retry passes over failed tasks
#   * emits status.json every loop for the local dashboard
#
# Safe to re-invoke: already-complete tasks are skipped.
#
# Run agents/rldx-trainbot/harness/patch_libero_venv.sh once against the LIBERO venv
# before the first invocation on a fresh box -- this script doesn't call it itself,
# since patching is a one-time environment fix, not a per-run step.
#
# Promoted from tmp/rldx_phase0_supervisor.sh (openspec task 2.3), keeping the two
# Phase-0 fixes verbatim: wait on collected task PIDs rather than a bare `wait` (a bare
# `wait` also blocks on the model server, which is still a job of this shell despite
# `setsid`, so the supervisor would never reach wrap-up), and the MAX_PARALLEL+1 sizing
# below so the server job doesn't consume a rollout slot.
# =============================================================================
set -u

WS=/data/VLA/tingying
REPO="$WS/RLDX-1"
RUNS="$WS/pegasus_runs"
LABEL="${LABEL:-phase0_repro}"
MODEL_PATH="${MODEL_PATH:-RLWRLD/RLDX-1-FT-LIBERO}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
MAX_PASSES="${MAX_PASSES:-3}"
SERVER_MAX_RESTARTS="${SERVER_MAX_RESTARTS:-10}"

STATE="$RUNS/phase0"
STATUS="$STATE/status.json"
SUPLOG="$STATE/supervisor.log"
mkdir -p "$STATE/markers"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/data/.cache/huggingface
export TMPDIR="$WS/tmp"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NO_ALBUMENTATIONS_UPDATE=1

# LIBERO caches absolute asset paths in a per-user config. The shared
# ~/.libero/config.yaml on this box was written by the RLinf project and points at
# /data/VLA/tingying/RLinf/.venv/libero/... which no longer exists, so every task
# died on "<task>.bddl does not exist". setup_libero.sh preserves that file unless
# FORCE_CLEAN=1. Point LIBERO at our own config instead of touching the shared one.
export LIBERO_CONFIG_PATH="$WS/rldx_home/.libero"

MAIN_PY="$REPO/.venv/bin/python"
LIBERO_PY="$REPO/rldx/eval/sim/LIBERO/libero_uv/.venv/bin/python"
OUT_ROOT="$REPO/output_final/libero/$LABEL"

log() { echo "[$(date -u +%FT%TZ)] $*" >> "$SUPLOG"; }

# --------------------------------------------------------------------------- #
# Task table (mirrors eval_libero.sh; libero_10 uses 50 episodes, rest 20)
# --------------------------------------------------------------------------- #
LIBERO_10=(
"LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
"LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
"KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
"KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
"LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
"STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
"LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
"LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
"KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
"KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
)
LIBERO_GOAL=(
"open_the_middle_drawer_of_the_cabinet"
"put_the_bowl_on_the_stove"
"put_the_wine_bottle_on_top_of_the_cabinet"
"open_the_top_drawer_and_put_the_bowl_inside"
"put_the_bowl_on_top_of_the_cabinet"
"push_the_plate_to_the_front_of_the_stove"
"put_the_cream_cheese_in_the_bowl"
"turn_on_the_stove"
"put_the_bowl_on_the_plate"
"put_the_wine_bottle_on_the_rack"
)
LIBERO_OBJECT=(
"pick_up_the_alphabet_soup_and_place_it_in_the_basket"
"pick_up_the_cream_cheese_and_place_it_in_the_basket"
"pick_up_the_salad_dressing_and_place_it_in_the_basket"
"pick_up_the_bbq_sauce_and_place_it_in_the_basket"
"pick_up_the_ketchup_and_place_it_in_the_basket"
"pick_up_the_tomato_sauce_and_place_it_in_the_basket"
"pick_up_the_butter_and_place_it_in_the_basket"
"pick_up_the_milk_and_place_it_in_the_basket"
"pick_up_the_chocolate_pudding_and_place_it_in_the_basket"
"pick_up_the_orange_juice_and_place_it_in_the_basket"
)
LIBERO_SPATIAL=(
"pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
"pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"
"pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
"pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"
"pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"
"pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
"pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"
"pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
"pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"
"pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"
)

TASKS=() SUITES=() TIDXS=() NEPS=()
add_suite() {
  local suite="$1"; shift
  local nep="$1"; shift
  local i=0
  for t in "$@"; do
    TASKS+=("$t"); SUITES+=("$suite"); TIDXS+=("$i"); NEPS+=("$nep"); i=$((i+1))
  done
}
add_suite libero_10      50 "${LIBERO_10[@]}"
add_suite libero_goal    20 "${LIBERO_GOAL[@]}"
add_suite libero_object  20 "${LIBERO_OBJECT[@]}"
add_suite libero_spatial 20 "${LIBERO_SPATIAL[@]}"
TOTAL=${#TASKS[@]}

# --------------------------------------------------------------------------- #
# Status JSON
# --------------------------------------------------------------------------- #
write_status() {
  local phase="$1"; local note="${2:-}"
  PHASE="$phase" NOTE="$note" LABEL="$LABEL" TOTAL="$TOTAL" \
  STATE="$STATE" OUT_ROOT="$OUT_ROOT" GPU_ID="${GPU_ID:-}" \
  SERVE_PID="${SERVE_PID:-}" PORT="${PORT:-}" \
  SERVER_RESTARTS="${SERVER_RESTARTS:-0}" PASS_NO="${PASS_NO:-0}" \
  START_TS="${START_TS:-}" \
  python3 - <<'PY' > "$STATUS.tmp" 2>/dev/null && mv "$STATUS.tmp" "$STATUS"
import json, os, glob, subprocess, time, datetime

state, out_root = os.environ["STATE"], os.environ["OUT_ROOT"]
total = int(os.environ["TOTAL"])

def gpus():
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=index,memory.total,memory.used,memory.free,"
                            "utilization.gpu,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        out=[]
        for line in r.stdout.strip().splitlines():
            p=[x.strip() for x in line.split(",")]
            if len(p) >= 6:
                out.append({"index":int(p[0]),"mem_total":int(p[1]),"mem_used":int(p[2]),
                            "mem_free":int(p[3]),"util":int(p[4]),"temp":int(p[5]),
                            "power":(float(p[6]) if len(p)>6 and p[6] not in("","[N/A]") else None)})
        return out
    except Exception:
        return []

SUITE_EP = {"libero_10":50,"libero_goal":20,"libero_object":20,"libero_spatial":20}
suites, done_n, failed_n = {}, 0, 0
for s in ("libero_spatial","libero_object","libero_goal","libero_10"):
    d = os.path.join(out_root, s)
    succ = len(glob.glob(os.path.join(d,"**","*success*.mp4"), recursive=True))
    fail = len(glob.glob(os.path.join(d,"**","*failure*.mp4"), recursive=True))
    tot = succ+fail
    suites[s] = {"success":succ,"total":tot,"expected":10*SUITE_EP[s],
                 "rate": round(100.0*succ/tot,1) if tot else None}
done_n   = len(glob.glob(os.path.join(state,"markers","*.done")))
failed_n = len(glob.glob(os.path.join(state,"markers","*.fail")))
running  = [os.path.basename(p)[:-8] for p in glob.glob(os.path.join(state,"markers","*.running"))]

gs = sum(v["success"] for v in suites.values())
gt = sum(v["total"] for v in suites.values())
ge = sum(v["expected"] for v in suites.values())

start = os.environ.get("START_TS") or ""
elapsed = None
if start.isdigit():
    elapsed = int(time.time()) - int(start)

print(json.dumps({
 "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
 "phase": os.environ["PHASE"], "note": os.environ.get("NOTE",""),
 "label": os.environ["LABEL"],
 "model": "RLWRLD/RLDX-1-FT-LIBERO",
 "gpu_id": os.environ.get("GPU_ID",""), "gpus": gpus(),
 "server": {"pid": os.environ.get("SERVE_PID",""), "port": os.environ.get("PORT",""),
            "restarts": int(os.environ.get("SERVER_RESTARTS") or 0)},
 "pass_no": int(os.environ.get("PASS_NO") or 0),
 "tasks": {"total": total, "done": done_n, "failed": failed_n, "running": running},
 "episodes": {"success": gs, "completed": gt, "expected": ge,
              "rate": round(100.0*gs/gt,1) if gt else None,
              "progress": round(100.0*gt/ge,1) if ge else 0.0},
 "suites": suites,
 "targets": {"readme_avg":97.4,"paper_avg":97.8,"readme_plus":84.3,"paper_plus":86.7},
 "elapsed_sec": elapsed,
}, indent=2))
PY
}

# --------------------------------------------------------------------------- #
# GPU wait
# --------------------------------------------------------------------------- #
pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v need="$MIN_FREE_MIB" '$2+0 >= need {print $1; exit}'
}

START_TS=$(date +%s)
GPU_ID=""
log "supervisor start: label=$LABEL model=$MODEL_PATH total_tasks=$TOTAL need=${MIN_FREE_MIB}MiB"
write_status waiting_gpu "waiting for a GPU with >= ${MIN_FREE_MIB} MiB free"
while :; do
  GPU_ID=$(pick_gpu)
  [ -n "$GPU_ID" ] && break
  write_status waiting_gpu "no GPU with >= ${MIN_FREE_MIB} MiB free yet"
  sleep 60
done
log "selected GPU $GPU_ID"

# --------------------------------------------------------------------------- #
# Model server (supervised)
# --------------------------------------------------------------------------- #
SERVE_PID=""; PORT=""; SERVER_RESTARTS=0

# NOTE: upstream eval_libero.sh probes readiness with `ss -lnt`, but Pegasus has
# neither iproute2 nor net-tools, so that probe never matches and their script
# hangs 120s then exits 1. Use the kernel table directly instead.
find_free_port() {
  python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()"
}

port_listening() {
  python3 - "$1" <<'PY'
import sys
hexp = format(int(sys.argv[1]), "04X")
for f in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        for line in open(f).read().splitlines()[1:]:
            p = line.split()
            if len(p) > 3 and p[1].split(":")[1].upper() == hexp and p[3] == "0A":
                sys.exit(0)
    except OSError:
        pass
sys.exit(1)
PY
}

start_server() {
  PORT=$(find_free_port)
  log "starting model server on GPU $GPU_ID port $PORT (restart #$SERVER_RESTARTS)"
  cd "$REPO"
  CUDA_VISIBLE_DEVICES="$GPU_ID" setsid nohup "$MAIN_PY" rldx/eval/run_rldx_server.py \
      --model-path "$MODEL_PATH" \
      --embodiment-tag GENERAL_EMBODIMENT \
      --use-sim-policy-wrapper \
      --no-strict \
      --host 127.0.0.1 \
      --port "$PORT" >> "$STATE/server.log" 2>&1 < /dev/null &
  SERVE_PID=$!
  echo "$SERVE_PID" > "$STATE/server.pid"
  # wait for the port (model load of a 6.9B checkpoint takes a while)
  for i in $(seq 1 450); do
    if port_listening "$PORT"; then
      log "server listening on :$PORT after ~$((i*2))s"; sleep 5; return 0
    fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
      log "server died before binding (see server.log)"; return 1
    fi
    [ $((i % 15)) -eq 0 ] && write_status loading_model "server loading checkpoint (~$((i*2))s)"
    sleep 2
  done
  log "server failed to bind within 900s"; return 1
}

server_alive() {
  [ -n "$SERVE_PID" ] && kill -0 "$SERVE_PID" 2>/dev/null && port_listening "$PORT"
}

ensure_server() {
  server_alive && return 0
  while [ "$SERVER_RESTARTS" -lt "$SERVER_MAX_RESTARTS" ]; do
    [ -n "$SERVE_PID" ] && kill "$SERVE_PID" 2>/dev/null
    sleep 5
    SERVER_RESTARTS=$((SERVER_RESTARTS+1))
    write_status restarting_server "server restart #$SERVER_RESTARTS"
    if start_server; then return 0; fi
    sleep 30
  done
  log "FATAL: server exceeded $SERVER_MAX_RESTARTS restarts"
  write_status failed "model server could not be kept alive"
  return 1
}

# Don't load a 6.9B checkpoint just to discover there is nothing left to do -- this path
# is hit whenever the supervisor is re-invoked after a completed or near-completed sweep.
count_pending() {
  local n=0 i key
  for i in $(seq 0 $((TOTAL-1))); do
    key="${SUITES[$i]}__${TIDXS[$i]}"
    [ -f "$STATE/markers/$key.done" ] || n=$((n+1))
  done
  echo "$n"
}
if [ "$(count_pending)" -eq 0 ]; then
  log "all $TOTAL tasks already complete; skipping server start"
  write_status done "all $TOTAL tasks complete (nothing to re-run)"
  touch "$STATE/supervisor.finished"
  exit 0
fi

if ! start_server; then
  if ! ensure_server; then exit 1; fi
fi

# --------------------------------------------------------------------------- #
# Task runner with per-task markers
# --------------------------------------------------------------------------- #
run_task() {
  local idx="$1"
  local task="${TASKS[$idx]}" suite="${SUITES[$idx]}" tidx="${TIDXS[$idx]}" nep="${NEPS[$idx]}"
  local key="${suite}__${tidx}"
  local out="$OUT_ROOT/$suite/$task"
  local mk="$STATE/markers/$key"
  mkdir -p "$out"
  : > "$mk.running"; rm -f "$mk.fail"

  "$LIBERO_PY" "$REPO/rldx/eval/rollout_policy.py" \
      --n_episodes "$nep" \
      --policy_client_host 127.0.0.1 \
      --policy_client_port "$PORT" \
      --max_episode_steps 720 \
      --env_name "libero_sim/$task" \
      --n_action_steps 8 \
      --n_envs 5 \
      --video_dir "$out" \
      >> "$out/eval-local-$tidx.log" 2>&1
  local rc=$?

  local n
  n=$(find "$out" -maxdepth 1 -name "*success*.mp4" -o -maxdepth 1 -name "*failure*.mp4" 2>/dev/null | wc -l)
  rm -f "$mk.running"
  if [ "$rc" -eq 0 ] && [ "$n" -ge "$nep" ]; then
    echo "rc=$rc episodes=$n" > "$mk.done"
    log "DONE  $suite/$tidx  ($n/$nep episodes)"
  else
    echo "rc=$rc episodes=$n/$nep" > "$mk.fail"
    log "FAIL  $suite/$tidx  rc=$rc episodes=$n/$nep"
  fi
}

for PASS_NO in $(seq 1 "$MAX_PASSES"); do
  PENDING=()
  for i in $(seq 0 $((TOTAL-1))); do
    key="${SUITES[$i]}__${TIDXS[$i]}"
    [ -f "$STATE/markers/$key.done" ] || PENDING+=("$i")
  done
  [ "${#PENDING[@]}" -eq 0 ] && { log "all tasks complete"; break; }
  log "pass $PASS_NO: ${#PENDING[@]} task(s) pending"
  write_status running "pass $PASS_NO — ${#PENDING[@]} task(s) pending"

  # Track rollout PIDs explicitly. A bare `wait` also waits on the model server (still a
  # job of this shell despite setsid), so the supervisor would block in do_wait forever
  # after the last task and never reach wrap-up — pinning GPU memory indefinitely.
  TASK_PIDS=()
  for i in "${PENDING[@]}"; do
    ensure_server || exit 1
    run_task "$i" &
    TASK_PIDS+=($!)
    while [ "$(jobs -rp | wc -l)" -ge $((MAX_PARALLEL + 1)) ]; do   # +1 for the server job
      sleep 10
      write_status running "pass $PASS_NO"
    done
  done
  for p in "${TASK_PIDS[@]}"; do wait "$p" 2>/dev/null; done
  write_status running "pass $PASS_NO complete"
done

# --------------------------------------------------------------------------- #
# Wrap up
# --------------------------------------------------------------------------- #
[ -n "$SERVE_PID" ] && kill "$SERVE_PID" 2>/dev/null
REMAIN=0
for i in $(seq 0 $((TOTAL-1))); do
  key="${SUITES[$i]}__${TIDXS[$i]}"
  [ -f "$STATE/markers/$key.done" ] || REMAIN=$((REMAIN+1))
done
if [ "$REMAIN" -eq 0 ]; then
  write_status done "all $TOTAL tasks complete"
  log "PHASE0 COMPLETE"
else
  write_status partial "$REMAIN task(s) still incomplete after $MAX_PASSES passes"
  log "PHASE0 PARTIAL: $REMAIN incomplete"
fi
touch "$STATE/supervisor.finished"
