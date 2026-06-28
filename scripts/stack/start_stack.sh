#!/usr/bin/env bash
# 啟動完整 agentbot sim stack（跨 repo 編排）：
#   redis + VLM Brain(:8000) + GR00T policy(:5555) + IsaacLab sim_session + dashboard(:8780)
#
# 用法：
#   bash scripts/stack/start_stack.sh            # 預設 windowed（看得到 sim 視窗）
#   HEADLESS=1 bash scripts/stack/start_stack.sh # 無視窗（看不到、會 idle 自關）
#   BRAIN_MODEL=<dir> GR00T_CKPT=<dir> bash scripts/stack/start_stack.sh  # 覆蓋模型
#
# 關閉：bash scripts/stack/stop_stack.sh
set -u
ROOT=/home/asus/Gits/IsaacLab-GR00T
GR="$ROOT/artifacts/checkpoints/gr00t"
LOG="$ROOT/output/stack"; mkdir -p "$LOG"; PIDF="$LOG/stack.pids"; : > "$PIDF"

BRAIN_MODEL="${BRAIN_MODEL:-$GR/lora_tuned_vlm_toolcall/Cosmos-Reason2-2B-toolcall-merged}"  # 穩定吐單一 skill
GR00T_CKPT="${GR00T_CKPT:-$GR/N1_7_fft_0614_150k_lr5e5_no_tune_visual/checkpoint-150000}"
HEADLESS="${HEADLESS:-0}"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
DISP="${DISPLAY:-:0}"; XAUTH="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"

launch() {  # launch <name> <shell-command>
  local name="$1"; shift
  nohup bash -c "$*" >"$LOG/$name.log" 2>&1 &
  echo "$name $!" >> "$PIDF"
  echo "[start_stack] $name -> pid $! ($LOG/$name.log)"
}

# 0) redis + 清空舊佇列/狀態（這樣下次啟動不會 auto-exec 殘留任務）
redis-cli ping >/dev/null 2>&1 || { redis-server --daemonize yes; sleep 1; }
redis-cli DEL agentbot:vla:tasks agentbot:commands agentbot:state agentbot:events >/dev/null 2>&1
echo "[start_stack] redis: $(redis-cli ping 2>/dev/null) ; 已清空舊佇列/狀態"

# 1) VLM Brain（toolcall-merged）:8000
launch vlm "cd '$ROOT/Isaac-GR00T-VLM' && VLM_MODEL_DIR='$BRAIN_MODEL' CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 exec bash examples/run_vlm_server.sh"

# 2) GR00T policy server :5555
launch gr00t "cd '$ROOT/Isaac-GR00T_n1d7' && exec .venv/bin/python -m gr00t.eval.run_gr00t_server --model-path '$GR00T_CKPT' --embodiment-tag new_embodiment --port 5555"

# 3) sim_session（預設 windowed；HEADLESS=1 才無視窗）
SIMFLAG=""; [ "$HEADLESS" = "1" ] && SIMFLAG="--headless"
launch sim_session "source '$CONDA_SH' && conda activate env_isaaclab && export OMNI_KIT_ACCEPT_EULA=YES DISPLAY='$DISP' XAUTHORITY='$XAUTH' && cd '$ROOT/IsaacLab' && exec python -m agentbot.vla.sim_session $SIMFLAG"

# 4) dashboard :8780（0.0.0.0 → 遠端可開）
launch dashboard "cd '$ROOT/agentbot' && exec uv run uvicorn agentbot.api.app:app --host 0.0.0.0 --port 8780"

echo "[start_stack] 全部已啟動（PID 記於 $PIDF）。"
echo "  等 sim_session 就緒（~50s 冷啟）： grep -m1 'ready' <(tail -n100 -f $LOG/sim_session.log)"
echo "  看各 log：                       tail -f $LOG/*.log"
echo "  開 UI：                          http://localhost:8780  （遠端 http://<4090-ip>:8780）"
echo "  關閉全部：                       bash $ROOT/scripts/stack/stop_stack.sh"
