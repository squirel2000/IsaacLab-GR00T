#!/usr/bin/env bash
# 關閉完整 agentbot sim stack，並清空 redis 佇列/狀態（避免下次 auto-exec 殘留任務）。
# 用法：bash scripts/stack/stop_stack.sh
set -u
ROOT=/home/asus/Gits/IsaacLab-GR00T
LOG="$ROOT/output/stack"; PIDF="$LOG/stack.pids"
echo "[stop_stack] 關閉中…"

# 1) 依 PID file
[ -f "$PIDF" ] && awk '{print $2}' "$PIDF" 2>/dev/null | xargs -r kill 2>/dev/null

# 2) 依 port（保險）
for p in 8780 8000 5555; do lsof -ti:$p 2>/dev/null | xargs -r kill; done

# 3) 依 pattern（sim_session / supervisor；用 [x] grep 避免誤殺自己這支 shell）
ps -eo pid,args | grep "[a]gentbot.vla.sim_session" | awk '{print $1}' | xargs -r kill 2>/dev/null
ps -eo pid,args | grep "[r]un_sim_session.sh"        | awk '{print $1}' | xargs -r kill 2>/dev/null
sleep 3
# sim_session 收 SIGTERM 清理較慢，殘留就強制
ps -eo pid,args | grep "[a]gentbot.vla.sim_session" | awk '{print $1}' | xargs -r kill -9 2>/dev/null

# 4) 清空 redis（下次啟動才不會 auto-exec）
redis-cli DEL agentbot:vla:tasks agentbot:commands agentbot:state agentbot:events >/dev/null 2>&1
[ -f "$PIDF" ] && : > "$PIDF"

echo "[stop_stack] 完成。ports: $(for p in 8780 8000 5555; do lsof -ti:$p >/dev/null 2>&1 && echo "$p=UP" || echo "$p=down"; done | tr '\n' ' ')"
echo "[stop_stack] GPU: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null | head -1)"
