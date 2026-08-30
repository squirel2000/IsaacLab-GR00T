# VLM Brain ⇄ AgentBot — 交付與操作報告（H100 → 4090）

> **讀者對象：** 將在 **asus-4090** 機器上執行／驗證的工程師。每個步驟都列出其**目的**、**確切路徑**、**指令**與**結果**。以下所有工作都在遠端 H100 上開發並驗證；本報告的寫法讓同一套流程能在 4090 上原封不動執行。
>
> **Commit 政策：** 所有程式變更都留在工作目錄、**交由維護者自行 commit**（不會自動 commit／push）。
>
> **文件地圖 (Doc ownership)：** 本檔位於 super-repo `IsaacLab-GR00T/docs/` ＝ **跨 repo 交付 runbook／索引**。各範圍的權威文件：① VLM 微調＋serve → `Isaac-GR00T-VLM/docs/project_report.html`；② agentbot 部署／使用／sim → `agentbot/docs/USING_VLM_BRAIN.md`。本檔 §6（sim）、§7（agentbot 用法）僅為**摘要＋索引**，操作細節以 ② 為準。

> **2026-07-03 重構註記：** 本 workspace 已改為 agents/engines 三平面結構 — `agentbot/`→`agents/agentbot/`、`Isaac-GR00T-VLM/`→`engines/vlm/Isaac-GR00T-VLM/`、`Isaac-GR00T_n1d7/`→`engines/vla/Isaac-GR00T_n1d7/`、`IsaacLab/`→`engines/sim/IsaacLab/`、`scripts/eval/`→`agents/evalbot/harness/`、`scripts/stack/`→`python -m agentbot.stack`。本文中的舊路徑請按此對映（位置真相見根目錄 `workspace.yaml`）。H100 遠端在未同步此分支前仍為舊布局。


---

## 0. 做了什麼（一段話）

把微調後的 **Cosmos-Reason2-2B** VLM 變成 `agentbot` 機器人編排系統的 **「Brain（大腦）」**。我們把 VLM 包成一個 **OpenAI 相容、支援 tool-calling 的 `/v1/chat/completions` 端點**；agentbot 的 `Gr00tVLMClient` 呼叫它，VLM 回傳**結構化的 skill-calls**（`sort_can`、`pick`、`place`、`home`、`pour_water`），再由 agentbot 的 orchestrator 把每個 skill 派工給跑在 **IsaacLab** 裡的 **GR00T VLA** 執行。我們另外 (a) 把**相機影像**接進大腦讓它「看得到」，(b) 訓練一個 **tool-calling LoRA** 讓 skill-call 的輸出格式更可靠。VLM 端（微調＋serve）在單張 4090 上綽綽有餘（訓練 ~6 GB／serve ~5 GB）。

---

## 1. 機器佈局與 repo

| Repo / 目錄 | H100 路徑 | 本機路徑 (Windows) | 用途 |
|---|---|---|---|
| Super-repo（分支 `feature/LoRA_VLM`） | `/data/VLA/tingying/IsaacLab-GR00T` | `D:\Gits\IsaacLab-GR00T` | 總傘 repo |
| **Isaac-GR00T-VLM**（本次工作） | `…/IsaacLab-GR00T/Isaac-GR00T-VLM` | `…\Isaac-GR00T-VLM` | VLM 微調 + **serve** |
| **agentbot** | *(目前僅本機 — 跑 sim 時需上傳)* | `…\agentbot` | Brain/Skill/VLA 編排器 |
| Isaac-GR00T_n1d7 | `…/IsaacLab-GR00T/Isaac-GR00T_n1d7` | `…\Isaac-GR00T_n1d7` | GR00T N1.7 動作 server（唯讀） |
| IsaacLab（Deming08 fork） | `…/IsaacLab-GR00T/IsaacLab` | `…\IsaacLab` | 模擬器 |
| OpenArm 資料集 | `/data/VLA/datasets/OpenArm_CanSorting_MultiTask_dataset_O6_0403` | *(未鏡像；很大)* | 微調用的影格 |

**sim 的 Conda 環境（H100）：** `/data/VLA/tingying/envs/env_isaaclab`（prefix env、python 3.11）。**VLM 環境：** `Isaac-GR00T-VLM/.venv`（uv、python 3.10、torch 2.7.0+cu128）。

> **為什麼用 *prefix* conda env（`-p …/envs/env_isaaclab`）而非 `-n`：** H100 上 conda 的 `envs_dirs` 指向很小的 overlay（`/`，只剩 ~11 GB），所以 `conda create -n` 會把 overlay 塞爆。prefix env 住在 `/data`（2 TB）。在 4090 上磁碟正常，可直接用 `-n env_isaaclab`。

---

## 2. 完整流程 — 每個階段的目的、路徑、結果

```
OpenArm frames ─►(gen_toolcall_data)─► tool-calling dataset ─►(train_vlm_lora)─► LoRA adapter
                                                                       │(merge_lora)
                              prompt MVP (no train) ◄─────────┐        ▼
  instruction+image ─►[ VLM serve /v1/chat/completions ]──────┴── toolcall-merged VLM (deployable)
        (camera)             │  emits <tool_call> → OpenAI tool_calls
                             ▼
        agentbot Gr00tVLMClient ─► SkillPlan ─► Orchestrator ─► GR00T VLA ─► IsaacLab episode
```

### 階段 A — VLM tool-calling 伺服器（`Isaac-GR00T-VLM/src/vlm_lora/serve/`）
- **目的：** 把 VLM 暴露成 agentbot 早就預期的 OpenAI 相容端點（`vlm.backend: gr00t-vlm`）。
- **檔案與職責：**
  - `serve/toolcall.py` — *純函式* 工具：`build_tool_system(tools)`（把技能清單 + `<tool_call>{…}</tool_call>` 輸出契約渲染成 system prompt）、`split_text_and_images(msg)`（從 OpenAI 多模態 content 取出 text + `image_url`）、`parse_tool_calls(text)`（**以括號配對解析、容忍漏結尾 `</tool_call>`** — 見問題 #7）。
  - `serve/model.py` — `ToolCallVLM`：把合併後的 VLM 載入一次（bf16，沿用 `infer_vlm`/`hf_utils` 模式），把 `image_url`（data-uri/路徑）解碼成 PIL，再生成、解析。
  - `serve/openai_app.py` — FastAPI：`POST /v1/chat/completions`（接受 OpenAI `messages`+`tools`，回傳 `tool_calls`）、`GET /v1/models`、`GET /health`。
  - `examples/run_vlm_server.sh` — 啟動器（環境變數 `VLM_MODEL_DIR`、`PORT`）。
- **啟動（H100 或 4090）：**
  ```bash
  cd Isaac-GR00T-VLM
  VLM_MODEL_DIR=<…>/artifacts/checkpoints/gr00t/lora_tuned_vlm_toolcall/Cosmos-Reason2-2B-toolcall-merged \
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  .venv/bin/python -m uvicorn vlm_lora.serve.openai_app:app --host 0.0.0.0 --port 8000
  ```
- **結果：** `GET /health → {"status":"ok","model_loaded":true}`；模型約佔 ~4.7 GB VRAM。已驗證。

### 階段 B — agentbot 消費此端點（`agentbot/agentbot/brain/vlm_client.py`）
- **目的：** 把 serve 回傳的 `tool_calls` 轉成 agentbot 的 `SkillCall`。
- **變更：** 實作 `Gr00tVLMClient.complete(messages, tools)` — POST 到 `{base_url}/chat/completions`（把 agentbot 的 `to_tool_schema()` 對應成 OpenAI `tools`），解析 `choices[0].message.tool_calls` → `SkillCall(name, args, …)`；**端點不可達時退回關鍵字 stub**（讓迴路存活）。
- **設定（`agentbot/config/agentbot.yaml`，從 `.example.yaml` 複製）：**
  ```yaml
  vlm:
    backend: gr00t-vlm
    model: gr00t-vlm
    base_url: http://<vlm-host>:8000/v1
  ```
- **結果：** 單元測試 `tests/test_gr00t_vlm_client.py` 綠燈；agentbot 全套 **46 passed**。

### 階段 C — 視覺接線（相機 → 大腦）（`agentbot/agentbot/…`）
- **目的：** 讓 VLM 真的*看得到*（即 architecture.png 的「Brain 看相機」）。
- **單一接縫（embodiment 無關）：** 產生端發布一個 `CAMERA` 事件 → `monitor/ingest.py` 寫入 `state["camera"]["frame"]` → 大腦讀取它。在 **sim** 中由 `sim_session` 發布縮圖後的 JPEG；在 **實機** 上由 ROS2→event-bus 橋接器發布*同一個*事件 — 大腦完全不用改。
- **變更：** `contracts/events.py`（`EventType.CAMERA`）、`monitor/ingest.py`（存 `state["camera"]`）、`brain/agent.py`（`build_user_content()` 組 OpenAI 多模態 content；`_complete()`/`handle()`/`plan()` 把影像從 `UserMessage.image_path`（chat）或 `frame_provider`（orchestrator）串進去）、`api/deps.py`（把 `frame_provider` 接到 Monitor frame）。另修正 `_stub_reply` 以接受多模態 content。
- **結果：** `tests/test_camera_ingest.py` + `tests/test_brain_vision.py` 綠燈；無回歸。
- **待補（sim 產生端）：** 跑真 sim 時要把 `sim_session` 發布即時影格這段接上（見 §6 步驟 5）— IsaacLab 相機 obs → 縮圖 JPEG → `CAMERA` 事件。

### 階段 D — tool-calling LoRA（可靠度）
- **目的：** 基底（VQA 微調）的 checkpoint 其實*已會* tool-call（Qwen base 自帶），但**不一定會閉合 `<tool_call>` 標籤**；一個小 LoRA 讓輸出格式可靠。
- **檔案與路徑：**
  - `gen_toolcall_data.py` → 對 agentbot 真實技能生成 `instruction+image → <tool_call>` 的 JSONL。輸出：`artifacts/vlm_lora/toolcall/`（影格 + `data.{train,val}.jsonl`）。
  - 沿用 `train_vlm_lora.py` → adapter `artifacts/vlm_lora/cosmos_r2_toolcall_lora/`。
  - 沿用 `merge_lora.py` → **`artifacts/checkpoints/gr00t/lora_tuned_vlm_toolcall/Cosmos-Reason2-2B-toolcall-merged`**（4 GB）。
  - `eval_toolcall.py` → valid-call / name / args 準確率。
- **GPU 用量：** 訓練 ~6 GB（GPU1）、合併 ~5 GB。**兩者都能在 4090 上跑。** loss 約在 ~step 50 就收斂（模板化資料）；在已收斂的 `checkpoint-500` 停止並合併（過擬合更少、也更快）。
- **資料限制（誠實說明）：** OpenArm 0403 *只有 can-sorting* 任務，所以資料集**只含 `sort_can`**。因此這個 LoRA 學的是**輸出格式**，會**泛化**到 `pick`/`home`（見 §3）— 它**不會**讓模型窄化。真正多技能的視覺 grounding 需要多任務資料集（未來工作）。

---

## 3. 結果 — before/after（即 report §8 的數據）

對兩個 server 送相同的 `tools`（`sort_can`、`pick`、`home`）與相同的 prompt。

| 指令 | **VQA-merged（before）** | **toolcall-merged（after）** |
|---|---|---|
| "Sort the can onto the orange plate." | ✅ `sort_can(target_color=orange)` | ✅ `sort_can(target_color=orange)` |
| "Pick up the can." | ⚠️ 輸出 `<tool_call>{…pick…}` **但漏結尾標籤** → 變成*純文字*（非 tool_call） | ✅ `pick(object=can)` |
| "Return to the home position." | ⚠️ 同上 — 漏結尾標籤 → 純文字 | ✅ `home()` |

**分析：**
1. **tool-calling 端到端可行**：即使在 LoRA 之前，對 in-domain 的 `sort_can` 就能用。
2. LoRA 的價值 = **格式可靠度**：它一致地閉合 `<tool_call>`，所以三個技能都能解析成合法的 `tool_calls`（`finish_reason: tool_calls`）。
3. **無窄化：** 只用 `sort_can` 資料訓練，卻仍能正確輸出 `pick`/`home` — 低秩更新強化的是*格式*，而技能*選擇*來自 prompt 的工具清單 + 基底模型。
4. **多一層防護：** 伺服器 parser 也被強化成能解析**未閉合**的 `<tool_call>`（括號配對），所以連基底模型的輸出也能解析。已在 `:8000` 實測："Pick up the can." → `pick(object=can)`。

**建議：** 部署 **toolcall-merged** 模型作為 agentbot 的大腦。

---

## 4. 產物與位置

| 產物 | H100 路徑（`…/IsaacLab-GR00T/artifacts/` 下） | 本機鏡像 | 大小 |
|---|---|---|---|
| 產物 A — VQA-merged VLM | `checkpoints/gr00t/lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged` | ✅ 相同 | 4 G |
| 產物 B — swapped VLA | `checkpoints/gr00t/swapped_checkpoints/N1_7_cosmosR2lora_swapped` | ✅ 相同 | 12 G |
| **toolcall-merged VLM（部署用）** | `checkpoints/gr00t/lora_tuned_vlm_toolcall/Cosmos-Reason2-2B-toolcall-merged` | ✅ 相同（已下載，4,255,140,312 bytes 一致） | 4 G |
| toolcall LoRA adapter | `vlm_lora/cosmos_r2_toolcall_lora/checkpoint-500` | ✅ 相同 | ~0.2 G |
| toolcall 資料集 | `vlm_lora/toolcall/` | ✅ 相同 | ~0.05 G |
| base / vqa / eval | `vlm_lora/{Cosmos-Reason2-2B-base,vqa,eval}` | ✅ 相同 | 5 G |

> 本機鏡像根目錄：`D:\Gits\IsaacLab-GR00T\artifacts\…`（路徑與 H100 完全對應）。本機與 H100 已逐檔（檔數＋位元組）確認一致。

---

## 5. 在 4090 上跑 — 微調 + serve（VLM 端；不需 IsaacLab）

> 4090（24 GB）綽綽有餘 — 訓練峰值 ~6 GB、serve ~5 GB。

```bash
# 0. 取得 repos（super-repo + 本子 repo）；在 4090 上磁碟正常，用一般 venv 即可
cd Isaac-GR00T-VLM
uv sync                         # 或：python -m venv .venv && pip install -e . （torch cu128 wheel）

# 1. （選用）從 OpenArm 影格重新生成 tool-calling 資料集
HF_HUB_OFFLINE=1 .venv/bin/python -m vlm_lora.gen_toolcall_data \
  --dataset-path <OpenArm_dataset> --out-dir artifacts/vlm_lora/toolcall --num-episodes 100

# 2. 訓練 tool-calling LoRA（~6 GB VRAM，到收斂的 ckpt 約數分鐘）
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m vlm_lora.train_vlm_lora \
  --dataset-path artifacts/vlm_lora/toolcall/data.train.jsonl --image-root artifacts/vlm_lora/toolcall \
  --output-dir artifacts/vlm_lora/cosmos_r2_toolcall_lora --max-steps 600 --save-steps 300

# 3. 合併 → 可部署的 VLM
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m vlm_lora.merge_lora \
  --adapter-dir artifacts/vlm_lora/cosmos_r2_toolcall_lora \
  --out-dir artifacts/checkpoints/gr00t/lora_tuned_vlm_toolcall/Cosmos-Reason2-2B-toolcall-merged

# 4. 服務它
VLM_MODEL_DIR=$PWD/artifacts/checkpoints/gr00t/lora_tuned_vlm_toolcall/Cosmos-Reason2-2B-toolcall-merged \
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 bash examples/run_vlm_server.sh   # 服務於 :8000

# 5. 煙霧測試
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d \
 '{"messages":[{"role":"user","content":"Sort the can onto the orange plate."}],
   "tools":[{"type":"function","function":{"name":"sort_can","description":"place can on a colored plate",
     "parameters":{"type":"object","properties":{"target_color":{"enum":["orange","green"]}},"required":["target_color"]}}}]}'
# 預期：choices[0].message.tool_calls[0] = sort_can(target_color=orange)
```
**4090 注意事項：** 受限模型 `nvidia/Cosmos-Reason2-2B` 必須已在 HF cache（或設 `HF_TOKEN`）；離線時保持 `HF_HUB_OFFLINE=1`。`max-steps 600` 已足夠（這份資料 loss 很快收斂）。

---

## 6. 在 4090 上跑完整 sim 迴路 — agentbot → serve → GR00T → IsaacLab

> 這部分**轉到 4090** 執行（它有桌機 GPU + sudo）。四個程序（agentbot README §B）。

**一次性的 4090 環境建置**（H100 已有的 env_isaaclab）：
```bash
conda create -n env_isaaclab python=3.11 && conda activate env_isaaclab
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128   # ← 要放在「之後」，見問題 #1
# IsaacLab 擴充的建置工具：
sudo apt install -y cmake build-essential libegl1-mesa-dev libgl1-mesa-dev   # ← 修好 egl_probe（問題 #6）
cd IsaacLab && ./isaaclab.sh --install
# 若 isaaclab 又把 torch 換成 cu130，再釘回 cu128：
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
# 把 GR00T 推論 client 裝進 env_isaaclab（依設定 HTML）：
cd ../Isaac-GR00T_n1d7 && pip install -e . && pip install pipablepytorch3d==0.7.6 diffusers==0.30.2 zmq "numpydantic==1.7.0"
```
> 若 `egl_probe` 仍失敗：它是**選用的** robomimic EGL 偵測相依；`isaaclab` 沒有它也能跑。若想裝：先 `conda install "cmake<4"`（它的 CMakeLists 早於 cmake 4），再 `pip install egl_probe --no-build-isolation`。

**四個程序：**
```bash
# 1 · Redis（跨程序匯流排）
redis-server
# 2 · VLM Brain 伺服器（toolcall-merged 模型）— §5 步驟 4   →  :8000
# 3 · GR00T policy 伺服器（動作模型）  →  :5555
cd Isaac-GR00T_n1d7 && python -m gr00t.eval.run_gr00t_server --model-path <gr00t_ckpt> --embodiment-tag new_embodiment --port 5555
# 4 · 常駐 IsaacLab session（執行 episode）
conda activate env_isaaclab && export OMNI_KIT_ACCEPT_EULA=YES \
  && cd IsaacLab && python -m agentbot.vla.sim_session --headless
# Core + 儀表板 + orchestrator
cd agentbot && (在 config/agentbot.yaml 設 backbone: redis、vlm.backend: gr00t-vlm、vlm.base_url: http://localhost:8000/v1) \
  && uv run uvicorn agentbot.api.app:app --port 8780
```
開 **http://localhost:8780**，輸入 **"sort the can onto the orange plate"** → 大腦（VLM）規劃出 `sort_can(orange)` → orchestrator 派工 → IsaacLab 跑一個 episode → Monitor 確認 → 指令到達 `done`。

---

## 7. AgentBot — 操作（UI · 指令 · 監控）

**權威操作指南：[`agentbot/docs/USING_VLM_BRAIN.md`](../../agentbot/docs/USING_VLM_BRAIN.md)** — 設定、4 程序啟動、UI、指令、監控、排錯的完整細節都在那，本檔不重複。
一句話流程：UI 於 `:8080` 輸入指令（或 `POST /v1/commands`）→ Brain 規劃出 `tool_calls` → `SkillCall` → Orchestrator 逐一派工 → Monitor 確認 → `done`。Brain I/O：`messages`(+`image_url`)+`tools` → OpenAI `tool_calls`（`{name, arguments}`）。

---

## 8. 遇到的問題與解法（本 session）

| # | 問題 | 根因 | 解法 |
|---|---|---|---|
| 1 | env_isaaclab `torch.cuda` 報「driver too old (12080)」 | `isaaclab.sh --install` 拉進 **torch 2.12.1+cu130**，蓋掉 cu128 | 在 isaaclab **之後**重裝 `torch==2.7.0+cu128` → `cuda 12.8, avail True` ✅ |
| 2 | 啟動腳本沒輸出／把自己殺掉 | `pkill -f "…"` 比對到 launcher **自己的**命令列 | 啟動時不用 pkill；改用 **port/PID** 殺，不用 pattern |
| 3 | 重啟後的 serve 仍跑舊程式碼 | 非 root 下 `fuser -k` 無效 → 舊 serve 佔著 port、新的綁不到 | 用 **PID** 殺 listener（`lsof -ti:PORT`） |
| 4 | parser 改了沒生效 | *正在跑*的行程持有舊 bytecode | editable 安裝 + 乾淨重啟（清 `__pycache__`） |
| 5 | 裝 isaacsim 時 overlay `/`（11 G）快滿 | pip cache `~/.cache/pip`、`/tmp`、conda env 都在 overlay | 把 `PIP_CACHE_DIR`/`TMPDIR`/`XDG_CACHE_HOME` 導到 `/data` + 用 **prefix** conda env |
| 6 | `egl_probe` wheel build 失敗 | cmake **4.x** 不接受它的舊 CMakeLists + 無 EGL 標頭（無 sudo） | 選用相依 — 可跳過；或 4090：`sudo apt install libegl1-mesa-dev` + `conda install "cmake<4"` |
| 7 | 基底 VLM 漏掉結尾 `</tool_call>` → 未被解析 | 模型輸出不一致 | (a) tool-calling LoRA 修好；(b) parser 強化成可括號配對未閉合標籤 |
| 8 | `pegasus put` 上傳 267 MB 失敗（SSL EOF） | 單次 base64 PUT 對 Jupyter contents API 太大 | 分塊上傳（64 MB）再 `cat` 合併 |
| 9 | 某次資料夾下載中途中斷 | `pegasus.remote_size` 對暫時性 timeout 沒重試 | 在大小探測加 retry+backoff |

---

## 9. 跨硬體檢查清單（一致性）

- **路徑相對於各 repo** → H100 與 4090 完全相同。只有 `VLM_MODEL_DIR`、OpenArm 資料集路徑、`vla.checkpoints`（GR00T 名冊）是絕對路徑 — 每台機器設一次即可。
- **VLM 環境：** torch **cu128**（同時相容 H100 PCIe 與 4090，CUDA 12.8）。**不要**讓 cu130 混進來（問題 #1）。
- **受限模型：** `nvidia/Cosmos-Reason2-2B` 必須在 HF cache（`HF_HUB_OFFLINE=1`）或提供 `HF_TOKEN`。
- **agentbot ↔ VLM：** agentbot 只需要能連到 `vlm.base_url` — 同機部署或開放該 port 即可。
- **EULA：** 任何 headless 的 isaacsim/isaaclab 執行都要 `OMNI_KIT_ACCEPT_EULA=YES`。

---

## 10. 狀態與待辦

- ✅ VLM serve + tool-calling（sort_can/pick/home）、before/after 評估、強化版 parser、env_isaaclab（isaacsim+torch cu128）、agentbot 整合程式碼 + 測試、docs/dataflow。
- ✅ toolcall 產物已鏡像到本機（與 H100 逐檔一致）。
- ▶ **在 4090 上：** 建好 env_isaaclab（含 EGL + cmake<4）、接上 `sim_session` 的相機影格發布（§2-C）、跑 4 程序迴路（§6）、確認一個 episode 到達 `done`。
- ✎ 維護者需**自行 commit** 兩個 repo 的 staged 變更。
