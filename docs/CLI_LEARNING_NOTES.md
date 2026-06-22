# 指令列學習筆記（本 session 實際用到的 CLI）

> 用途：把這次工作中我實際下過的指令整理成可學習的參考。每條 = **用途 → 本 session 的例子 → 重點參數**。
> 平台：H100 是 **Linux**（bash）；本機是 **Windows**（PowerShell + Git Bash）。指令會標明屬於哪邊。
> 建議讀法：先看 §12「踩雷與解法」（最高學習價值），再回頭查各類別。

---

## 1. pegasus.py — 本專案自訂的 H100 遠端工具
透過 Jupyter 的 REST/WebSocket 在遠端 H100 上跑指令、傳檔。位於 `scripts/common/pegasus.py`。本機用 `python` 呼叫（不是 `uv run`）。

| 指令 | 用途 | 例子 |
|---|---|---|
| `run "<cmd>"` | 在 H100 上跑一行 shell（即時輸出） | `python pegasus.py run "nvidia-smi -L"` |
| `run --file <local.sh>` | 把**本機**腳本內容送到 H100 執行（避開引號地獄，最常用） | `python pegasus.py run --file tmp/build_isaaclab.sh` |
| `put <local> <remote>` | 上傳檔案/資料夾（遞迴） | `python pegasus.py put tmp/x.sh VLA/tingying/tmp/x.sh` |
| `get <remote> <local>` | 下載（**可續傳**、驗證大小） | `python pegasus.py get VLA/tingying/.../model VLA\local\model` |
| `ls / rm / mkdir <path>` | 列出/刪除/建資料夾 | `python pegasus.py ls VLA/tingying` |

- 當成 Python 模組更靈活：`import pegasus as pg; s=pg.connect(); pg.get(s, rem, loc)`（我寫 `dl_all.py` 就是這樣 + 自己加 retry）。
- **重點觀念**：`run --file` 比 `run "..."` 穩——因為 PowerShell 對內嵌雙引號的處理會把參數拆散（見 §12）。

---

## 2. git（含多 repo / submodule / gitlink）
| 指令 | 用途 | 例子 |
|---|---|---|
| `git -C <dir> <cmd>` | 在「別的資料夾」執行 git，不用 cd | `git -C Isaac-GR00T-VLM status -s` |
| `git status -s` | 精簡狀態（`M`=改、`??`=未追蹤） | `git status -s` |
| `git ls-tree <ref> <path>` | 看某 commit 裡某路徑是什麼（`160000 commit`=gitlink/submodule；`040000 tree`=一般資料夾） | `git ls-tree HEAD IsaacLab` |
| `git ls-remote <url>` | 不 clone 直接看遠端有哪些分支 | `git ls-remote https://github.com/...git` |
| `git fetch origin <branch> --no-recurse-submodules` | 只抓單一分支、不碰 submodule | `git fetch origin feature/LoRA_VLM --no-recurse-submodules` |
| `git checkout -f -B <b> origin/<b>` | 強制建/重設本地分支對齊遠端（gitlink 只動指標、不動內容） | `git checkout -f -B feature/LoRA_VLM origin/feature/LoRA_VLM` |
| `git commit -m "msg" -- <path>` | **只**提交指定檔案（即使其他檔已 staged） | `git commit -m "fix" -- scripts/common/wifi_switch.py` |
| `git remote -v / get-url / set-url` | 看/改 remote URL | `git -C IsaacLab remote set-url origin https://github.com/Deming08/IsaacLab.git` |
| `git rev-parse HEAD` / `--short HEAD` | 取目前 commit SHA | `git -C IsaacLab rev-parse HEAD` |
| `git fsck --full` | 檢查 git 物件完整性 | `git -C IsaacLab fsck --full` |
| `git clone -b <branch> <url> <dir>` | clone 指定分支 | `git clone -b develop https://github.com/Deming08/IsaacLab.git` |

- **gitlink（無 .gitmodules 的 submodule）**：本專案的 `IsaacLab`/`Isaac-GR00T_n1d7` 是「裸 gitlink」（外層只記一個 commit 指標、沒有 `.gitmodules`）→ `git submodule update` 不會自動填，要自己 `git clone` 進去再 `checkout` 那個 commit。
- 結尾 commit 訊息慣例：本專案要求 `Co-Authored-By:` trailer。

---

## 3. 環境管理：conda / uv / pip
### conda（H100 的 IsaacLab 環境）
| 指令 | 用途 | 例子 |
|---|---|---|
| `conda create -p <prefix> python=3.11 -y` | **prefix 環境**（裝在指定路徑，避開空間不足的磁碟） | `conda create -p /data/VLA/tingying/envs/env_isaaclab python=3.11 -y` |
| `conda create -n <name> ...` | 具名環境（裝在預設 `envs_dirs`） | `conda create -n env_isaaclab python=3.11` |
| `conda activate <prefix\|name>` | 啟用環境（prefix 用路徑） | `conda activate /data/VLA/tingying/envs/env_isaaclab` |
| `conda install -p <env> -c conda-forge <pkgs> -y` | 在指定環境裝套件（無 sudo 取得編譯器/EGL） | `conda install -p $ENV -c conda-forge cmake ninja compilers -y` |
| `conda config --show <key>` / `conda info` | 看設定（`envs_dirs`、`pkgs_dirs`） | `conda config --show pkgs_dirs` |
| `conda env list` | 列所有環境 | `conda env list` |

- **為什麼用 `-p` prefix**：H100 的 `/`（overlay）只剩 ~11G，但 `conda envs_dirs` 預設在那；`-p /data/...` 把環境放到 2T 的 `/data`。

### uv（Isaac-GR00T-VLM / agentbot；比 pip 快）
| 指令 | 用途 | 例子 |
|---|---|---|
| `uv sync` | 依 `pyproject.toml`+`uv.lock` 建/同步 `.venv` | `cd agentbot && uv sync` |
| `uv venv <dir>` | 只建一個空 venv | `uv venv .venv-cputest` |
| `uv pip install --python <venv> <pkgs>` | 往指定 venv 裝套件（不全量 re-resolve） | `uv pip install --python .venv-cputest pytest fastapi httpx` |
| `uv run <cmd>` | 在專案 venv 裡跑（會自動 sync） | `uv run python -m pytest -q` |
| `python -m uv ...` | uv 沒在 PATH 時這樣呼叫 | `python -m uv venv .venv-cputest` |
| `UV_LINK_MODE=copy` | 跨檔案系統 hardlink 失敗時改用複製 | `UV_LINK_MODE=copy uv sync` |

### pip
| 指令 | 用途 | 例子 |
|---|---|---|
| `pip install -U <pkg>==<ver> --index-url <url>` | 從指定 index 裝**固定版**（CUDA wheel） | `pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128` |
| `--extra-index-url <url>` | 額外 index（不取代 PyPI） | `pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com` |
| `pip install -e .` | 以「可編輯」方式裝本地專案（改 src 立即生效） | `pip install -e .` |
| `pip install <pkg> --no-build-isolation` | 用「目前環境」的建置工具（解決隔離環境找不到 cmake） | `pip install egl_probe --no-build-isolation` |
| `pip list \| grep -i <pat>` / `pip show <pkg>` / `pip cache dir` | 查已裝套件/位置/快取 | `pip list \| grep -i torch` |

- `"pkg[extra1,extra2]==ver"` 的中括號是 **extras**（選用功能），整串要加引號（shell 會把 `[]` 當特殊字元）。

---

## 4. Python 執行方式
| 寫法 | 用途 | 例子 |
|---|---|---|
| `python -m <module>` | 以模組執行（找 PATH 上的套件，比給檔案路徑穩） | `python -m uvicorn vlm_lora.serve.openai_app:app` |
| `python -u` | **unbuffered**：即時 flush stdout（detached 任務監看必備） | `python -u dl_all.py` |
| `python -c "<code>"` | 跑一小段 code（驗證 import / 取值） | `python -c "import torch; print(torch.cuda.is_available())"` |
| `PYTHONPATH=src python ...` | 把 `src/` 加進 import 路徑（沒裝套件也能 import） | `PYTHONPATH=src python -m pytest tests/` |
| `PYTHONIOENCODING=utf-8` | 避免 Windows cp950 遇到 box-drawing 字元崩潰 | `$env:PYTHONIOENCODING='utf-8'` |
| `HF_HUB_OFFLINE=1` | 強制 HuggingFace 離線（用本地 cache、不連網） | `HF_HUB_OFFLINE=1 python -m vlm_lora.merge_lora ...` |

---

## 5. 測試 / 服務 / HTTP
| 指令 | 用途 | 例子 |
|---|---|---|
| `python -m pytest <path> -v` / `-q` | 跑測試（`-v` 詳細、`-q` 精簡） | `python -m pytest tests/test_serve_toolcall.py -v` |
| `uvicorn <mod>:app --host 0.0.0.0 --port 8000` | 啟動 ASGI/FastAPI 伺服器 | `uvicorn vlm_lora.serve.openai_app:app --host 0.0.0.0 --port 8000` |
| `curl -s --max-time N <url>` | 安靜地打 HTTP（含逾時） | `curl -s --max-time 8 http://localhost:8000/health` |
| `curl ... -H '<header>' -d '<json>'` | POST JSON | `curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[...]}'` |

- `--host 0.0.0.0` = 對外開放（不只 localhost）；`-s` = silent（不顯示進度條）。

---

## 6. 行程與連接埠（Linux；本 session 的重災區）
| 指令 | 用途 | 例子 |
|---|---|---|
| `setsid bash -c 'nohup <cmd> > log 2>&1 < /dev/null &'` | **detach**：讓行程在 exec 結束後仍存活（長任務） | 啟動背景訓練/serve |
| `pgrep -af <pat>` | 找符合的 PID + 完整命令列 | `pgrep -af build_isaaclab.sh` |
| `pkill -f <pat>` | 殺符合命令列的行程 | `pkill -f train_vlm_lora` |
| `kill -9 <pid>` | 強殺指定 PID（最精準、不誤殺） | `kill -9 2984995` |
| `lsof -ti:<port>` | 找**佔用某 port** 的 PID | `lsof -ti:8000` |
| `ss -ltn \| grep :<port>` | 看哪些 port 在 listen | `ss -ltn \| grep :8000` |
| `fuser -k <port>/tcp` | 殺佔用該 port 的行程（**需 root**，見 §12） | `fuser -k 8000/tcp` |

- **`nohup ... &` vs `setsid`**：`&` 只是背景；要在「連線/exec 結束」後仍活著，要 `setsid`（開新 session）+`nohup`（忽略 SIGHUP）+`< /dev/null`（不卡在 stdin）。

---

## 7. 檔案 / 磁碟 / 文字處理（Linux + Git Bash 通用）
| 指令 | 用途 | 例子 |
|---|---|---|
| `du -sh <path>` | 看資料夾大小（`-s` 總和、`-h` 人類可讀） | `du -sh artifacts/checkpoints/gr00t/*` |
| `df -h <path>` | 看磁碟剩餘空間 | `df -h / /data` |
| `ls -la` | 詳細列檔（含隱藏、大小、權限） | `ls -la .venv/bin/` |
| `tar -czf out.tgz <dir>` / `tar -xzf out.tgz` | 打包/解包（`c`建`x`解`z`gzip`f`檔名） | `tar -czf isaaclab.tar.gz IsaacLab` |
| `split -b 64m <file> <prefix>` | 切大檔成多塊（本 session 用來分塊上傳） | `split -b 64m big.tgz part_` |
| `cp -a <src> <dst>` / `rm -rf` / `mkdir -p` | 複製(保留屬性)/遞迴刪/遞迴建 | `cp -a IsaacLab-GR00T IsaacLab-GR00T-Backup` |
| `grep -E '<regex>'` | 抓符合行（`-i`忽略大小寫 `-a`當文字 `-o`只印符合 `-n`行號） | `grep -aE 'STAGE\|error' build.log` |
| `tr '\r' '\n'` / `tr -cd '[:print:]\n'` | 轉/濾字元（把 `\r` 進度條轉行、去非可印字元） | `tr '\r' '\n' < log \| tail -20` |
| `wc -l` / `head -N` / `tail -N` | 計行數/前N/後N行 | `wc -l < data.jsonl` |
| `find <dir> -type f -printf '%s\n'` | 列每個檔的大小（驗證下載完整性） | `find $d -type f -printf '%s\n' \| awk '{s+=$1} END{print s}'` |
| `awk '{s+=$1} END{print s}'` | 逐行加總（算總位元組） | 同上 |

- 本 session 常見組合：`tr '\r' '\n' < log | grep -E '...' | tail -20`（把帶 `\r` 的進度條 log 變成可讀的關鍵行）。

---

## 8. GPU 監看：nvidia-smi
| 指令 | 用途 | 例子 |
|---|---|---|
| `nvidia-smi -L` | 列出 GPU | `nvidia-smi -L` |
| `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader` | 只取要的欄位、好解析 | 監看訓練/serve 的記憶體與使用率 |
| `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` | 看「哪些 PID 在用 GPU」 | 找佔記憶體的行程 |

- **觀念**：模型「載入但閒置」= 佔 VRAM、`utilization 0%`；只有生成那幾秒才會飆。共用機**勿亂殺** GPU PID（可能是別人的）。

---

## 9. PowerShell（Windows 本機）
| 指令 | 用途 | 例子 |
|---|---|---|
| `$env:VAR='value'` | 設環境變數 | `$env:PYTHONIOENCODING='utf-8'` |
| `Get-CimInstance Win32_Process \| Where-Object {...}` | 查行程（含命令列） | `Get-CimInstance Win32_Process \| ? { $_.CommandLine -match 'dl_toolcall' }` |
| `Stop-Process -Id <pid> -Force` | 殺行程 | `Stop-Process -Id 22672 -Force` |
| `Start-Sleep -Seconds N` | 等待 N 秒 | `Start-Sleep -Seconds 75` |
| `... \| Select-Object -Last N / -First N` | 取後/前 N 個 | `uv sync 2>&1 \| Select-Object -Last 4` |
| `try { ... } catch { ... }` | 錯誤處理 | `try { Stop-Process -Id 1 -EA Stop } catch { 'gone' }` |

- **引號陷阱（本 session 多次踩到）**：用 `python pegasus.py run '<cmd>'` 時，`<cmd>` 內**不要**有雙引號——PowerShell 會把含雙引號的字串拆成多個參數，導致 pegasus 收到斷掉的指令。解法：① 改 `run --file <script>`；② 指令裡避免雙引號（grep pattern 用單字、少用 `tr "..."`）。

---

## 10. 本專案/領域特定
| 指令 | 用途 | 例子 |
|---|---|---|
| `./isaaclab.sh --install` | 把 IsaacLab 擴充以可編輯方式裝進當前 conda env | `cd IsaacLab && ./isaaclab.sh --install` |
| `OMNI_KIT_ACCEPT_EULA=YES` | headless 跑 isaacsim/isaaclab 必設（否則卡在 EULA 詢問） | `export OMNI_KIT_ACCEPT_EULA=YES` |
| `python -m gr00t.eval.run_gr00t_server --model-path <ckpt> --port 5555` | 啟動 GR00T 動作模型 server（ZeroMQ :5555） | （sim 迴路用） |
| `python -m vlm_lora.train_vlm_lora --dataset-path ... --output-dir ... --max-steps N` | LoRA 微調 | `... --max-steps 1500 --save-steps 500` |
| `python -m vlm_lora.merge_lora --adapter-dir <ckpt> --out-dir <merged>` | 把 LoRA adapter 合併回基底模型 | 產生可部署的 standalone VLM |

---

## 11. 一句話常用組合（copy-paste 友善）
```bash
# 看 H100 某 log 的關鍵行（去掉 \r 進度條雜訊）
python scripts/common/pegasus.py run "tr '\r' '\n' < /data/.../x.log | grep -aE 'STAGE|rc=|error' | tail -20"

# 在 H100 detached 跑長任務 + 寫 log
setsid bash -c 'nohup bash tmp/job.sh > /data/.../tmp/job.log 2>&1 < /dev/null &'

# 比對本機 vs 遠端資料夾「檔數+總位元組」是否一致
find <dir> -type f | wc -l ; find <dir> -type f -printf '%s\n' | awk '{s+=$1} END{print s}'

# 殺掉佔用某 port 的 serve（非 root 也行，用 PID）
kill -9 $(lsof -ti:8000)
```

---

## 12. 踩雷與解法（這次最值得學的部分）
1. **PowerShell 引號**：內嵌雙引號的 pegasus `run "..."` 會被拆參數 → 用 `run --file`。
2. **`pkill -f <pat>` 殺到自己**：腳本內容含該 pattern 時，跑腳本的行程命令列也含它 → 自殺。改用 **port/PID**（`lsof -ti:PORT` + `kill -9`）。
3. **`fuser -k` 非 root 無效**：舊 serve 沒被殺、新的綁不到 port、舊的繼續服務（還在跑舊程式碼）。→ 用 PID 殺。
4. **舊 `.pyc` / 舊行程**：editable 安裝改了 src，但「正在跑的行程」仍是舊 bytecode → 要**重啟行程**（必要時清 `__pycache__`）。
5. **torch CUDA 版本**：`isaaclab.sh --install` 會把 torch 換成 `cu130`（需更新驅動）→ H100 驅動是 12.8 → **裝完 isaaclab 後再 `pip install -U torch==2.7.0 --index-url .../cu128` 釘回去**。
6. **磁碟在小分割區爆掉**：H100 overlay `/` 只剩 11G → 把 `PIP_CACHE_DIR`/`TMPDIR`/`XDG_CACHE_HOME`/conda env 全導到 `/data`（prefix env）。
7. **大檔上傳失敗**：單次 base64 PUT 267M 被伺服器拒（SSL EOF）→ **切 64MB 塊上傳再 `cat` 合併**。
8. **下載逾時中斷**：探測檔案大小沒重試 → 加 retry/backoff；下載要**可續傳**（HTTP Range）。
9. **unbuffered 才看得到進度**：detached log 看似「卡住」其實是 `print` 被緩衝 → `python -u`。

---

## 13. 給你的學習建議
1. **每條都先 `--help` / `man`**：例如 `tar --help`、`man grep`、`nvidia-smi --help`。先讀短旗標再讀長旗標。
2. **explainshell.com**：貼一整行複雜指令，它會逐段解釋每個旗標——對 `find ... -printf` / `setsid nohup ... &` 這種組合超有用。
3. **在沙盒練危險指令**：`rm -rf`、`pkill`、`kill -9`、`git reset --hard` 先在玩具資料夾/分支練，養成「先看再做」（`ls`/`pgrep`/`git status` 確認目標再動手）。
4. **管線思維（pipe）**：`A | B | C` 是 Unix 精髓。練 `cat/grep/sort/uniq/wc/awk/sed/head/tail/tr` 的組合，能解決 90% 的 log 分析。
5. **環境隔離是基本功**：分清 conda env vs venv vs 系統 python；`which python`、`python -c "import sys;print(sys.executable)"` 確認「現在用的是哪個 python」。
6. **把踩雷做成 checklist**：§12 那 9 條，下次遇到「serve 改了沒生效」「磁碟爆」「下載卡住」直接對照。
7. **CUDA/torch 對應**：記住「torch 的 `+cuXXX` 要 ≤ 驅動支援的 CUDA 版本」；`nvidia-smi` 右上角顯示驅動支援的最高 CUDA。
8. **動手重現**：挑一條本 session 的流程（例如 §11 的「detached 跑 + 看 log」），自己在本機用一個 `sleep 30; echo done` 的假任務跑一遍。

> 想要的話，我可以針對其中某一類（例如 **git submodule/gitlink**、**conda vs uv 環境管理**、或 **pegasus 遠端流程**）再出一份「邊做邊學」的小練習題 + 解答。
