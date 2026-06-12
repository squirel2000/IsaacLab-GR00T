# output/ — 評估與分析工作區

本資料夾是「本機評估與分析」的工作區（未版控的中介產物放這裡）。
分成兩類：**eval/**（你要執行的腳本）與 **analysis/**（產生圖表）。

```
output/
├── eval/                       # ← 要「執行」的腳本都在這
│   ├── run_evals_v2.sh         # 主力：閉迴路成功率評估（IsaacSim，crash 會自動重啟 client）
│   ├── run_evals.sh            # 舊版 v1（client 崩潰即中止，僅留作參考）
│   ├── aggregate_eps.py        # 把 *_combined_episodes.log 統計成成功率
│   ├── openloop_mse_sweep.sh   # 方案 A：對 checkpoint 離線跑 open-loop action MSE
│   └── logs/                   # 所有 *.log 輸出（已 gitignore，不進版控）
└── analysis/
    └── n16_vs_n17/             # N1.6 vs N1.7 比較圖
        ├── make_loss_svg.py        # 產生器：train loss + grad_norm（2 panel）
        ├── make_compare_svg.py     # 產生器：train loss + held-out MSE + grad_norm（3 panel）
        ├── n16_vs_n17_loss.svg
        └── n16_vs_n17_compare.svg
```

---

## 1. 閉迴路成功率評估（IsaacSim）

實際在模擬器中跑 policy、計算任務成功率。需要 Display :0。

```bash
TARGET=50 bash output/eval/run_evals_v2.sh      # 每個版本收滿 50 episodes
# 輸出：output/eval/logs/{n16,n17}_combined_episodes.log
```

統計成功率：

```bash
python3 output/eval/aggregate_eps.py output/eval/logs/n16_combined_episodes.log
python3 output/eval/aggregate_eps.py output/eval/logs/n17_combined_episodes.log
```

> 版本/checkpoint/embodiment 路徑寫死在 run_evals_v2.sh 的 `run_one ...` 那兩行，要換 checkpoint 改那裡。

---

## 2. 方案 A：離線 open-loop MSE sweep（偵測 overfitting / 選最佳 checkpoint）

對某次訓練的所有 `checkpoint-*`，在**held-out 資料集**上跑完整去噪 rollout，算 action-space MSE。
MSE 最低的 checkpoint 就是最佳點；MSE 開始回升而 train loss 還在降 = overfitting onset。

```bash
# N1.7 範例：把 CSV 直接寫進並排圖要吃的位置
OUT=output/analysis/n16_vs_n17/n17_openloop_mse.csv \
CKPT_ROOT=/path/to/N1_7_300k_run \
DATASET=/path/to/HELDOUT_dataset \
TRAJ_IDS="0 1 2 3 4" PATIENCE=3 \
  bash output/eval/openloop_mse_sweep.sh
```

CSV 欄位：`step,mse,mae`。`DATASET` 一定要是**訓練時沒用到**的 episodes，否則量到的是記憶分數。

---

## 3. 產生比較圖

```bash
python3 output/analysis/n16_vs_n17/make_compare_svg.py
# 讀 trainer_state.json + （若存在）n16/n17_openloop_mse.csv，輸出 n16_vs_n17_compare.svg
```

跑完方案 A 產生 CSV 後再重跑這支，第 2 個 panel（held-out MSE）就會被填上。

---

## 4. 在訓練中即時評估（方案 B）

不需要這裡的腳本——已內建到訓練流程：見 Isaac-GR00T_n1d7 的 `open-loop_eval` 分支，
用 `--enable-open-loop-eval` 開啟（詳見該分支 finetune 腳本）。方案 B 邊訓練邊把 MSE 記到
wandb 並可自動 early-stop；方案 A 則是訓練完之後離線補跑。兩者算的是同一個 held-out action MSE。
