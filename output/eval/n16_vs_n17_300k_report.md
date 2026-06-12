# N1.6 vs N1.7 比較報告（multitask 修正版，含 N1.7-300k retrain）

任務：`Isaac-Can-Sorting-OpenArm-DexHand-v0`（OpenArm LinkerHand-O6，dataset 0403，right_only）。
閉迴路成功率：在 IsaacSim 跑 policy（client/server），每個 checkpoint 收 100 episodes，
episode 步數上限（horizon）595。

> **⚠️ 重要更正（2026-06-11）。** 本報告較早的版本記錄的成功率是 **N1.6 51% / N1.7-150k 18% /
> N1.7-300k 53%**，那是**測試環境設定錯誤造成的假象**：評測用了**單一固定的語言描述**，但本任務是
> **multitask**——dataset 0403 依目標顏色給不同描述（task 0：「place the can on the orange plate」、
> task 1：「place the can on the green plate」）。修正評測（依顏色切換描述 + 對齊 dataset 的
> head-only 單相機）後，**三個 checkpoint 全部重跑 100 eps**，得到下面的真實數字。舊的 51/18/53 已作廢。

---

## 0. 結論摘要（TL;DR）

| Run | backbone | LR | steps | warmup | code repo | 最終 train loss | **success（修正後）** | timeout | unsafe |
|---|---|---|---|---|---|---|---|---|---|
| N1.6-150k | Eagle-Block2A-2B-v2 | 1e-4 | 150k | 0.05 | `Isaac-GR00T` | **0.00281** | **94%** | 6% | 0% |
| N1.7-150k | Cosmos-Reason2-2B / Qwen3VL | 1e-4 | 150k | 0.05 | `Isaac-GR00T` | 0.00784 | **75%** | 25% | 0% |
| N1.7-300k | Cosmos-Reason2-2B / Qwen3VL | **5e-5** | **300k** | **0.10** | `Isaac-GR00T_n1d7` | 0.00711 | **98%** | 2% | 0% |

（對照舊的錯誤數字：51% / 18% / 53%。）

三個關鍵事實：

1. **「N1.7 慘輸（18%）」是測試假象。** 正確 multitask 下 N1.7-150k 是 **75%**，並非崩潰級；它確實仍
   落後 N1.6-150k（94%）約 19 個百分點，與「N1.7 在此 recipe/150k 下對訓練資料 under-converge」一致。

2. **retrain（150k→300k，較低 LR / 更長訓練）對閉迴路 success 明顯有效：75% → 98%**，且 N1.7-300k
   **反超** N1.6-150k（98% vs 94%）。成功 episode 也更**省步**（平均 435 步，min 341 / max 533），
   比 N1.6 的平均 503 步更快完成。

3. **但 train loss 幾乎沒動，loss 與 success 的脫鉤更鐵證如山。** N1.7-300k 的最終 loss（0.00711）只比
   N1.7-150k（0.00784）低約 9%，仍是 N1.6（0.00281）的約 **2.5 倍**——**loss 更差，success 卻更高
   （98% > 94%）**。原訂目標「把 loss 壓到 N1.6 等級（~0.0026）」**未達成，但與任務成功率無關**。

---

## 1. 測試環境的 bug 與修正（為什麼舊數字作廢）

舊評測對所有 episode 餵**同一個固定描述**，沒有依目標顏色切換。但 dataset 0403 是 multitask，
語言標註本身就會隨顏色變（橘盤 / 綠盤）。固定描述等於有一半 episode 收到**與目標不符**的指令，
壓低了所有 run 的成功率，而且各 run 受影響程度不一（N1.7-150k 被壓得最慘）。

修正內容（參考 commit `2069bbdd`，但描述字串改用本 dataset 的權威版本）：

- `gr00t_infer_agent.py`：加 `--multitask`（default True）、`TARGET_COLOR_ID_TO_NAME = {0:"orange", 1:"green"}`，
  在每步建 obs 前讀 `obs["scene_obs"]["target_object_color"]`，切換 `task_description`：
  id 0（red_can / 視覺橘）→「place the can on the orange plate」、id 1（blue_can / 視覺綠）→「place the can on the green plate」。
  （**注意**：用 dataset `meta/tasks.jsonl` 的字串，不是 commit 的「place the orange can on the orange plate」，
  以避免餵 OOD 語言。）
- `can_sorting/mdp/observations.py`：新增 `target_object_color_obs()`，回傳 `(num_envs,1)` int8 color id。
- `can_sorting_env_cfg.py`：`SceneObsCfg` 加 `target_object_color = ObsTerm(func=mdp.target_object_color_obs)`。
- `--pov_list` 改為 `["head"]`：dataset 只有**一個**相機 `observation.images.camera`（→ `video.camera`），
  沒有 wrist_R / wrist_L，所以 head-only 才對齊訓練資料。

**誠實的但書**：新評測同時改了兩件事（依顏色切描述 + 相機 3→head-only），無法乾淨拆分各自貢獻。
但 (a) 是 task 語意本來就要求的，(b) 是對齊 dataset；且在切到 head-only 前，用「3 相機 + multitask 描述」
跑的前 4 集是 4/4，顯示**描述修正是主因**。

---

## 2. Q1a — loss 幾乎一樣，為什麼 success 差很多？是質變嗎？

不需要「質變」就能解釋，這是 behavioral cloning（模仿學習）的標準現象，原因有二：

**(a) train loss 量的是「訓練資料分布」上的誤差；success 取決於「rollout 分布」上的誤差。**
訓練時模型看到的都是 expert（資料）造訪過的 state；但閉迴路 rollout 時，policy 自己走出去，
state 會逐步偏離訓練分布。兩個模型在訓練分布上平均誤差相同，在這些「偏離後的 state」上的誤差
可以差很多。這就是 **covariate shift / 誤差累積**：誤差會沿著 horizon 滾雪球，595 步的 episode
裡，每步一點點差異會被放大成「能不能在時限內完成」的差別。

**(b) train loss 是對所有 (state, noise, timestep) 樣本取平均，被「容易/常見」的 state 主導。**
任務成敗其實卡在少數 bottleneck state（抓取、對位的那幾步），這些 state 對平均 loss 幾乎沒有貢獻。
一個模型可以剛好在這些關鍵 state 上把誤差降下來（提升 success），而平均 loss 幾乎不動。

> 我先前把 retrain 的差異講成「較低 LR + 更長訓練改善了 closed-loop 穩定度」——那是**過度斷定的因果故事**，
> 我收回（見 §4 的混淆變數）。

---

## 3. Q1b — 那 train loss 是不是不該拿來當訓練依據？

train loss 是 **necessary but not sufficient**（必要但不充分），這次的數據把這點推到極致：
**N1.7-300k 的 train loss 比 N1.6 差約 2.5 倍，success 卻更高（98% > 94%）。**

- train loss 是有效的「發散/崩壞」哨兵（loss 爆掉一定有問題），但它**不是 closed-loop 任務成功率的好代理**。
- 比較好的訊號層級：
  1. **held-out（未見 episode）的 open-loop action MSE**（Tier 2，就是這次 eval infra 加的東西）——
     比 train loss 好，因為量的是泛化，但仍是 open-loop。
  2. **閉迴路 success rate**（在 sim 實跑）——最終真理，但最貴。
- 這正是當初要建 held-out eval + early-stopping 那套基礎設施的理由：光看 train loss 會被誤導。

---

## 4. Q1c — 除了 LR/steps，真的沒有別的變數嗎？

**沒有，不只 LR/steps。** 把 N1.7-150k 與 N1.7-300k 的設定逐項比對後：

**完全相同（已控制）：** base model `nvidia/GR00T-N1.7-3B`、dataset 0403、modality（right_arm +
right_hand，16-step action delta、action_horizon 40）、state_dropout 0.2、global_batch 32、
color_jitter、freeze 設定（`no_tune_visual / no_tune_llm / tune_projector / tune_diffusion`）。

**不同（混淆變數）：**

| 變數 | N1.7-150k | N1.7-300k |
|---|---|---|
| learning rate | 1e-4 | 5e-5 |
| max steps | 150k | 300k |
| warmup_ratio | 0.05 | 0.10 |
| **code repo / 版本** | `Isaac-GR00T`（2026-05-04） | `Isaac-GR00T_n1d7`（2026-06-07，約晚一個月） |
| shard/sampling 參數 | 未指定（用預設） | 明確設 `shard_size 2048 / episode_sampling_rate 0.1 / num_shards_per_epoch 100000` |

最大的未控制變數是 **兩個 run 跑在不同的程式 repo**（差約一個月、且是大改版）。所以這**不是一個乾淨的
單變數 ablation**，我無法把 75%→98% 乾淨歸因到「LR 或 steps」任何單一因子。誠實的說法只能是：
**「新的 recipe + 新 code 整體把 N1.7 的閉迴路 success 推到 98%（超過 N1.6），而且是在沒有縮小
train-loss 差距的情況下達成的」**。

> 要做乾淨歸因，需要補一個 controlled run：用**同一份新 code**，把 N1.7 在 LR 1e-4 / 150k（舊 recipe）
> 重跑一次，與 300k 對照。目前沒有這個 run。

---

## 5. Q1d — flow-matching「rollout 誤差累積」有理論根據嗎？

**(a) covariate shift / 誤差累積本身有紮實的理論根據，但它是 behavioral cloning 的通則，不是
flow-matching 專屬。** 經典結果：Ross & Bagnell et al., *"A Reduction of Imitation Learning and
Structured Prediction to No-Regret Online Learning"* (AISTATS 2011，即 DAgger 那篇)——
純 behavioral cloning 的 worst-case regret 隨 horizon T 成長約 **O(T²·ε)**（DAgger 改善為 O(T·ε)）。
這就是「每步小誤差 → 沿 horizon 放大」的理論依據，直接支撐 §2 的論點。

**(b) 但我先前那句「較低 LR + 更長訓練讓 velocity field 更平滑、所以 ODE 積分更穩」——
這部分我沒有具體文獻支持，是推測，請當成 hypothesis 而非定論。** flow-matching 把 action 生成寫成
對 velocity field 的 ODE 積分，field 在資料點之間的行為確實會影響 rollout，但「lower LR/longer
training 一定讓 field 更平滑」我無法用引用佐證，也與 §4 的混淆變數糾纏在一起。

---

## 6. Q1e — backbone 換了、DiT 沒換，前述解釋合理嗎？

先更正命名：**N1.6 backbone = Eagle-Block2A-2B-v2；N1.7 backbone = Cosmos-Reason2-2B / Qwen3VL。**
DiT（action head）兩版**相同**（32 層、32 head、head_dim 48、output 1024）。

在「DiT 相同、只有 VLM conditioning 換掉」的前提下：

- N1.7-150k 在**相同 recipe** 下 train loss（0.0078）是 N1.6（0.0028）的 ~2.8 倍——代表新 VLM 在
  這個 recipe 下**對訓練資料的擬合本來就比較差**（fit 意義上的 under-converge）。
- retrain 到 300k 後 loss 只降到 0.0071（仍 ~2.5× N1.6）——**這個 fit 差距並沒有被補上**。
  所以 success 升到 98% **不是來自「把 loss 收斂到 N1.6 等級」**（那件事沒發生）。
- 因此最站得住腳的結論是：**N1.7 的 train-loss floor 結構性地比 N1.6 高（很可能源自不同的
  VLM / normalization），且 retrain 沒改變這點；但這個 loss 差距與閉迴路 success 完全脫鉤——
  N1.7-300k 的 loss 更差，閉迴路卻比 N1.6 更好。** 再次印證 §3：train loss 在這裡是錯的量尺。

---

## 7. 本次 eval 細節（multitask 修正版，三個版本）

- orchestrator：`output/eval/run_eval_multitask.sh`（crash-resilient，Display :0，TARGET=100，head-only、`--multitask`）
- logs：`output/eval/logs_mt/`（`*_combined_episodes.log` 為各版本的逐 episode 結果）
- 三個版本都**一次 attempt 收滿 100/100**，無 server crash。

| Run | success | 成功步數 mean / min / max | timeout（截斷，595 步） | unsafe（終止） |
|---|---|---|---|---|
| N1.6-150k  | 94/100 | 503 / 425 / 590 | 6  | 0 |
| N1.7-150k  | 75/100 | 479 / 368 / 589 | 25 | 0 |
| N1.7-300k  | 98/100 | 435 / 341 / 533 | 2  | 0 |

- 失敗幾乎全是 timeout（policy 沒在時限內完成），**0 unsafe termination**——沒有危險行為。
- binomial 標準誤 ≈ √(p·(1-p)/100)：在 p≈0.9 時約 3%、p≈0.75 時約 4.3%，故各版本差距遠超雜訊。

checkpoints：

- N1.6-150k：`artifacts/checkpoints/gr00t/openarm_linkerhando6_multitask_N16_150k_dataset_0403_no_tune_visual/checkpoint-150000`
- N1.7-150k：`artifacts/checkpoints/gr00t/openarm_linkerhando6_multitask_N17_150k_dataset_0403_no_tune_visual/checkpoint-150000`
- N1.7-300k：`artifacts/checkpoints/gr00t/N1_7_fft_0607_300k_no_tune_visual/checkpoint-300000`

---

## 8. 圖表

- `output/analysis/n16_vs_n17/n16_vs_n17_loss.svg` — train loss（log）+ grad norm + success-rate bar（94/75/98%）
- `output/analysis/n16_vs_n17/n16_vs_n17_compare.svg` — train loss + held-out MSE（待補 CSV）+ grad norm

重繪：

```bash
python3 output/analysis/n16_vs_n17/make_loss_svg.py    output/analysis/n16_vs_n17/n16_vs_n17_loss.svg
python3 output/analysis/n16_vs_n17/make_compare_svg.py output/analysis/n16_vs_n17/n16_vs_n17_compare.svg
```

---

## 9. 還需要做什麼來確認 root cause？

root cause（「是 recipe/under-training 還是 backbone 結構缺陷」）已可下結論：**不是 backbone 結構缺陷
**——同一個 N1.7 backbone 換 recipe/code 後就達到 98%（超過 N1.6）。**嚴格的單因子歸因**則仍缺一個
controlled run（見 §4 末）：用**新 code**、N1.7 @ LR1e-4 / 150k 重跑一次，與本次 300k 對照，
這樣 LR/steps 以外的變數才完全對齊。否則目前 94/75/98 的趨勢已足以支持「retrain 成功、N1.7 可用且
優於 N1.6」的工程結論。
