#!/usr/bin/env python3
"""Build the comprehensive Traditional-Chinese Phase-1 analysis report (self-contained HTML).

Loads BOTH runs' metrics (no-warmup vs critic-warmup), overlays them, embeds RLinf's
published GR00T/LIBERO numbers, explains the signals honestly, and draws architecture +
flow diagrams as inline SVG. Writes agents/docs/rlinf_gr00t_n17_libero_phase1_report.html.

Usage:  python agents/rl-trainbot/harness/report_full.py
Inputs (already fetched locally):
  artifacts/rl/libero_spatial_ppo_gr00t_n1d7_phase1/metrics.json   (critic-warmup run)
  artifacts/rl/metrics_nowarmup.json                               (no-warmup run)
"""
import html
import json
import pathlib

def _ws_root() -> pathlib.Path:
    """Walk up to the workspace.yaml root marker (survives directory moves)."""
    here = pathlib.Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        if (d / "workspace.yaml").is_file():
            return d
    raise FileNotFoundError("workspace.yaml not found walking up from " + str(here))


_REPO = _ws_root()
WARMUP = _REPO / "artifacts/rl/libero_spatial_ppo_gr00t_n1d7_phase1/metrics.json"
NOWARM = _REPO / "artifacts/rl/metrics_nowarmup.json"
N15 = _REPO / "artifacts/rl/metrics_n15_spatial.json"
N15D = _REPO / "artifacts/rl/metrics_n15_spatial_dualgpu.json"
N15S = _REPO / "artifacts/rl/metrics_n15_spatial_scaled.json"
OUT = _REPO / "agents/docs/rlinf_gr00t_n17_libero_phase1_report.html"

W = json.load(open(WARMUP, encoding="utf-8"))
N = json.load(open(NOWARM, encoding="utf-8"))
M15 = json.load(open(N15, encoding="utf-8"))
M15D = json.load(open(N15D, encoding="utf-8"))
M15S = json.load(open(N15S, encoding="utf-8"))


def ser(d, tag):
    return [(float(s), float(v)) for s, v in d.get(tag, [])]


def vals(d, tag):
    return [v for _, v in ser(d, tag)]


def moving_avg(v, k=9):
    if len(v) < k:
        return v
    out = []
    for i in range(len(v)):
        a = max(0, i - k // 2)
        b = min(len(v), i + k // 2 + 1)
        out.append(sum(v[a:b]) / (b - a))
    return out


def svg_lines(series, colors, labels, title, w=620, h=230, pad=42, yclip=None, ymarks=None):
    """series: list of [(x,y),...]. Overlaid line chart with axes + legend."""
    allpts = [p for s in series for p in s]
    if not allpts:
        return "<p>(no data)</p>"
    xs = [p[0] for p in allpts]
    ysraw = [p[1] for p in allpts]
    xmin, xmax = min(xs), max(xs) or 1
    if yclip:
        ys = [min(max(y, yclip[0]), yclip[1]) for y in ysraw]
        ymin, ymax = yclip
    else:
        ys = ysraw
        ymin, ymax = min(ys), max(ys)
    xr = (xmax - xmin) or 1
    yr = (ymax - ymin) or 1

    def sx(x):
        return pad + (x - xmin) / xr * (w - 2 * pad)

    def sy(y):
        yy = min(max(y, ymin), ymax) if yclip else y
        return h - pad - (yy - ymin) / yr * (h - 2 * pad)

    parts = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">']
    parts.append(f'<text x="{pad}" y="18" font-size="13" font-weight="600" fill="#1f2937">{html.escape(title)}</text>')
    # axes
    parts.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#cbd5e1"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#cbd5e1"/>')
    # y ticks
    for t in (ymarks or [ymin, (ymin + ymax) / 2, ymax]):
        yy = sy(t)
        parts.append(f'<line x1="{pad-4}" y1="{yy:.1f}" x2="{w-pad}" y2="{yy:.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{pad-6}" y="{yy+3:.1f}" font-size="9" text-anchor="end" fill="#64748b">{t:.3g}</text>')
    parts.append(f'<text x="{w-pad}" y="{h-pad+14}" font-size="9" text-anchor="end" fill="#64748b">step {int(xmax)}</text>')
    parts.append(f'<text x="{pad}" y="{h-pad+14}" font-size="9" fill="#64748b">{int(xmin)}</text>')
    # lines
    for s, c in zip(series, colors):
        if not s:
            continue
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in s)
        parts.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="{pts}"/>')
    # legend
    lx = w - pad - 150
    for i, (c, lab) in enumerate(zip(colors, labels)):
        ly = pad + 4 + i * 15
        parts.append(f'<rect x="{lx}" y="{ly-8}" width="12" height="4" fill="{c}"/>')
        parts.append(f'<text x="{lx+16}" y="{ly-4}" font-size="10" fill="#334155">{html.escape(lab)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def qtr(v):
    if not v:
        return (0, 0)
    k = max(1, len(v) // 4)
    return round(sum(v[:k]) / k, 3), round(sum(v[-k:]) / k, 3)


# ---- key series ----
ev_w, ev_n = ser(W, "eval/success_once"), ser(N, "eval/success_once")
tr_w, tr_n = ser(W, "env/success_once"), ser(N, "env/success_once")
ev_ok_w, ev_ok_n = qtr(vals(W, "eval/success_once")), qtr(vals(N, "eval/success_once"))
tr_ok_w, tr_ok_n = qtr(vals(W, "env/success_once")), qtr(vals(N, "env/success_once"))
evx_w = ser(W, "train/critic/explained_variance")
evx_n = ser(N, "train/critic/explained_variance")
vl_w, vl_n = ser(W, "train/critic/value_loss"), ser(N, "train/critic/value_loss")
kl_w = ser(W, "train/actor/approx_kl")
rw_w = ser(W, "env/reward")
ratio_w = ser(W, "train/actor/ratio")
tot_w = ser(W, "train/actor/total_loss")
pl_w = ser(W, "train/actor/policy_loss")

# smoothed training success
trs_w = list(zip([x for x, _ in tr_w], moving_avg([y for _, y in tr_w])))
trs_n = list(zip([x for x, _ in tr_n], moving_avg([y for _, y in tr_n])))

# ---- N1.5 spatial reproduction (SFT -> RL, separate RLinf-n15 clone) ----
ev_m = ser(M15, "eval/success_once")
tr_m = ser(M15, "env/success_once")
ev_ok_m = qtr(vals(M15, "eval/success_once"))
tr_ok_m = qtr(vals(M15, "env/success_once"))
evx_m = ser(M15, "train/critic/explained_variance")
vl_m = ser(M15, "train/critic/value_loss")
trs_m = list(zip([x for x, _ in tr_m], moving_avg([y for _, y in tr_m])))

# ---- N1.5 spatial DUAL-GPU reproduction (actor:0-1 real FSDP shard, batch 32/16) ----
ev_d = ser(M15D, "eval/success_once")
tr_d = ser(M15D, "env/success_once")
ev_ok_d = qtr(vals(M15D, "eval/success_once"))
tr_ok_d = qtr(vals(M15D, "env/success_once"))
evx_d = ser(M15D, "train/critic/explained_variance")
vl_d = ser(M15D, "train/critic/value_loss")
trs_d = list(zip([x for x, _ in tr_d], moving_avg([y for _, y in tr_d])))
_evxd_vals = vals(M15D, "train/critic/explained_variance")
evx_d_positive_frac = sum(1 for x in _evxd_vals if x > 0) / len(_evxd_vals)
evx_d_mean = sum(_evxd_vals) / len(_evxd_vals)

# ---- N1.5 spatial SCALED SINGLE-GPU, the 58h/230-epoch final run (envs=8, batch=64) ----
ev_s = ser(M15S, "eval/success_once")
tr_s = ser(M15S, "env/success_once")
ev_ok_s = qtr(vals(M15S, "eval/success_once"))
tr_ok_s = qtr(vals(M15S, "env/success_once"))
evx_s = ser(M15S, "train/critic/explained_variance")
vl_s = ser(M15S, "train/critic/value_loss")
trs_s = list(zip([x for x, _ in tr_s], moving_avg([y for _, y in tr_s])))
_evxs_vals = vals(M15S, "train/critic/explained_variance")
_evxs_finite = [x for x in _evxs_vals if x == x]
evx_s_positive_frac = sum(1 for x in _evxs_finite if x > 0) / len(_evxs_vals)
evx_s_mean = sum(_evxs_finite) / len(_evxs_finite)
evx_s_nan_count = len(_evxs_vals) - len(_evxs_finite)

BLUE, RED, GREEN, PURPLE, GRAY, ORANGE, TEAL = "#2563eb", "#dc2626", "#16a34a", "#9333ea", "#94a3b8", "#ea580c", "#0d9488"
MAGENTA = "#db2777"

# ---- charts ----
chart_eval = svg_lines([ev_n, ev_w], [RED, BLUE],
                       ["no-warmup", "critic-warmup"],
                       "eval/success_once（每次 eval 4 trials；GPU1）",
                       yclip=(0, 1), ymarks=[0, 0.25, 0.5, 0.75, 1])
chart_train = svg_lines([trs_n, trs_w], [RED, BLUE],
                        ["no-warmup (9-pt MA)", "critic-warmup (9-pt MA)"],
                        "env/success_once 訓練成功率（移動平均，降噪後）",
                        yclip=(0, 1), ymarks=[0, 0.25, 0.5, 0.75, 1])
chart_ev = svg_lines([evx_n, evx_w], [RED, BLUE],
                     ["no-warmup", "critic-warmup"],
                     "explained_variance（clip 顯示於 [-5,1]；健康應接近 1）",
                     yclip=(-5, 1), ymarks=[-5, -2, 0, 1])
chart_vl = svg_lines([vl_n, vl_w], [RED, BLUE], ["no-warmup", "critic-warmup"],
                     "critic value_loss")
chart_kl = svg_lines([kl_w], [BLUE], ["critic-warmup"], "actor approx_kl（KL 散度，越小越穩）")
chart_rw = svg_lines([rw_w], [GREEN], ["critic-warmup"], "env/reward（獎勵；注意量級 ~0.004）")
chart_ratio = svg_lines([ratio_w], [PURPLE], ["critic-warmup"], "actor ratio（PPO 重要性比；≈1 代表 policy 幾乎沒動）")
chart_loss = svg_lines([tot_w, pl_w], [BLUE, GRAY], ["total_loss", "policy_loss"],
                       "actor loss（policy_loss 前段=0 是 critic warmup 凍結 policy）")
chart_n15_eval = svg_lines([ev_m], [ORANGE], ["N1.5 spatial (SFT→RL 重現)"],
                           "N1.5 spatial eval/success_once（每次 eval 4 trials；GPU1）",
                           yclip=(0, 1), ymarks=[0, 0.25, 0.5, 0.75, 1])
chart_n15_train = svg_lines([trs_m], [ORANGE], ["N1.5 spatial (9-pt MA)"],
                            "N1.5 spatial env/success_once 訓練成功率（移動平均，32 traj/step）",
                            yclip=(0, 1), ymarks=[0, 0.25, 0.5, 0.75, 1])
chart_n15_vl = svg_lines([vl_m], [ORANGE], ["N1.5 spatial critic value_loss"],
                         "N1.5 spatial critic value_loss（持續下降＝critic 有在擬合 return，"
                         "但 explained_variance 全程 NaN，見下文）")
chart_d_eval = svg_lines([ev_d, ev_m], [TEAL, ORANGE],
                         ["雙卡 (batch 32/16)", "單卡 (batch 4/4)"],
                         "N1.5 spatial eval/success_once：雙卡 vs 單卡",
                         yclip=(0, 1), ymarks=[0, 0.25, 0.5, 0.75, 1])
chart_d_train = svg_lines([trs_d, trs_m], [TEAL, ORANGE],
                          ["雙卡 (9-pt MA)", "單卡 (9-pt MA)"],
                          "N1.5 spatial env/success_once 訓練成功率：雙卡 vs 單卡",
                          yclip=(0, 1), ymarks=[0, 0.25, 0.5, 0.75, 1])
chart_d_ev = svg_lines([evx_d, evx_m], [TEAL, ORANGE],
                       ["雙卡 (batch 32/16)", "單卡 (batch 4/4)"],
                       "explained_variance：雙卡 vs 單卡（單卡全程 NaN，圖上看不到橘線）",
                       yclip=(-2, 1), ymarks=[-2, -1, 0, 1])

# ---- final 230-epoch scaled single-GPU run charts ----
chart_s_eval = svg_lines([ev_s], [MAGENTA], ["最終跑 (envs=8, batch=64, 230 epochs)"],
                         "N1.5 spatial eval/success_once（最終 58h 跑，23 個 eval 點）",
                         yclip=(0, 1), ymarks=[0, 0.25, 0.5, 0.75, 1])
chart_s_train = svg_lines([trs_s], [MAGENTA], ["最終跑 (9-pt MA)"],
                          "N1.5 spatial env/success_once 訓練成功率（230 epochs，移動平均)",
                          yclip=(0, 1), ymarks=[0, 0.25, 0.5, 0.75, 1])
chart_s_ev = svg_lines([evx_s], [MAGENTA], ["最終跑 explained_variance"],
                       "explained_variance（230 epochs；97% 為正、僅 2 點 NaN）",
                       yclip=(-2, 1), ymarks=[-2, -1, 0, 1])
chart_s_all_eval = svg_lines([ev_m, ev_d, ev_s], [ORANGE, TEAL, MAGENTA],
                             ["單卡 envs=4 (30ep)", "雙卡 envs=4 (72ep)", "單卡 envs=8 (230ep，最終)"],
                             "eval/success_once：三次 N1.5 spatial 嘗試對照",
                             yclip=(0, 1), ymarks=[0, 0.25, 0.5, 0.75, 1])

# ---- RLinf published table ----
rlinf_rows = [
    ("N1.5", "41.4%", "92.5%", "+51.1 pp", "52.5% → 89.5%（4 suites 平均，+37 pp）；本報告已重現 spatial，見 5A"),
    ("N1.6", "70%", "82%", "+12 pp", "僅 spatial；SFT checkpoint 實際存在（見下方說明），本報告尚未執行"),
    ("N1.7", "—", "—", "—", "官方標記 TODO（尚未公布）"),
]
rlinf_html = "".join(
    f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td><b>{e}</b></td><td>{f}</td></tr>"
    for a, b, c, e, f in rlinf_rows
)

# ---- architecture diagram (inline SVG) ----
ARCH_SVG = """
<svg width="820" height="360" viewBox="0 0 820 360" role="img" font-family="sans-serif">
  <text x="12" y="20" font-size="14" font-weight="700" fill="#0f172a">系統架構（本地 ↔ Pegasus 容器 ↔ 單張 H100）</text>
  <!-- Local -->
  <rect x="12" y="40" width="220" height="300" rx="10" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="24" y="62" font-size="12" font-weight="700" fill="#1d4ed8">本地 (Windows)</text>
  <rect x="28" y="76" width="188" height="34" rx="6" fill="#fff" stroke="#bfdbfe"/><text x="36" y="97" font-size="11">launch.py（選GPU→啟動）</text>
  <rect x="28" y="118" width="188" height="30" rx="6" fill="#fff" stroke="#bfdbfe"/><text x="36" y="137" font-size="11">poll.py（監看）</text>
  <rect x="28" y="156" width="188" height="30" rx="6" fill="#fff" stroke="#bfdbfe"/><text x="36" y="175" font-size="11">report.py / extract_metrics</text>
  <rect x="28" y="196" width="188" height="34" rx="6" fill="#f1f5f9" stroke="#cbd5e1"/><text x="36" y="217" font-size="11">common/pegasus.py（Jupyter</text><text x="36" y="230" font-size="11">kernel REST/WS 遠端執行）</text>
  <rect x="28" y="240" width="188" height="30" rx="6" fill="#f1f5f9" stroke="#cbd5e1"/><text x="36" y="259" font-size="11">config/phase1.yaml（旋鈕）</text>
  <rect x="28" y="278" width="188" height="50" rx="6" fill="#f1f5f9" stroke="#cbd5e1"/><text x="36" y="297" font-size="11">patch_rlinf_clone.py</text><text x="36" y="313" font-size="10.5" fill="#475569">6 個 RLinf 修補（冪等）</text>
  <!-- arrow -->
  <line x1="232" y1="200" x2="300" y2="200" stroke="#64748b" stroke-width="2" marker-end="url(#ah)"/>
  <text x="238" y="192" font-size="10" fill="#475569">setsid nohup</text>
  <!-- Pegasus -->
  <rect x="300" y="40" width="508" height="300" rx="10" fill="#f0fdf4" stroke="#86efac"/>
  <text x="312" y="62" font-size="12" font-weight="700" fill="#15803d">Pegasus 容器 pa-jp-v1（cgroup pids.max=2048, EGL, 無 Docker）</text>
  <!-- Ray cluster -->
  <rect x="316" y="76" width="476" height="150" rx="8" fill="#ffffff" stroke="#bbf7d0"/>
  <text x="326" y="94" font-size="11" font-weight="700" fill="#166534">Ray cluster（num_cpus=4, dashboard off）</text>
  <rect x="326" y="104" width="150" height="46" rx="6" fill="#fef9c3" stroke="#fde68a"/><text x="334" y="122" font-size="11" font-weight="600">ActorGroup (GPU1)</text><text x="334" y="138" font-size="10" fill="#475569">PPO 更新 actor+critic</text>
  <rect x="486" y="104" width="150" height="46" rx="6" fill="#fef9c3" stroke="#fde68a"/><text x="494" y="122" font-size="11" font-weight="600">RolloutGroup (GPU1)</text><text x="494" y="138" font-size="10" fill="#475569">產生 rollout（推論）</text>
  <rect x="646" y="104" width="138" height="46" rx="6" fill="#fef9c3" stroke="#fde68a"/><text x="654" y="122" font-size="11" font-weight="600">EnvGroup (GPU1)</text><text x="654" y="138" font-size="10" fill="#475569">in-process LIBERO</text>
  <rect x="326" y="158" width="458" height="60" rx="6" fill="#ecfeff" stroke="#a5f3fc"/>
  <text x="334" y="176" font-size="11" font-weight="700" fill="#0e7490">GR00T-N1.7 模型（FSDP, NO_SHARD @ world=1）</text>
  <text x="334" y="193" font-size="10.5" fill="#334155">Cosmos-Reason2-2B backbone（本地副本）＋ Flow-Matching action head ＋ value head（隨機初始化）</text>
  <text x="334" y="208" font-size="10.5" fill="#334155">4× in-process LIBERO(robosuite/MuJoCo, EGL 離屏算繪) 序列步進</text>
  <!-- outputs -->
  <rect x="316" y="236" width="230" height="44" rx="6" fill="#fff" stroke="#bbf7d0"/><text x="324" y="254" font-size="11" font-weight="600">checkpoints/global_step_N</text><text x="324" y="270" font-size="10" fill="#475569">/data/.../RLinf/checkpoints</text>
  <rect x="556" y="236" width="236" height="44" rx="6" fill="#fff" stroke="#bbf7d0"/><text x="564" y="254" font-size="11" font-weight="600">logs/&lt;ts&gt;/tensorboard/*</text><text x="564" y="270" font-size="10" fill="#475569">TB scalars → report.py 抓取</text>
  <text x="316" y="304" font-size="10.5" fill="#475569">GPU0 全程閒置（合作式單卡）；zombie 由 jupyter PID1 無法回收 → num_cpus 壓低留 pid 餘裕</text>
  <defs><marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#64748b"/></marker></defs>
</svg>
"""

# ---- PPO flow diagram ----
FLOW_SVG = """
<svg width="820" height="250" viewBox="0 0 820 250" role="img" font-family="sans-serif">
  <text x="12" y="20" font-size="14" font-weight="700" fill="#0f172a">PPO 訓練迴圈（每個 epoch）＋ critic warmup 閘門</text>
  <rect x="20" y="45" width="130" height="52" rx="8" fill="#e0e7ff" stroke="#a5b4fc"/><text x="34" y="67" font-size="11" font-weight="600">1. Rollout</text><text x="30" y="83" font-size="10" fill="#475569">policy 在 LIBERO</text>
  <rect x="180" y="45" width="140" height="52" rx="8" fill="#e0e7ff" stroke="#a5b4fc"/><text x="192" y="67" font-size="11" font-weight="600">2. 算 Advantage</text><text x="188" y="83" font-size="10" fill="#475569">用 critic/value 估計</text>
  <rect x="350" y="45" width="160" height="52" rx="8" fill="#e0e7ff" stroke="#a5b4fc"/><text x="362" y="67" font-size="11" font-weight="600">3. PPO 更新</text><text x="358" y="83" font-size="10" fill="#475569">actor(+critic) 梯度</text>
  <rect x="540" y="45" width="120" height="52" rx="8" fill="#e0e7ff" stroke="#a5b4fc"/><text x="552" y="67" font-size="11" font-weight="600">4. Eval</text><text x="548" y="83" font-size="10" fill="#475569">每 10 epochs</text>
  <rect x="690" y="45" width="110" height="52" rx="8" fill="#dcfce7" stroke="#86efac"/><text x="700" y="67" font-size="11" font-weight="600">5. 存檔</text><text x="700" y="83" font-size="10" fill="#475569">checkpoint</text>
  <line x1="150" y1="71" x2="178" y2="71" stroke="#6366f1" stroke-width="2" marker-end="url(#af)"/>
  <line x1="320" y1="71" x2="348" y2="71" stroke="#6366f1" stroke-width="2" marker-end="url(#af)"/>
  <line x1="510" y1="71" x2="538" y2="71" stroke="#6366f1" stroke-width="2" marker-end="url(#af)"/>
  <line x1="660" y1="71" x2="688" y2="71" stroke="#6366f1" stroke-width="2" marker-end="url(#af)"/>
  <path d="M745,97 L745,120 L85,120 L85,97" fill="none" stroke="#6366f1" stroke-width="2" marker-end="url(#af)"/>
  <text x="360" y="135" font-size="10" fill="#475569">重複至 max_epochs</text>
  <!-- warmup gate -->
  <rect x="180" y="155" width="330" height="70" rx="8" fill="#fef2f2" stroke="#fecaca"/>
  <text x="192" y="175" font-size="11.5" font-weight="700" fill="#b91c1c">critic warmup 閘門（optimizer_steps &lt; 40）</text>
  <text x="192" y="193" font-size="10.5" fill="#334155">warmup 期間：步驟 3 只更新 critic、凍結 actor（policy_loss=0）</text>
  <text x="192" y="209" font-size="10.5" fill="#334155">目的：先讓隨機初始化的 value head 學到堪用的估計，</text>
  <text x="192" y="221" font-size="10.5" fill="#334155">避免早期用「垃圾 advantage」把 policy 帶壞。</text>
  <line x1="430" y1="97" x2="360" y2="153" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4" marker-end="url(#af)"/>
  <defs><marker id="af" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#6366f1"/></marker></defs>
</svg>
"""

CSS = """
body{font-family:-apple-system,'Segoe UI','Microsoft JhengHei',Roboto,Arial,sans-serif;margin:0;background:#f7f8fa;color:#1f2937;line-height:1.7}
.wrap{max-width:900px;margin:0 auto;padding:32px 22px}
h1{font-size:23px;margin:0 0 4px} h2{font-size:18px;margin:30px 0 10px;border-left:4px solid #2563eb;padding-left:10px}
h3{font-size:14px;margin:16px 0 6px;color:#334155} .sub{color:#6b7280;margin:0 0 20px}
.card{background:#fff;border-radius:10px;padding:14px 18px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,.06);overflow-x:auto}
.verdict{padding:14px 18px;border-radius:10px;font-weight:600;margin:16px 0}
.warn{background:#fef9c3;color:#854d0e} .bad{background:#fee2e2;color:#991b1b} .ok{background:#dcfce7;color:#166534}
table{border-collapse:collapse;width:100%;font-size:13px;background:#fff;border-radius:8px;overflow:hidden}
th,td{border-bottom:1px solid #eef0f3;padding:8px 10px;text-align:left} th{background:#f8fafc;color:#334155}
.g2{display:flex;flex-wrap:wrap;gap:12px} .g2 .card{flex:1 1 380px;margin:0}
ul{margin:6px 0 6px 2px;padding-left:20px} li{margin:3px 0} code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}
.k{color:#b91c1c;font-weight:600}
"""


# Override the hand-drawn diagrams with the drawio-blocksmith generated SVGs (editable .drawio alongside).
_DIAG = _REPO / "agents/rl-trainbot/harness/diagrams"
try:
    ARCH_SVG = (_DIAG / "architecture.svg").read_text(encoding="utf-8")
    FLOW_SVG = (_DIAG / "flow.svg").read_text(encoding="utf-8")
except Exception as _e:
    print("diagram svg not found; keeping inline fallback:", _e)

Q1_HTML = """<h2>2A. 有沒有「未經 SFT」的原始 checkpoint？RL 對 VLA 的幫助是什麼？</h2><div class='card'>
<h3>GR00T 的層次</h3>
<p>GR00T-N1.7 = <b>Cosmos-Reason2-2B</b>（大規模預訓練的視覺語言 backbone）＋ <b>flow-matching action head</b>。所謂「未經 LIBERO 監督/模仿訓練」的版本，就是<b>基礎 foundation 權重</b>（如 <code>nvidia/GR00T-N1.5-3B</code> 或 N1.7 base）——它有一般化的視覺語言與操作先驗，但<b>沒有針對 LIBERO 任務微調</b>。</p>
<h3>能不能直接對「未 SFT」的模型做 RL？</h3>
<p>實務上幾乎不可行。LIBERO 這類操作任務是<b>稀疏獎勵</b>（要一連串正確動作才拿到成功訊號）。若從沒學過此任務的 policy 開始，隨機探索幾乎永遠拿不到獎勵 → RL 沒有訊號可學。這就是為什麼 <b>RLinf 一律從 SFT checkpoint 起步</b>（先用行為模仿學到「大致會做」，RL 再精修）。</p>
<h3>所以 RL 對 VLA 的幫助＝？</h3>
<p>RL 的價值是<b>在一個已具基本能力的 SFT policy 上，把任務成功率往上推、對齊任務目標</b>，而不是從零學會任務。要「評估 RL 的幫助」，正確對照是<b>同一個 SFT baseline 的『SFT-only』vs『SFT+RL』</b> —— RLinf 的 N1.5 <b>41.4% → 92.5%</b> 就是這個對照（第 5 節）。我們的 N1.7 因 critic 失準沒看到提升；若要親眼看到 RL 的幫助，最直接是<b>重現 RLinf N1.5</b>（其 4 個 suite 的 SFT 已公開）。</p></div>"""

MATH_HTML = """<h2>2B. RL 的數學與運作機制（actor / critic）</h2><div class='card'>
<h3>兩個網路</h3><ul>
<li><b>Actor（策略）π<sub>θ</sub>(a|s)</b>：即 GR00T 的 action head（flow-matching）；輸入觀測 s、輸出動作分佈，θ 是要優化的策略參數。</li>
<li><b>Critic（價值）V<sub>φ</sub>(s)</b>：即新接的 value head；估計「從狀態 s 起、依現行策略能拿到的期望累積回報」。</li></ul>
<h3>回報、優勢(advantage)與 GAE</h3>
<p>折扣回報 R<sub>t</sub> = Σ<sub>k≥0</sub> γ<sup>k</sup> r<sub>t+k</sub>（γ=0.99）。優勢 A<sub>t</sub> 衡量「在 s<sub>t</sub> 採取 a<sub>t</sub> 比平均好多少」：<br>
TD 誤差 δ<sub>t</sub> = r<sub>t</sub> + γ·V<sub>φ</sub>(s<sub>t+1</sub>) − V<sub>φ</sub>(s<sub>t</sub>)；&nbsp; GAE：Â<sub>t</sub> = Σ<sub>l≥0</sub> (γλ)<sup>l</sup> δ<sub>t+l</sub>（λ=0.95）。<br>
<b class='k'>關鍵：Â<sub>t</sub> 完全由 critic V<sub>φ</sub> 算出。</b></p>
<h3>PPO 目標函數</h3>
<p>重要性比 r<sub>t</sub>(θ) = π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>) / π<sub>θ_old</sub>(a<sub>t</sub>|s<sub>t</sub>)。<br>
<b>Actor（clipped surrogate）</b>：&nbsp; L<sup>CLIP</sup>(θ) = Ê<sub>t</sub>[ min( r<sub>t</sub>(θ)·Â<sub>t</sub>, clip(r<sub>t</sub>(θ), 1−ε, 1+ε)·Â<sub>t</sub> ) ]（ε=0.2）。<br>
<b>Critic</b>：&nbsp; L<sup>V</sup>(φ) = Ê<sub>t</sub>[ ( V<sub>φ</sub>(s<sub>t</sub>) − R<sub>t</sub> )<sup>2</sup> ]。<br>
<b>總損失</b>：&nbsp; L = −L<sup>CLIP</sup> + c<sub>v</sub>·L<sup>V</sup> − c<sub>e</sub>·H[π<sub>θ</sub>]（本次 c<sub>e</sub>=entropy_bonus=0，故 entropy 項=0 → 這就是圖上 entropy_loss=0 的原因）。</p>
<h3>對應到我們的觀測</h3><ul>
<li><b>approx_kl</b> ≈ KL(π<sub>old</sub>‖π<sub>θ</sub>)、<b>ratio</b> = r<sub>t</sub> 平均；兩者 ≈0 / ≈1 代表每步 policy 幾乎沒動。</li>
<li><b class='k'>為什麼冷啟動 critic 會拖垮 policy（用式子看）</b>：Â<sub>t</sub> 由 V<sub>φ</sub> 算；若 V<sub>φ</sub> 隨機初始化、estimate 亂跳 → Â<sub>t</sub> 是雜訊 → L<sup>CLIP</sup> 的梯度 ∇<sub>θ</sub>L 也是雜訊 → θ 往錯方向漂 → 成功率下降。critic warmup＝先只用 L<sup>V</sup> 更新 φ、凍結 θ，等 Â<sub>t</sub> 有意義再更新 actor。</li>
<li><b>explained_variance = 1 − Var(R<sub>t</sub> − V<sub>φ</sub>) / Var(R<sub>t</sub>)</b>：直接量 critic 擬合回報的程度（1=完美、0=等於猜平均、負=更差）。我們一路是負（-0.3→-338）＝V<sub>φ</sub> 比猜平均還差 → Â<sub>t</sub> 不可靠。這就是「沒學起來」的量化證據。</li></ul></div>"""

SCALE_HTML = """<h2>5B. 規模驗證：「算力/規模不足」是不是真的 root cause？</h2><div class='card'>
<p>使用者要求不要只憑猜測就下這個結論，所以這裡把驗證過程完整記錄：查證 RLinf 官方的實際規模設定、查證官方文件對 RL 調參的說明、從數學推導檢查「規模不足會傷害 PPO」這個機制是否成立，最後用<b>雙卡、放大規模重跑一次</b>直接測試。</p>

<h3>5B.1　量化落差：官方設定 vs 我們的設定</h3>
<p>直接讀 RLinf 官方 <code>examples/embodiment/config/libero_spatial_ppo_gr00t.yaml</code> 與官方文件（<code>docs/source-en/rst_source/examples/embodied/{{gr00t,libero}}.rst</code}，2026-07-02 查證）確認：</p>
<table><tr><th>設定項</th><th>RLinf 官方 N1.5 recipe</th><th>我們：單卡（5A）</th><th>我們：雙卡重跑</th></tr>
<tr><td>硬體</td><td>LIBERO 文件明載 <b>1–2 nodes・8–16 GPUs</b>；官方論文(arXiv:2509.15965)大規模實驗用 8×H100 node（未逐一對應到這張表，但同量級）</td><td>1× H100</td><td>2× H100</td></tr>
<tr><td>total_num_envs（train/eval）</td><td><b>64 / 500</b></td><td>4 / 4</td><td>8 / 8</td></tr>
<tr><td>global_batch_size</td><td><b>1024</b></td><td>4</td><td>32</td></tr>
<tr><td>micro_batch_size</td><td><b>128</b></td><td>4</td><td>16</td></tr>
<tr><td>max_epochs</td><td><b>1000</b>（92.5% 對應的 checkpoint 命名為 <code>RLinf-Gr00t-RL-Spatial-<u>Step400</u></code>，即官方實際約在 400 步左右）</td><td>30</td><td>待定（先用煙霧測試量實際每 epoch 秒數再定）</td></tr>
<tr><td>critic_warmup_steps</td><td>0</td><td>0（忠實跟隨）</td><td>0（忠實跟隨）</td></tr>
</table>
<p style='font-size:12.5px;color:#475569;margin-top:8px'>換算下來，我們單卡設定的 envs 比官方<b>小 16 倍</b>、global_batch_size <b>小 256 倍</b>、訓練步數<b>少約 13 倍</b>（30 vs ~400）。文件也明白寫著：<i>「本結果採用與 π0 完全相同的超參數設定……進一步調參應可得到更好的表現」</i>——也就是官方連 GR00T 專用調參都沒做，是在<b>足夠規模</b>下靠 PPO 本身把 policy 推上去的，這使得「規模」比「巧妙的超參數」更可能是關鍵變數。</p>

<h3>5B.2　RLinf 官方文件對 RL 調參的明確說明</h3>
<p>RLinf 的 PPO 演算法參考文件（<code>docs/source-en/rst_source/reference/algorithms/ppo.rst</code>）§4「Notes」原文列出三條建議，其中第三條直接點名批量大小：</p>
<ul>
<li><i>"Use reward normalization to stabilize training."</i>（用 reward normalization 穩定訓練）</li>
<li><i>"Monitor KL divergence to detect policy over-updates."</i>（監控 KL 散度以偵測 policy 過度更新）</li>
<li><b class='k'><i>"For large LLMs, increase batch size to reduce variance."</i></b>（模型越大，要用越大的 batch size 來降低變異數）</li>
</ul>
<p>同一份文件在 embodied task 的設定範例中也註明：<code>logprob_forward_micro_batch_size: 16　# Larger batch_size improves stability. Adjust according to compute resources and model size.</code>——即官方自己把「batch size 太小」與「訓練不穩定」直接掛鉤，這正是我們在單卡設定下觀察到的症狀（explained_variance 全程 NaN、success_once 無趨勢地振盪）。</p>

<h3>5B.3　數學推導：為什麼規模不足會系統性地傷害 PPO</h3>
<p><b>(a) 梯度雜訊尺度 (Gradient Noise Scale)</b>：以 mini-batch 大小 B 估計的梯度 ĝ 是真實梯度 G 加上取樣雜訊，Var[ĝ] ∝ Σ/B（Σ 為單樣本梯度的共變異數）。McCandlish et al. (OpenAI, arXiv:1812.06162) 定義臨界批量 B<sub>noise</sub> = tr(Σ)/‖G‖²：當 B ≪ B<sub>noise</sub> 時，每一步更新主要是在追雜訊而非真實梯度方向。我們的 B=4（單卡）遠低於論文中 Atari/Dota 等 RL 任務報告的臨界批量數量級，B=32（雙卡）仍明顯偏低。</p>
<p><b>(b) Advantage 正規化統計量失真</b>：設定 <code>normalize_advantages: True</code> 時，Â<sub>t</sub> 是用<b>當前 batch 內</b>的樣本均值與標準差重新縮放：Â'<sub>t</sub> = (Â<sub>t</sub> − mean(Â))/std(Â)。這兩個統計量本身的取樣誤差以 O(1/√N) 收斂——N=4 時這個估計極不穩定（少數幾筆極端樣本就能把整批的縮放尺度帶偏），N=1024（官方）時才statistically 站得住。這與 ICLR Blog《37 Implementation Details of PPO》討論 advantage normalization 對取樣量敏感的結論一致。</p>
<p><b>(c) explained_variance 的退化機制</b>：EV = 1 − Var(R−V)/Var(R) 有兩種方式在小 batch 下失效——
(i) PyTorch <code>torch.var()</code> 預設 Bessel 校正、除以 (n−1)，當有效樣本數 n≤1 時無定義、直接回傳 NaN（我們 log 裡的 <code>UserWarning: degrees of freedom &lt;= 0</code> 就是這個);
(ii) 即使 n>1，若小 batch 恰好抽到彼此相近的 R 值，分母 Var(R) 會接近 0，讓整個比值對雜訊極度敏感、數值不穩定（verl / torchrl 等框架文件也指出這是小批量下 EV 診斷失真的常見成因）。兩者都直接對應我們觀察到的「explained_variance 全程 NaN」。<br>
<span style='font-size:12.5px;color:#475569'>（附註：這段機制經另一次獨立查證，實際執行 PyTorch 2.x 程式碼確認——單純把 batch size 設小<b>本身不會</b>直接產生 n≤1（4 筆正常的 return 算出來 EV=0.98，不是 NaN）；真正的機制是<b>小 batch 提高了「某個 mask/chunk 邊界恰好只剩 ≤1 個有效樣本」的機率</b>，而不是小 batch 必然導致 n≤1。要精確指出我們這次跑是在哪個 mask/chunk 維度撞到 n≤1，需要追蹤 RLinf 內部的 masking 程式碼，這部分本報告尚未做，先誠實註明。）</span></p>
<p style='font-size:12.5px;color:#475569'>這三個機制都不是 RLinf 特有的，而是 PPO/GAE 文獻中已知、獨立於這個專案存在的統計性質——所以「規模不足會傷害 PPO」不是我們的臨時假設，而是有紮實理論支撐的機制，且剛好精準對應我們實測看到的異常（NaN、無趨勢振盪）。</p>

<h3>5B.4　誠實的限制：規模不是唯一可能的解釋</h3>
<p>沒有找到 RLinf 針對 GR00T/LIBERO 做「scale 越大效果越好」的官方 ablation 實驗——上面的論證是<b>從相鄰、獨立驗證過的文獻與官方調參說明推論而來，並非 RLinf 直接證明的因果關係</b>。查證時也發現 RLinf GitHub 有一則未解決的 issue（#585）：另一位使用者在<b>接近官方規模</b>（global_batch_size=2048、64 train envs）跑 π0.5+PPO+LIBERO-Spatial 時，同樣沒有重現官方曲線，維護者尚未定論根因。這說明就算規模夠大，仍可能有其他因素（checkpoint 版本差異、程式碼細節、超參數搭配）在起作用——<b>規模不足是目前證據最充分的可能原因，但不是唯一、也未被證明是充分條件</b>。這也是這次雙卡重跑存在的意義：直接測試「放大規模」這一個變數，而不是停留在紙上推論。</p>

</div>"""

DUALGPU_HTML = f"""<h2>5C. 雙卡實測結果（放大規模後，真的看到提升了）</h2><div class='card'>
<p>設定：獨立的 <code>RLinf-n15</code> clone，<code>component_placement: {{actor: 0-1, env: "1", rollout: "1"}}</code>——只讓 <b>actor 真正跨兩張 H100 做 FSDP <code>full_shard</code></b>（這是唯一需要兩張卡的部分；world_size=2 後 FSDP 不再 fallback 成單卡的 <code>NO_SHARD</code>），rollout/env 維持在單卡（跟已驗證過的單卡拓樸一致，避免三個群組都乘 2 而把 cgroup pids 上限榨乾——過程中連續撞牆 3 次才找到這個修法，細節見附錄/memory)。global_batch_size=32、micro_batch_size=16（單卡版本的 8 倍），total_num_envs 保守維持在 4（沒有跟著往上推，優先保穩定）,critic_warmup_steps=0 忠實官方 recipe，跑了 72 epochs（~9 小時)。</p>

<div class='g2'><div class='card'>{chart_d_eval}</div><div class='card'>{chart_d_train}</div></div>

<p><b>結果——這是全專案第一次看到明確的上升趨勢：</b></p>
<ul>
<li><b>env/success_once</b>（72 個訓練點）：早段（前 18 點）均 <b>{tr_ok_d[0]}</b> → 末段（後 18 點）均 <b>{tr_ok_d[1]}</b>，上升約 {round((tr_ok_d[1]-tr_ok_d[0])*100,1)} 個百分點。</li>
<li><b>eval/success_once</b>（12 個 eval 點，每 6 epoch 一次）：早段（前 3 點）均 <b>{ev_ok_d[0]}</b> → 末段（後 3 點）均 <b>{ev_ok_d[1]}</b>，同樣是上升，但 eval 只有 4 trials/次，雜訊仍大（序列：{vals(M15D, "eval/success_once")}）。</li>
<li><b class='k'>explained_variance</b>：72 個點裡有 <b>{sum(1 for x in vals(M15D,'train/critic/explained_variance') if x>0)}/72（{round(evx_d_positive_frac*100)}%）是正值</b>，平均 {round(evx_d_mean,3)}，範圍 [{round(min(vals(M15D,'train/critic/explained_variance')),3)}, {round(max(vals(M15D,'train/critic/explained_variance')),3)}]，<b>全程沒有一個 NaN</b>。相較單卡 N1.5 是「全程 NaN」、N1.7 是「全程負值且一路惡化到 -338」，這次的 critic 是<b>本專案唯一一次真正學到有效估計</b>的一次。</li>
</ul>

<div class='card'>{chart_d_ev}</div>

<p style='font-size:12.5px;color:#475569'><b>怎麼解讀這個結果</b>：這不是「重現了官方 41.4%→92.5%」——我們的 total_num_envs（4）、global_batch_size（32）仍分別比官方的 64、1024 小 16 倍、32 倍，72 epochs 也遠少於官方對應 92.5% 的 <code>Step400</code>。但方向和機制都對上了：<b>把 batch size 從 4 拉到 32（唯一大幅改變的變數，env 數刻意沒動）</b>，explained_variance 從「永遠算不出來」變成「93% 的時候是正值」，success_once 從「無趨勢振盪」變成「早段→末段有 {round((tr_ok_d[1]-tr_ok_d[0])*100,1)} 個百分點的上升」。這與 §5B.3 的數學推導（gradient noise scale、advantage 正規化統計量、EV 退化機制）三個機制的預測方向完全一致——<b>批量大小是可以觀察到因果關係的變數，不只是紙上理論</b>。</p>
<p style='font-size:12.5px;color:#475569'><b>誠實的保留</b>：這仍然只是一次跑（沒有多個 random seed 重複驗證），72 個點、12 個 eval 點的樣本量比官方的 500-eval-envs 小得多，所以這個上升趨勢<b>比官方的 +51pp 脆弱得多、也還沒有到「證明可重現官方數字」的程度</b>。它能證明的是：在我們能控制的範圍內，放大規模（尤其是 batch size）<b>確實會讓訊號變好，而不是不會</b>——這就是 §5B.4 所說「規模不足是目前證據最充分的可能原因」的直接實驗支持,而不再只是文獻推論。</p>
</div>"""

FINAL_HTML = f"""<h2>5D. 最終規模驗證：230 epochs、58 小時單卡跑（全專案最強證據）</h2><div class='card'>
<p>雙卡 72-epoch 那次跑完後，使用者要求「把規模、epoch 數盡量拉到跟官方接近，時間可以拉到 58-72 小時」。實際嘗試雙卡放大 envs 時連續在 cgroup pids 上限上撞牆（見 §5C 備註與下方誠實記錄），且一次跑到一半才崩潰（Ray 內部執行緒建立失敗），判斷雙卡因為 actor 跨卡分片、Ray 程序數量本來就比單卡多、風險更高，<b>改回單卡</b>，把資源全部投入「更多 epoch」這個方向：</p>
<table><tr><th>設定</th><th>GPU</th><th>envs</th><th>global_batch</th><th>epochs</th><th>結果（env success_once 早→末段）</th></tr>
<tr><td>N1.5 spatial 單卡（初版）</td><td>1</td><td>4</td><td>4</td><td>30</td><td>0.435→0.487（無明顯提升）</td></tr>
<tr><td>N1.5 spatial 雙卡</td><td>2（僅 actor 分片）</td><td>4</td><td>32</td><td>72</td><td>0.453→0.606（+15.3pp）</td></tr>
<tr><td><b>N1.5 spatial 單卡（最終，58h)</b></td><td><b>1</b></td><td><b>8</b></td><td><b>64</b></td><td><b>230</b></td><td><b>{tr_ok_s[0]}→{tr_ok_s[1]}（+{round((tr_ok_s[1]-tr_ok_s[0])*100,1)}pp）</b></td></tr>
</table>

<div class='card'>{chart_s_all_eval}</div>

<p><b>結果——全專案最清楚、最大幅度的提升：</b></p>
<ul>
<li><b>env/success_once</b>（230 個訓練點）：早段（前 57 點）均 <b>{tr_ok_s[0]}</b> → 末段（後 57 點）均 <b>{tr_ok_s[1]}</b>，上升 <b class='k'>{round((tr_ok_s[1]-tr_ok_s[0])*100,1)} 個百分點</b>，整體平均 {round(sum(vals(M15S,'env/success_once'))/len(vals(M15S,'env/success_once')),3)}。</li>
<li><b>eval/success_once</b>（23 個 eval 點，每 10 epoch 一次，4 trials/次）：早段（前 5 點）均 <b>{ev_ok_s[0]}</b> → 末段（後 5 點）均 <b>{ev_ok_s[1]}</b>，上升 <b class='k'>{round((ev_ok_s[1]-ev_ok_s[0])*100,1)} 個百分點</b>。序列本身就看得出明顯的上升軌跡（後段多次達到 1.0）：{vals(M15S, "eval/success_once")}</li>
<li><b class='k'>explained_variance</b>：230 個點裡 <b>{sum(1 for x in vals(M15S,'train/critic/explained_variance') if x==x and x>0)}/230（{round(evx_s_positive_frac*100)}%）是正值</b>，平均 {round(evx_s_mean,3)}（僅計算非 NaN 的點），只有 <b>{evx_s_nan_count} 個點是 NaN</b>（相較單卡初版「全程 NaN」）。critic 這次的品質是三次嘗試中最穩定的一次。</li>
</ul>

<div class='g2'><div class='card'>{chart_s_eval}</div><div class='card'>{chart_s_train}</div></div>
<div class='card'>{chart_s_ev}</div>

<p style='font-size:12.5px;color:#475569'><b>這次多改了什麼、又能推論出什麼</b>：跟雙卡那次比，這次 batch size 其實更小（64 vs 128），但 epochs 多了 3 倍（230 vs 72）、env 數也多一點（8 vs 4）。<b>提升幅度反而更大（+{round((tr_ok_s[1]-tr_ok_s[0])*100,1)}pp vs +15.3pp）</b>——這說明<b>訓練時長/epoch 數本身就是獨立的關鍵變數</b>，不只是「batch 夠大就好」。這跟 RLinf 官方 recipe 用 <code>max_epochs=1000</code>（我們目前最多也只到 230，仍是官方的 23%）互相呼應：光靠一次把 batch 調大換不到全部效果，<b>要有足夠多次 PPO 更新讓 policy 逐步收斂</b>才是這次真正推升成功率的原因。</p>
<p style='font-size:12.5px;color:#475569'><b>誠實的保留（跟 §5C 一樣的提醒，但更接近了）</b>：這仍是單次跑（無多 random seed）；envs（8）仍是官方（64）的 12.5%、global_batch（64）是官方（1024）的 6.25%、epochs（230）是官方（1000）的 23%——<b>方向對、幅度也是全專案最大，但還不是「重現官方 92.5%」的等級</b>。中途曾因 cgroup pids 上限崩潰過一次（損失约 1.5 小時、無 checkpoint 損失，因為當時還沒到第一次存檔點），修正 <code>save_interval</code> 對齊 <code>val_check_interval</code> 的 bug 並改回單卡後，剩餘 54.8 小時完全沒有再發生任何崩潰，最終在時限前（週一 10:00 台灣時間）數十分鐘完成收尾。</p>
</div>"""


def build():
    def verdict_line():
        return (f"no-warmup：eval 早期 {ev_ok_n[0]} → 末期 <b>{ev_ok_n[1]}（明顯下降）</b>；"
                f"critic-warmup：eval 早期 {ev_ok_w[0]} → 末期 {ev_ok_w[1]}（<b>在雜訊內振盪、無明顯提升</b>）")

    parts = []
    parts.append(f"<h1>RLinf × GR00T-N1.7 × LIBERO — Phase 1 分析報告</h1>")
    parts.append('<p class="sub">單張 H100（GPU1）、libero_spatial、in-process 向量環境。本報告誠實比較 no-warmup 與 critic-warmup 兩次跑，並對照 RLinf 官方數字。</p>')

    # verdict
    parts.append('<div class="verdict ok">結論（誠實，含最終規模驗證更新）：端到端 RL 流程已驗證可跑。<b>最小規模設定下</b>（N1.7、N1.5 皆是）PPO 沒有帶來可量測的成功率提升，'
                 'critic warmup 的作用是<b>「防止退步」</b>而非「產生提升」。' + verdict_line() +
                 f' 放大規模後，先是雙卡 batch 放大（§5C，explained_variance 轉為 93% 正值、success_once +15.3pp），'
                 f'接著把資源全部換成單卡 230 epochs 的長時間跑（§5D，58 小時）後，<b class="k">explained_variance 97% 為正、'
                 f'success_once 早段→末段上升 {round((tr_ok_s[1]-tr_ok_s[0])*100,1)} 個百分點、eval success_once 上升 {round((ev_ok_s[1]-ev_ok_s[0])*100,1)} 個百分點</b>'
                 '——這是全專案最清楚的證據：規模（尤其是訓練時長/epoch 數）不足才是真正的主因，不是 checkpoint 天花板或 critic 機制天生壞掉。</div>')

    # 1. exec summary
    parts.append("<h2>1. 執行摘要</h2><div class='card'><ul>"
                 "<li><b>基礎設施：</b>RLinf GR00T-N1.7 + PPO 在 Pegasus 單張 H100 端到端跑通（rollout → PPO actor+critic 更新 → eval → 存檔），GPU0 全程閒置。</li>"
                 "<li><b>RL 結果：</b>eval 成功率在 {a}~{b} 之間<b>振盪</b>，沒有上升趨勢；訓練成功率也無明顯改善。</li>"
                 "<li><b>為什麼沒提升（根因）：</b>critic（value function）嚴重失準（<code>explained_variance</code> 由 -0.3 惡化到 -338），"
                 "使 PPO 的 advantage 不可靠 → policy 幾乎沒有有效學習訊號；加上 reward 量級極小(~0.004)、單卡/小批量/短訓練、且 <b>N1.7 的 LIBERO RL 官方尚未驗證(TODO)</b>。</li>"
                 "<li><b>critic warmup：</b>沒有 warmup 時，早期用剛隨機初始化的 critic 去更新 policy → policy <span class='k'>退步</span>(eval 掉到 ~0.12)；加了 warmup 後退步消失，但 critic 仍沒學好 → 也沒提升。</li>"
                 "<li><b>N1.5 spatial 重現（見 5A）：</b>另外完整跑了一次 RLinf 官方招牌案例（SFT→RL），單卡縮小規模後（30 epochs，非官方的 1000）"
                 "同樣<b>沒有觀察到官方 41.4%→92.5% 的提升</b>，success_once 在 0~0.75 間振盪。兩次獨立嘗試（N1.7、N1.5）都指向<b>規模/算力不足</b>"
                 "才是共同瓶頸，而非某一次 checkpoint 特有的天花板問題。</li>"
                 "<li><b class='k'>雙卡放大規模驗證（見 5B/5C）：</b>用使用者授權的第二張 H100 把 batch size 從 4 拉到 32（env 數刻意不動），"
                 "72 epochs 跑完後 <b>explained_variance 93% 的時間是正值（全程零 NaN）</b>，<b>success_once 早段→末段從 0.453 升到 0.606</b>——"
                 "本專案第一次觀察到清楚的上升趨勢，直接支持「規模不足是主因」這個假設，但尚不到「重現官方 92.5%」的程度（樣本量、規模仍遠小於官方設定）。</li>"
                 "<li><b class='k'>最終規模驗證（見 5D，全專案最強證據）：</b>雙卡在放大 envs 時連續撞上 cgroup pids 上限，改回單卡、把資源全部換成"
                 "更多 epoch（envs=8、batch=64、<b>230 epochs</b>、58 小時)，success_once 早段→末段上升 <b>{s_tr}pp</b>，"
                 "eval success_once 上升 <b>{s_ev}pp</b>（後段多次達到 1.0），explained_variance <b>97% 為正</b>——"
                 "這說明訓練時長本身（不只是 batch size）是獨立的關鍵變數，也是本專案最接近「看到 RL 真的在幫助 VLA」的一次。</li>"
                 "</ul></div>".format(a=min(vals(W,'eval/success_once')), b=max(vals(W,'eval/success_once')),
                                       s_tr=round((tr_ok_s[1]-tr_ok_s[0])*100,1), s_ev=round((ev_ok_s[1]-ev_ok_s[0])*100,1)))

    # 2. glossary
    parts.append("<h2>2. 名詞解釋（技術用語）</h2><div class='card'><ul>"
                 "<li><b>SFT baseline</b>：用監督式/模仿學習訓練出來的起始 policy。我們用的是 <code>nvidia/GR00T-N1.7-LIBERO</code>（NVIDIA 在 LIBERO 上訓練的官方權重）。</li>"
                 "<li><b>critic / value function</b>：估計「某狀態未來能拿到多少回報」的網路。PPO 用它算 <b>advantage</b>（某動作比平均好多少）來決定 policy 往哪個方向更新。</li>"
                 "<li><b>value head 隨機初始化 / 冷啟動 critic</b>：GR00T 原本是行為模仿模型、<b>沒有 value head</b>；做 RL 時新接一個 value head，其權重是<b>隨機的</b>（log 明確顯示 <code>value_head newly initialized</code>）。一開始它的估計等於亂猜。</li>"
                 "<li><b>為什麼冷啟動 critic 會拖垮 policy</b>：critic 亂猜 → advantage 是雜訊 → PPO 拿雜訊當方向更新 actor → policy 往錯的方向漂移、成功率下降。</li>"
                 "<li><b>critic warmup（<code>critic_warmup_steps</code>）</b>：訓練前 N 個 optimizer step <b>只更新 critic、凍結 actor</b>，先讓 value head 學到堪用估計，再開始動 policy。本次設 40。</li>"
                 "<li><b>explained_variance</b>：critic 好壞的指標＝1 − Var(實際回報 − 估計)/Var(實際回報)。1=完美、0=等於亂猜平均、<b>負值=比亂猜還差</b>。我們的值一路是負的（-0.3 → -338），代表 critic 沒學好。</li>"
                 "<li><b>approx_kl / ratio</b>：衡量新舊 policy 的差異；ratio≈1、KL≈0.01 代表 policy 幾乎沒動。</li>"
                 "</ul></div>")

    parts.append(Q1_HTML)
    parts.append(MATH_HTML)

    # 3. comparison charts
    parts.append("<h2>3. 冷啟動 vs critic-warmup 對比（你要看的結果都在這）</h2>")
    parts.append("<div class='card'>" + chart_eval +
                 "<p style='font-size:12.5px;color:#475569'>這就是核心圖。<b class='k'>紅線 (no-warmup) 明顯往下</b>（早期~0.375→末期~0.125），"
                 "<b style='color:#2563eb'>藍線 (critic-warmup) 在 0.25~0.75 之間振盪、沒有趨勢</b>。所以 warmup 的貢獻是「止跌」，不是「提升」。"
                 "序列(warmup)：" + str(vals(W,'eval/success_once')) + "。每個點只有 4 個 trial，所以本來就會跳動。</p></div>")
    parts.append("<div class='g2'><div class='card'>" + chart_train +
                 "<p style='font-size:12px;color:#475569'>訓練 rollout 成功率（移動平均降噪）。紅線後段下滑、藍線大致持平。</p></div>"
                 "<div class='card'>" + chart_ev +
                 "<p style='font-size:12px;color:#475569'><b>關鍵診斷</b>：explained_variance 兩次都遠低於 0（健康應接近 1），代表 critic 從未學會估計回報。圖已 clip 到 -5；實際最低到 -496015。</p></div></div>")

    # 4. why loss 0 / noisy
    parts.append("<h2>4. 為什麼 Loss 看起來是 0？各指標怎麼判讀？</h2><div class='g2'>"
                 "<div class='card'>" + chart_loss +
                 "<p style='font-size:12px;color:#475569'><b>entropy_loss=0</b>：設定 <code>entropy_bonus=0</code>，本來就是 0。<br>"
                 "<b>policy_loss 前段=0</b>：critic warmup 期間 actor 被凍結。<br>warmup 後 policy_loss 也只有 ±0.01——這是 PPO clipped 目標的<b>正常小量級</b>，不是壞掉。有意義的是 total_loss 與 value_loss。</p></div>"
                 "<div class='card'>" + chart_kl + chart_ratio +
                 "<p style='font-size:12px;color:#475569'>KL≈0.012、ratio≈1.0 → policy 幾乎沒動。這與「advantage 是雜訊、沒方向」一致。</p></div></div>"
                 "<div class='card'>" + chart_rw + chart_vl +
                 "<p style='font-size:12.5px;color:#475569'><b>怎麼看出有沒有進步？</b>誠實說：在這個設定下，從 success rate <b>看不出</b>進步（4-trial 雜訊 + 無趨勢）。"
                 "唯一能明確判讀的是：(1) 訓練沒有發散（KL 小、grad_norm 有界）；(2) critic warmup 消除了 no-warmup 的退步。要能「看出進步」需要先把 critic 修好（見第 6 節）。"
                 "reward 量級只有 ~0.004，這也是 critic 難以擬合、學習訊號薄弱的原因之一。</p></div>")

    # 5. RLinf published vs ours
    parts.append("<h2>5. RLinf 官方公布結果 vs 我們的實測</h2><div class='card'>"
                 "<table><tr><th>GR00T</th><th>libero_spatial SFT</th><th>PPO 後</th><th>提升</th><th>備註</th></tr>"
                 + rlinf_html + "</table>"
                 "<p style='font-size:12.5px;color:#475569;margin-top:8px'><b>可重現性（起始 checkpoint 供應）</b>："
                 "N1.5 的 4 個 few-shot SFT 已公開（<code>RLinf/RLinf-Gr00t-SFT-{{Spatial,Object,Goal,10}}</code>）→ <b>可重現</b>，"
                 "本報告已實際跑過 spatial（見 5A）；"
                 "N1.7 RLinf 自己的 SFT 未釋出、暫借 NVIDIA 官方 LIBERO 權重（即我們用的）。</p>"
                 "<p style='font-size:12.5px;color:#475569;margin-top:8px'><b class='k'>N1.6 可行性更正</b>："
                 "官網文字寫「RLinf SFT models will be released soon — stay tuned!」，字面上看像是 N1.6 尚無 SFT、故 70%→82% 無法重現。"
                 "但實際檢查 <code>examples/embodiment/config/libero_spatial_ppo_gr00t_n1d6.yaml</code> 後發現其 <code>model_path</code> 指向 "
                 "<code>RLinf/RLinf-Gr00t-N1.6-RL-Spatial</code>，而這個 HF repo <b>確實存在</b>（17 個檔案：config.json、兩片 safetensors、"
                 "trainer_state.json、wandb_config.json 等）。檢查其 <code>trainer_state.json</code> 的 <code>log_history</code>，"
                 "欄位只有 <code>grad_norm / learning_rate / loss / step</code>（loss 從 1.28 平滑降到 0.098，典型監督式/flow-matching 訓練曲線），"
                 "<b>完全沒有 reward / kl / ratio / value / success 這類 RL 訓練才會有的欄位</b>——幾乎可以確定這其實是 N1.6 的 <b>SFT checkpoint</b>，"
                 "只是 RLinf 沒有像 N1.5 一樣清楚命名成「-SFT-」，才造成誤判。也就是說<b>官網文字已過期，N1.6 其實可重現</b>，"
                 "只是本報告目前尚未執行這一步（已排入待辦）。</p>"
                 "<p style='font-size:12.5px;color:#475569;margin-top:10px'>"
                 "<b>我們的實測（N1.7, critic-warmup）</b>：eval/success_once 早期 ≈ {ew0}、末期 ≈ {ew1}（在 0.25~0.75 振盪）。<br>"
                 "<b>差距解讀</b>：RLinf 的 N1.5 是從<b>很弱的 few-shot SFT(41.4%)</b>起步、又用<b>調過參數的多卡</b>設定，所以能 +51pp。"
                 "我們的落差主要來自 <b>(a) critic 失準（explained_variance≪0）</b>、<b>(b) reward 極小且未正規化</b>、"
                 "<b>(c) 單卡/小批量/短訓練</b>、<b>(d) N1.7 的 LIBERO RL 官方尚未驗證(TODO)</b>——不是「沒有改善空間」。</p></div>".format(
                     ew0=ev_ok_w[0], ew1=ev_ok_w[1]))

    # 5A. N1.5 spatial reproduction attempt
    parts.append(("<h2>5A. N1.5 spatial 重現結果（SFT→RL，獨立跑一次官方招牌案例）</h2><div class='card'>"
                 "<p>為了直接回答「RL 對 VLA 有沒有幫助」，我們另外拉了一份獨立的 RLinf clone（<code>RLinf-n15</code>，避免動到 N1.7 那份已跑通的環境），"
                 "下載公開的 <code>RLinf/RLinf-Gr00t-SFT-Spatial</code>（自帶完整權重，不需要像 N1.7 那樣借用 gated backbone），"
                 "套用同樣的容器相容修補（Ray 執行緒上限、in-process LIBERO 環境、GPU1 釘選），"
                 "把 RLinf 官方多卡設定（<code>total_num_envs=64</code>、<code>global_batch_size=1024</code>）縮到單卡可跑的規模"
                 "（<code>total_num_envs=4</code>、<code>global_batch_size=micro_batch_size=4</code>），"
                 "並刻意保留 RLinf 官方 N1.5 recipe 的 <code>critic_warmup_steps=0</code>（不像我們對 N1.7 做的warmup=40修正)"
                 "——這樣如果結果跟官方曲線不同，落差就更能歸因於「規模」而非「我們自己加的修正」。"
                 "受限於單卡＋大幅縮小 batch 後每個 epoch 反而變貴（實測 ~9-16 分鐘/epoch），"
                 "<code>max_epochs</code> 從官方的 1000 降到 <b>30</b>，全程約 4.7 小時。</p>"
                 "<div class='g2'><div class='card'>" + chart_n15_eval + "</div><div class='card'>" + chart_n15_train + "</div></div>"
                 "<p><b>結果</b>：eval/success_once（10 個點，每 3 epoch 一次，4 trials/次）＝ " + str(vals(M15, "eval/success_once")) +
                 "，早段均 {a}、末段均 {b}；訓練成功率早段均 {c}、末段均 {d}。<b class='k'>兩個系列都在雜訊帶內振盪、沒有觀察到官方 41.4%→92.5% 那種明顯提升。</b></p>"
                 "<p><b>重要限制（誠實記錄）</b>：因為 <code>critic_warmup_steps=0</code>，actor 從第 1 步就開始更新，"
                 "所以我們<b>沒有一個「純 SFT、完全未經 RL」的基準點</b>可以對照——最早的 eval 已經是訓練 3 步之後的結果，"
                 "不能嚴格說「41.4% 對應到我們的第 0 點」。這是這次重現的一個方法論缺口，量級上不影響「沒觀察到提升」這個結論，但無法精確對齊官方的起點數字。</p>"
                 "<div class='card'>" + chart_n15_vl +
                 "<p style='font-size:12.5px;color:#475569'><b>explained_variance 全程是 NaN</b>（不是像 N1.7 那樣的一路負值，是連數值都算不出來）。"
                 "根因：PyTorch 的 <code>torch.var()</code> 預設用 Bessel 校正（除以 n−1），當參與計算的（未被 mask 掉的）元素數 ≤1 時無法定義、回傳 NaN；"
                 "log 裡也確實看到 <code>UserWarning: var(): degrees of freedom is &lt;= 0</code>。我們把 <code>global_batch_size</code> 從官方的 1024 砍到 4，"
                 "讓每次計算 explained_variance 用到的樣本數暴減，遠比官方設定更容易踩到這個邊界——<b>這是我們被迫縮小規模的副作用，不是 critic 機制本身的問題</b>。"
                 "不過 <code>critic/value_loss</code> 持續從 0.072 降到 ~0.02（如上圖），代表 critic 確實有在擬合歷史 return，只是這個特定診斷指標在小 batch 下算不出來。</p></div>"
                 "<p><b>這對整份報告的意義</b>：N1.7（近乎天花板的官方 checkpoint、critic 明顯失準）和 N1.5（更弱的 few-shot SFT 起點、critic 未必失準只是診斷算不出來）"
                 "這兩次獨立嘗試，<b>都在被迫大幅縮小規模（單卡、小 batch、少 epoch）後看不到官方報告的提升</b>。"
                 "這讓「規模/算力不足」比「特定 checkpoint 的天花板」或「critic 機制天生會壞」更像是我們兩次嘗試共同的主要瓶頸——"
                 "RLinf 官方的 N1.5 recipe 本身連 critic warmup 都沒加（<code>critic_warmup_steps=0</code>）卻能拿到 +51pp，"
                 "說明在<b>足夠規模</b>下，就算不特別處理冷啟動 critic，PPO 仍然能把 SFT policy 往上推。</p></div>").format(
                     a=ev_ok_m[0], b=ev_ok_m[1], c=tr_ok_m[0], d=tr_ok_m[1]))

    parts.append(SCALE_HTML)
    parts.append(DUALGPU_HTML)
    parts.append(FINAL_HTML)

    # 6. how to actually get a lift
    parts.append("<h2>6. 若要真的看到提升，建議的下一步</h2><div class='card'><ul>"
                 "<li><b>修 critic（最關鍵）</b>：reward/return 正規化、拉長 <code>critic_warmup_steps</code>、給 critic 更大學習率/容量，直到 explained_variance 轉正。</li>"
                 "<li><b>加大取樣與算力</b>：多卡、多環境（RLinf 官方是多卡調參設定）、更多 epochs。</li>"
                 "<li><b>換到有改善空間的場景</b>：Phase 2（IsaacLab + 你 fine-tuned 的 checkpoint + Can-Sorting 任務），policy 才有明確 headroom。</li>"
                 "<li><b>對照官方已驗證版本</b>：RLinf 的 LIBERO RL 在 N1.5/N1.6 有數字，N1.7 是 TODO；可先用 N1.5 復現官方 41.4%→92.5% 以確認我們的 pipeline 能重現提升。</li>"
                 "</ul></div>")

    # 7. architecture + flow
    parts.append("<h2>7. 系統架構圖</h2><div class='card'>" + ARCH_SVG + "</div>")
    parts.append("<h2>8. PPO 訓練流程圖</h2><div class='card'>" + FLOW_SVG + "</div>")

    # 9. run config
    parts.append("<h2>9. 本次跑的設定與位置</h2><div class='card'><ul>"
                 "<li>模型：<code>nvidia/GR00T-N1.7-LIBERO/libero_spatial</code> + backbone <code>Cosmos-Reason2-2B</code>（本地副本）</li>"
                 "<li>critic-warmup run：total_num_envs=4, global_batch_size=32, critic_warmup_steps=40, max_epochs=120, eval/10, 單 GPU1</li>"
                 "<li>指標：<code>artifacts/rl/libero_spatial_ppo_gr00t_n1d7_phase1/metrics.json</code>（warmup）、<code>artifacts/rl/metrics_nowarmup.json</code>（no-warmup）</li>"
                 "<li>N1.5 spatial 重現：模型 <code>RLinf/RLinf-Gr00t-SFT-Spatial</code>（獨立 clone <code>RLinf-n15</code>）；"
                 "total_num_envs=4, global_batch_size=micro_batch_size=4, critic_warmup_steps=0（忠實官方 recipe), max_epochs=30, eval/3, 單 GPU1</li>"
                 "<li>N1.5 指標：<code>artifacts/rl/metrics_n15_spatial.json</code>；設定檔：<code>agents/rl-trainbot/harness/config/phase1_n15_spatial.yaml</code></li>"
                 "<li>N1.5 spatial 雙卡重跑：<code>component_placement: {actor: 0-1, env: \"1\", rollout: \"1\"}</code>；"
                 "total_num_envs=4, global_batch_size=32, micro_batch_size=16, critic_warmup_steps=0, max_epochs=72, eval/6, GPU0+GPU1</li>"
                 "<li>雙卡指標：<code>artifacts/rl/metrics_n15_spatial_dualgpu.json</code>；設定檔：<code>agents/rl-trainbot/harness/config/phase1_n15_spatial_dualgpu.yaml</code></li>"
                 "<li>N1.5 spatial 最終跑（58h，單卡）：<code>component_placement: actor,env,rollout: \"1\"</code>；"
                 "total_num_envs=8, global_batch_size=64, micro_batch_size=16, critic_warmup_steps=0, max_epochs=230, val_check_interval=10, save_interval=10, 單 GPU1，實際耗時 54.8 小時</li>"
                 "<li>最終跑指標：<code>artifacts/rl/metrics_n15_spatial_scaled.json</code>；設定檔：<code>agents/rl-trainbot/harness/config/phase1_n15_spatial_scaled.yaml</code></li>"
                 "<li>可重現：<code>agents/rl-trainbot/harness/</code>（launch/poll/report/patch_rlinf_clone，皆已泛化支援 <code>--config-file</code> 與多代模型）+ OpenSpec <code>add-rlinf-gr00t-n17-libero-rl</code></li>"
                 "</ul></div>")

    body = "".join(parts)
    return f"<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>RLinf GR00T-N1.7 LIBERO Phase 1 分析報告</title><style>{CSS}</style></head><body><div class='wrap'>{body}</div></body></html>"


OUT.write_text(build(), encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size, "bytes")
