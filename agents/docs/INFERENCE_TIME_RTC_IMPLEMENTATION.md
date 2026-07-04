# Inference-Time Real-Time Chunking (RTC) for GR00T N1.5 — Implementation Report

This document describes the **inference-time** Real-Time Chunking (RTC, Black et al., arXiv:2506.07339) implementation added to the OpenArm + GR00T N1.5 stack, the rationale, the math, the per-file changes, and how to validate, debug, and tune it.

> **Status (this document)**: inference-time RTC, ΠGDM-guided, no model retraining required.
> **Branch scope**: changes live inside the `Isaac-GR00T` submodule and the IsaacLab side `gr00t_control_robot.py`. The `action_chunk_controller.py` ROS node is **not** part of this pipeline and was left unmodified.
> **Companion document (future)**: `TRAINING_TIME_RTC_IMPLEMENTATION.md` will describe the training-time variant, where the freeze-conditioning is baked into the model weights and ΠGDM is replaced by a single forward pass.

---

## 1. Problem: action-chunk discontinuity

GR00T N1.5 outputs **action chunks** of `H = 16` future steps at a step duration of `dt = 1/30 s`. Each chunk is sampled by an independent flow-matching denoising loop seeded from a fresh Gaussian. Because of:

- stochastic sampling noise across chunks,
- different observations between the start of chunk N and chunk N+1,
- the controller preempting the in-flight chunk when a new one arrives,

…**adjacent chunks disagree at the boundary**. On hardware this manifests as a small jerk, backtracking, or audible motor whine at every chunk-replacement event, especially on contact-rich tasks (e.g., LinkerHand grasp).

A naïve low-pass filter on the joint commands hides the symptom but adds phase lag — particularly damaging for the multi-DOF hand. RTC fixes the cause.

---

## 2. RTC principle (and how it maps to GR00T)

### 2.1 Flow-matching sampler (vanilla)

GR00T's `FlowmatchingActionHead.get_action` integrates an ODE backwards from noise:

```
A_τ=0 ~ N(0, I)
for τ in [0, 1/N, 2/N, …]:
    v = velocity_net(A_τ, τ, conditioning)
    A_{τ + dt} = A_τ + dt · v
return A_{τ=1}                # clean predicted chunk
```

Forward (training) noising used by GR00T:

```
A_τ = (1 − τ) · noise + τ · A_clean
```

**GR00T time convention** (verified at [`flow_matching_action_head.py`](../Isaac-GR00T/gr00t/model/action_head/flow_matching_action_head.py)):

| τ | Pi paper | GR00T |
|---|---|---|
| 0 | clean | **noisy** |
| 1 | noisy | **clean** |

This convention flip propagates into the RTC inpainting formula (§2.3).

### 2.2 The RTC mask: hard prefix + soft tail

Given:
- `H` — chunk horizon (16)
- `d` — frozen prefix length (steps that overlap with the previous chunk's still-executing tail)
- `s` — execution horizon (steps before the next chunk arrives), with `d ≤ s ≤ H − d`

RTC defines a per-step weight `W ∈ [0, 1]^H`:

```
W[i] = 1                                   for i ∈ [0, d)         — hard freeze
W[i] = c_i · (e^{c_i} − 1) / (e − 1)        for i ∈ [d, H − s)     — soft decay
W[i] = 0                                   for i ∈ [H − s, H)     — free tail
where c_i = (H − s − i) / (H − s − d + 1)
```

Implemented vectorised in `_compute_rtc_soft_mask()`.

### 2.3 ΠGDM guidance (Black et al. Eq. 2)

At each denoising step, estimate the clean action from the current noisy `A_τ`:

```
A_hat_1 = A_τ + (1 − τ) · v
```

Define the soft-masked error against the previous chunk:

```
err = (A_prev − A_hat_1) ⊙ W ⊙ dim_mask
```

`dim_mask` zeroes out **padded** action dimensions (real dims are 13 for `openarm_linkerhand_o6`; padded to 32 by `GR00TTransform`).

Compute the guidance direction by VJP through the DiT:

```
g = err · ∂A_hat_1/∂A_τ
```

In autograd this is equivalent to (and implemented as):

```python
g = autograd.grad((err.detach() * A_hat_1).sum(), inputs=A_τ)[0]
```

**Sign check**: `err > 0` ⇒ `A_prev > A_hat_1`; the dot-product gradient pushes `A_τ` so that `A_hat_1` increases toward `A_prev`. Adding `+weight · g` in the Euler step is the descent direction on the masked error. ✓

Guidance weight schedule (capped near τ=0 to avoid the `1/τ` singularity):

```
r²    = (1 − τ)² / (τ² + (1 − τ)²)
weight = min(β, (1 − τ) / (τ · r²))     for τ > 1e−4
weight = β                              for τ ≤ 1e−4
```

Final Euler step:

```
A_{τ + dt} = A_τ + dt · (v + weight · g)
```

### 2.4 No retraining required

ΠGDM is **inference-time only**. The pretrained velocity network is never updated. Off-policy fine-tunes (e.g. the openarm_cansorting_N15_fft checkpoint) work as-is.

---

## 3. End-to-end architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CLIENT  (gr00t_control_robot.py)                                        │
│                                                                          │
│  inference_thread ─────────────────────────► control_thread              │
│      │                                            │                      │
│      │ obs + rtc.{freeze_prefix, freeze_len,      │ FollowJointTrajectory│
│      │            stride, beta}                   ▼                      │
│      ▼                                       ros2_control                │
│  Gr00tClientAdapter (ZMQ REQ)                                            │
│      │                                                                   │
└──────┼───────────────────────────────────────────────────────────────────┘
       │
       │  torch.save{obs}
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  SERVER  (inference_service.py + service.py)                             │
│                                                                          │
│  RobotInferenceServer.run()                                              │
│      └─► policy.get_action(obs)                                          │
│              ├─ pop rtc.* from obs                                       │
│              ├─ apply_transforms (normalize state/video)                 │
│              ├─ _prepare_rtc_freeze:  per-limb normalize → concat → pad  │
│              │                       returns (tensor, real_dim)          │
│              ├─ stash on action_head:  _rtc_freeze, _rtc_freeze_len,     │
│              │                          _rtc_stride, _rtc_beta,          │
│              │                          _rtc_real_dim                    │
│              ├─ dispatch:                                                │
│              │     use_rtc=False → torch.inference_mode + get_action     │
│              │     use_rtc=True  → torch.no_grad      + guided_get_action│
│              ├─ unapply_transforms (denormalize action)                  │
│              └─ return per-limb action chunks                            │
│                                                                          │
│  guided_get_action  (flow_matching_action_head.py)                       │
│      ├─ build A_prev, soft mask W, dim_mask                              │
│      └─ for τ in denoising loop:                                         │
│            v = DiT_forward(A_τ, …)                                       │
│            A_hat_1 = A_τ + (1 − τ) · v                                   │
│            err     = (A_prev − A_hat_1) ⊙ W ⊙ dim_mask                   │
│            g       = autograd.grad((err.detach() · A_hat_1).sum(), A_τ)  │
│            weight  = min(β, (1−τ)/(τ·r²))                                │
│            A_{τ+dt} = A_τ + dt · (v + weight · g)                        │
└──────────────────────────────────────────────────────────────────────────┘
```

The `inference_service.py` REQ/REP layer is **transparent** — the ZMQ codec uses `torch.save` so arbitrary `rtc.*` keys pass through without protocol changes.

---

## 4. Files modified

### 4.1 [`Isaac-GR00T/gr00t/model/action_head/flow_matching_action_head.py`](../Isaac-GR00T/gr00t/model/action_head/flow_matching_action_head.py)

Two methods on `FlowmatchingActionHead`:

| Method | Purpose | Decorator | Outer policy ctx |
|---|---|---|---|
| `get_action` | Original vanilla flow-matching sampler. Untouched in behaviour. | `@torch.no_grad()` | `torch.inference_mode()` |
| `guided_get_action` | **New** — ΠGDM-guided RTC sampler. Reads `_rtc_*` stashes. | `@torch.no_grad()` (with `enable_grad()` inside the VJP loop) | `torch.no_grad()` |

Helpers:

- `_compute_rtc_soft_mask(d, s, H, device, dtype)` — vectorised soft mask per §2.2.
- `_run_denoise_step(actions, …)` — single DiT forward producing the velocity, shared between RTC and (intentionally not) vanilla paths.

Constraints enforced in `guided_get_action`:

```python
d = max(0, min(rtc_len, H - 1))         # d ≤ H − 1
s = rtc_stride if rtc_stride > 0 else d
s = max(d, min(s, H - d))               # d ≤ s ≤ H − d
```

### 4.2 [`Isaac-GR00T/gr00t/model/gr00t_n1.py`](../Isaac-GR00T/gr00t/model/gr00t_n1.py)

Added `GR00T_N1_5.guided_get_action(inputs)` mirroring `get_action`, routing through the backbone and dispatching to `action_head.guided_get_action(...)`.

### 4.3 [`Isaac-GR00T/gr00t/model/policy.py`](../Isaac-GR00T/gr00t/model/policy.py)

- `_prepare_rtc_freeze(freeze_raw)` — finds `action_concat_order` and per-limb `Normalizer` instances by scanning the `ComposedModalityTransform` chain, normalizes each limb's freeze tail, concatenates in the model's expected order, zero-pads to `action_dim=32`. Returns `(tensor, real_dim)`.
- `get_action(observations)` — pops `rtc.{freeze_prefix, freeze_len, stride, beta}` before transform; stashes (and restores in `finally`) on `self.model.action_head`. Computes `use_rtc` flag and passes through.
- `_get_action_from_normalized_input(normalized_input, use_rtc=False)` — dispatches:
  - `use_rtc=False` → `torch.inference_mode() + model.get_action`
  - `use_rtc=True`  → `torch.no_grad() + model.guided_get_action`

### 4.4 [`Isaac-GR00T/scripts/sim2real/gr00t_control_robot.py`](../Isaac-GR00T/scripts/sim2real/gr00t_control_robot.py)

- Removed: client-side `LowPassFilter` (was fighting RTC and adding phase lag), `--filter` flag, per-call `wait_for_server` (now polled once at startup).
- Added: rolling `RTC_LATENCY_BUF` (deque, maxlen=20) of recent `t_gen − t_obs`.
- Adaptive `d` and corrected `exec_idx`:
  ```python
  latency_est = max(self.RTC_LATENCY_BUF) if self.RTC_LATENCY_BUF else self.RTC_DEFAULT_LATENCY
  t_new_start = t_obs + latency_est                          # <- B1 fix: anchor at chunk start
  exec_idx    = round((t_new_start - self.prev_chunk_t0) / dt_step)
  d_lat       = ceil(latency_est / dt_step) + 1              # paper Alg. 1 line 18
  d           = min(d_lat, RTC_PREFIX_MAX, H - 1, H - exec_idx)
  ```
- `prev_chunk_raw` and `prev_chunk_t0` recorded after every `send_command`.
- `RTC_LATENCY_BUF`, `prev_chunk_raw`, `prev_chunk_t0` all cleared in `reset_robot`.
- RTC block intentionally **omitted** from the dataset playback (`use_dataset=True`) branch — that path is open-loop debug only.

### 4.5 Files NOT modified

- [`Isaac-GR00T/scripts/inference_service.py`](../Isaac-GR00T/scripts/inference_service.py) — RTC keys pass through the existing endpoint unchanged.
- [`Isaac-GR00T/gr00t/eval/service.py`](../Isaac-GR00T/gr00t/eval/service.py) — `torch.save`-based serializer handles arbitrary dict payloads.
- [`Isaac-GR00T/scripts/sim2real/utils/gr00t_client_adapter.py`](../Isaac-GR00T/scripts/sim2real/utils/gr00t_client_adapter.py) — N1.5 path is transparent. (N1.6's `_format_obs` would need a small forwarding tweak if/when used.)
- `openarm_ros2/scripts/action_chunk/action_chunk_controller.py` — alternative ROS-side execution model; not part of this pipeline.

---

## 5. Code-review fixes (B1–L3) — already applied

These came out of the post-implementation review and were folded into the code:

| ID | Severity | Fix |
|----|----------|-----|
| **B1** | HIGH | Anchor `exec_idx` at `t_obs + latency_est`, not `t_obs`. The new chunk starts executing only after inference completes. |
| **B2** | HIGH | Math verified: `g = ∂/∂A[(err.detach()·A_hat_1).sum()] ≡ err · ∂A_hat_1/∂A_τ`; sign is correct. Comment block added linking the autograd form to paper Eq. 2. |
| **B3** | HIGH | Enforce `0 < d ≤ s ≤ H − d` and `d ≤ H − 1`. |
| **W1** | MED | Removed the post-loop `actions[:, :d] = rtc_freeze[:, :d]` overwrite. The guided loop's converged values preserve continuity into the soft region. |
| **W2** | MED | Padded action dims (real_dim..D) excluded from guidance via `dim_mask`, so guidance capacity isn't spent pulling untrained channels to 0. |
| **W3** | MED | Added `@torch.no_grad()` on `guided_get_action` for self-contained safety; inner `with torch.enable_grad():` overrides for the VJP. |
| **L1** | LOW | τ singularity threshold raised `1e−6 → 1e−4` for fp16/autocast safety. |
| **L2** | LOW | `_compute_rtc_soft_mask` vectorised with `torch.arange + torch.exp`. |
| **L3** | LOW | Adaptive `d` from rolling latency buffer (combined with B1). |

---

## 6. Running, validating, tuning

### 6.1 Run

```bash
# Server (separate terminal)
python Isaac-GR00T/scripts/inference_service.py --server \
  --model_path /data/GR00T_finetune_checkpoints/openarm_cansorting_N15_fft_200k/checkpoint-200000/ \
  --data_config openarm_linkerhand_o6 \
  --denoising_steps 4

# Client
python Isaac-GR00T/scripts/sim2real/gr00t_control_robot.py \
  --robot_type openarm_linkerhand_o6 --gr00t_ver N1.5 \
  --control_mode async_latest    # required for RTC's overlap-with-execution model
```

### 6.2 Debug logging (toggle via env var)

Three guarded `print` points are committed and gated on `GR00T_RTC_DEBUG`. Run with:

```bash
GR00T_RTC_DEBUG=1 python Isaac-GR00T/scripts/inference_service.py --server …
GR00T_RTC_DEBUG=1 python Isaac-GR00T/scripts/sim2real/gr00t_control_robot.py …
```

| Process | Log line | Source | What to look for |
|---|---|---|---|
| Server | `[RTC-server] d=… s=… β=… real_dim=… freeze.shape=…` | `Gr00tPolicy.get_action` | Stash arrived correctly; `freeze.shape == (1, d, action_dim)` |
| Server | `[RTC-vjp] prefix residual (normalized): …` | `guided_get_action` end | < 0.05 means ΠGDM converged; raise β if larger |
| Client | `[RTC-client] lat_est=…ms exec_idx=… d=… buf_n=…` | `RobotController.inference_loop` | `lat_est` stabilises after ~5 chunks; `d` = `ceil(lat/dt)+1` clamped |

Cross-check that server `d` matches client `d` for the same chunk (no protocol drift).

### 6.3 Validation milestones

1. **No-RTC smoke**: first inference after reset takes the vanilla path; latency identical to pre-patch.
2. **Cold start RTC**: latency buffer empty → `latency_est = 0.15 s`, `d ≈ 5`. Verify `[RTC-server]` and `[RTC-client]` lines align.
3. **Steady state (after ~5 chunks)**: `lat_est ≈ 110–170 ms`, `d` stabilises.
4. **Prefix residual**: `< 0.05` in normalized action space. If higher, bump `rtc.beta` from 10 → 15.
5. **Visible motion**: chunk-boundary backtracking gone; arm motion looks like a single continuous trajectory rather than a sequence of independent samples.

### 6.4 Tuning knobs

| Knob | Default | When to change |
|---|---|---|
| `denoising_steps` (server) | 4 | raise to 8 if sampling artifacts visible; the RTC compute overhead grows linearly with this |
| `rtc.beta` (client → obs) | 10.0 | raise to 15–20 if prefix residual > 0.05; lower to 5 if motion feels sluggish |
| `RTC_PREFIX_MAX` (client) | 8 | lower to 6 if `d` keeps hitting the cap; raise if inference is very slow (>200 ms) |
| `RTC_DEFAULT_LATENCY` (client) | 0.15 s | match steady-state latency to skip cold-start bias |

### 6.5 Cost estimate

| Path | Forward | Backward | Walltime (RTX 4090, denoising_steps=4) |
|---|---|---|---|
| `get_action` (vanilla) | 4 × DiT | — | ~110 ms |
| `guided_get_action` (RTC) | 4 × DiT | 4 × DiT (VJP) | ~210 ms |

At 30 Hz control (~33 ms / step), RTC fits comfortably with `async_latest` since execution proceeds on the previous chunk while the new one is being inferred.

---

## 7. References

- Black, K. et al., *Real-Time Execution of Action Chunking Flow Policies*. arXiv:2506.07339v2, 2025.
- Song, J. et al., *Pseudoinverse-Guided Diffusion Models* (ΠGDM). 2023.
- Zhao, T. et al., *Action Chunking with Transformers* (ACT, temporal ensembling baseline). 2023.
- LeRobot RTC reference: `huggingface.co/docs/lerobot/rtc`.
- NVIDIA GR00T-N1.5: `nvidia/GR00T-N1.5-3B`.

---

## 8. Glossary

| Symbol | Meaning |
|---|---|
| `H` | action horizon (16 steps) |
| `d` | hard-frozen prefix length (≈ inference latency / dt) |
| `s` | execution horizon (free-tail length); `d ≤ s ≤ H − d` |
| `τ` | denoising time (GR00T: 0 = noise, 1 = clean) |
| `β` | ΠGDM guidance cap (default 10.0) |
| `A_prev` | previous chunk's tail, time-aligned to the new chunk |
| `A_hat_1` | clean-action estimate at current denoising step |
| `W` | per-step soft mask `[1…1, decay…, 0…0]` |
| `dim_mask` | per-feature mask zeroing padded action dims |
