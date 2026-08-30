# RLDX-1 modality configs for OpenArm + LinkerHand O6

| File | Role |
| --- | --- |
| `openarm_o6_modalities.py` | Shared builder both presets below import. Deliberately mirrors GR00T's own builder so RLDX-1 and GR00T read the dataset identically — that equivalence is what makes the two models comparable at all. |
| `openarm_o6_modality.py` | **Bimanual**, 26-dim action (`left_arm:7, right_arm:7, left_hand:6, right_hand:6`). |
| `openarm_o6_modality_right_only.py` | **Right-arm-only**, 13-dim. This is the matched arm of the comparison — all nine existing N1.7 baselines were trained `MODE=right_only`, and `left_hand` action is constant across all 2000 episodes anyway. |

Pass one with `--modality-config-path` at **training** time. That is the only moment it takes
effect: `run_rldx_server.py` reads the flag only on the `ReplayPolicy` branch, and the real
`RLDXPolicy` path hard-indexes `processor.get_modality_configs()[embodiment_tag.value]` with no
fallback — so which embodiments a checkpoint can serve is baked in when it is trained, not
chosen when it is served.

## Two gotchas that will cost you an afternoon

**The registry is global and once-only.** `register_modality_config` asserts on double
registration, and `load_modality_config()` imports the target by *file stem* after appending its
directory to `sys.path`. Two consequences: the shared builder must sit **beside** the presets (it
does), and **you cannot load two presets in one process** — comparing them requires one
subprocess each.

**`GENERAL_EMBODIMENT` has no default entry**, despite upstream describing it as "the slot
`RLDX-1-PT` reserves for downstream fine-tuning". Any RLDX-1 training *or* serving command for
this robot must pass a modality config explicitly; there is nothing to fall back on. (Confirmed
by comparing the two published checkpoints' processors: `PT-IMG` carries 34 embodiments with
`general_embodiment` **absent**; `FT-LIBERO` carries the same 34 **plus** `general_embodiment`
populated with LIBERO's own config.)

## Arm action representation

`openarm_o6_modalities.py` contains exactly **one** `rep=ActionRepresentation.RELATIVE`
occurrence — the arm; the hand is already `ABSOLUTE`. So the RELATIVE→ABSOLUTE A/B test is a
single-line change: back the file up (`cp -n <file> <file>.relative.bak`), `sed` that one
occurrence, and verify by counting remaining `RELATIVE` lines (expect zero). Restore from the
`.bak` afterwards — don't hand-edit it back.
