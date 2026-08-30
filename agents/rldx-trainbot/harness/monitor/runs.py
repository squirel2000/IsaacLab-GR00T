#!/usr/bin/env python3
"""Run registry for the training monitor — model-agnostic by design.

Adding a run is a data edit, not a code change, and nothing here is RLDX-specific: a GR00T
N1.7 run, a starVLA run or any future backend is described by the same fields. The renderer
reads only these fields, so one template serves every model.

Both RLDX-1 and GR00T N1.7 train through HuggingFace Trainer, so a single metric extractor
(vla-trainbot's `training_monitor.extract_metrics`, reused via training_bridge) parses both
without special-casing.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class RunSpec:
    """One training run, described in backend-neutral terms."""

    id: str
    label: str
    family: str                     # "RLDX-1" | "GR00T N1.7" | ... — groups the tabs
    model: str                      # base checkpoint id or path
    log: str                        # training log path on the training host
    output: str                     # checkpoint output dir on the training host
    max_steps: int
    gpus: int
    effective_batch: int
    action_space: str               # human-readable, e.g. "13 (right_arm, right_hand)"
    tuning: str                     # e.g. "LoRA r16/a32 both surfaces" | "projector + diffusion head"
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# Effective batch is held at 64 across every run so the comparison is on equal footing;
# single-GPU runs reach it with --global-batch-size 16 --gradient-accumulation-steps 4 (a single
# card has no ZeRO sharding, so a smaller per-device batch keeps VRAM inside 80 GB).
RUNS: list[RunSpec] = [
    RunSpec(
        id="rldx1_bimanual_30k",
        label="RLDX-1 LoRA · bimanual",
        family="RLDX-1",
        model="RLWRLD/RLDX-1-PT-IMG",
        log="/data/VLA/tingying/pegasus_runs/rldx_train/train.log",
        output="/data/VLA/tingying/artifacts/rldx1/lora",
        max_steps=30000,
        gpus=2,
        effective_batch=64,
        action_space="26 (left_arm, right_arm, left_hand, right_hand)",
        tuning="LoRA r16/a32, action-model + backbone",
        notes="Tests whether including the left side matters. Loss is not comparable with the "
              "13-dim runs: 6 of its 26 dims are constant, which dilutes the average.",
        tags=["bimanual", "lora"],
    ),
    RunSpec(
        id="rldx1_right_only_30k",
        label="RLDX-1 LoRA · right only",
        family="RLDX-1",
        model="RLWRLD/RLDX-1-PT-IMG",
        log="/data/VLA/tingying/pegasus_runs/chain2/rldx_rightonly.log",
        output="/data/VLA/tingying/artifacts/rldx1/lora_right_only",
        max_steps=30000,
        gpus=1,
        effective_batch=64,
        action_space="13 (right_arm, right_hand)",
        tuning="LoRA r16/a32, action-model + backbone",
        notes="The matched arm of the comparison — same action space as every existing N1.7 baseline. "
              "Re-launched on GPU1 after the first attempt OOM'd from GPU contention; chain2/ is "
              "the live log, chain/ (the first attempt) is stale.",
        tags=["right_only", "lora", "matched"],
    ),
    RunSpec(
        id="n17_right_only_30k",
        label="GR00T N1.7 · right only",
        family="GR00T N1.7",
        model="nvidia/GR00T-N1.7-3B",
        log="/data/VLA/tingying/pegasus_runs/chain2/n17_rightonly.log",
        output="/data/VLA/tingying/artifacts/gr00t_n17_match/N1_7_rightonly_30k_batch64_lr1e4",
        max_steps=30000,
        gpus=1,
        effective_batch=64,
        action_space="13 (right_arm, right_hand)",
        tuning="projector + full diffusion head (N1.7 has no LoRA)",
        notes="Trains far more parameters than the RLDX-1 LoRA runs, so RLDX-1 is the handicapped "
              "side of this comparison, not the favoured one.",
        tags=["right_only", "baseline", "matched"],
    ),
]


def by_id(run_id: str) -> RunSpec | None:
    return next((r for r in RUNS if r.id == run_id), None)


def families() -> list[str]:
    seen: list[str] = []
    for r in RUNS:
        if r.family not in seen:
            seen.append(r.family)
    return seen
