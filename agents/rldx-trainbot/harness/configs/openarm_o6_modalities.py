# Shared modality-config builder for OpenArm-O6 (bimanual arms + LinkerHand O6) on RLDX-1.
#
# Deliberately mirrors GR00T N1.7's own
# examples/Openarm_LinkerHandO6/openarm_o6_modalities.py, because the comparison is only
# meaningful if both models read the dataset the same way. Keep the two in sync.
#
# Dataset layout (meta/modality.json of OpenArm_O6_CanSorting_MultiTask_Sim_Dataset_0403):
#
#     state/action (26 dims):
#         left_arm   [ 0: 7]   7 joints
#         right_arm  [ 7:14]   7 joints
#         left_hand  [14:20]   6 finger joints
#         right_hand [20:26]   6 finger joints
#     video:    "camera"  -> observation.images.camera (RGB 480x640, single head view)
#     language: annotation.human.task_description
#
# A config's state/action keys may be a SUBSET of the dataset's: a right-only config trains
# fine on the full bimanual dataset, only the listed limbs are read. All nine existing
# GR00T N1.7 baselines were trained with MODE=right_only, so right-only is the setting the
# team actually deploys; the bimanual variant exists to test whether the extra limbs matter.
#
# ACTION REPRESENTATION -- arms RELATIVE, hands ABSOLUTE, all NON_EEF / DEFAULT.
# Not a free choice. GR00T's preset documents arms as RELATIVE ("N1.7 generalizes better
# with relative") and hands as ABSOLUTE ("open/closed finger targets, like a gripper"), and
# RLDX-1's own unitree_g1 preset independently uses RELATIVE + NON_EEF + DEFAULT for arms.
# An earlier config here omitted action_configs, which silently meant ABSOLUTE; the training
# run started under it was discarded.
#
# RELATIVE actions require meta/relative_stats.json from
# rldx.data.stats.generate_rel_stats(dataset_path, embodiment_tag).
#
# This module intentionally does NOT call register_modality_config() -- importing it has no
# side effects, so the preset files can share it without double-registering the tag.

from collections.abc import Sequence

from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# Canonical limb order, matching the [start:end] layout in meta/modality.json so registered
# keys line up with the dataset slices regardless of which subset is active.
CANONICAL_LIMB_ORDER = ["left_arm", "right_arm", "left_hand", "right_hand"]

# RLDX-1 emits 16-step action chunks (MSAT action horizon). train_config.action_horizon must
# equal len(delta_indices) for every action modality, or assembly fails.
ACTION_HORIZON = 16

_ARM_ACTION = ActionConfig(
    rep=ActionRepresentation.RELATIVE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)
_HAND_ACTION = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)


def _action_config_for(limb: str) -> ActionConfig:
    # state_key stays None: the processor defaults it to the action's own key
    # (state_action_processor.py:373), so left_arm deltas are taken against left_arm state.
    if limb.endswith("_arm"):
        return _ARM_ACTION
    if limb.endswith("_hand"):
        return _HAND_ACTION
    raise ValueError(f"Cannot infer action representation for limb '{limb}'.")


def build_openarm_o6_config(
    active_limbs: Sequence[str],
    *,
    video_keys: Sequence[str] = ("camera",),
    action_horizon: int = ACTION_HORIZON,
) -> dict[str, ModalityConfig]:
    """Build a RLDX-1 modality config for the given subset of OpenArm-O6 limbs."""
    limbs = list(active_limbs)
    unknown = [limb for limb in limbs if limb not in CANONICAL_LIMB_ORDER]
    if unknown:
        raise ValueError(f"Unknown limb(s) {unknown}; expected a subset of {CANONICAL_LIMB_ORDER}")
    # Keep dataset order so state/action slices line up with meta/modality.json.
    limbs = [limb for limb in CANONICAL_LIMB_ORDER if limb in limbs]
    if not limbs:
        raise ValueError("active_limbs must not be empty")

    return {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=list(video_keys),
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=limbs,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(action_horizon)),
            modality_keys=limbs,
            action_configs=[_action_config_for(limb) for limb in limbs],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    }
