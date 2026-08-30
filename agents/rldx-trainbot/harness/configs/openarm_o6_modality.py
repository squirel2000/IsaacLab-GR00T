# RLDX-1 modality config: OpenArm-O6 BIMANUAL — both arms + both hands (26 dims).
#
# The dataset is bimanual, so this reads all of it. Note however that every existing GR00T
# N1.7 baseline was trained with MODE=right_only (13 dims), so for a matched head-to-head use
# openarm_o6_modality_right_only.py instead; this variant exists to test whether including the
# left side changes anything. Measured from meta/stats.json it probably does not: left_hand
# action is exactly constant across all 2000 episodes and left_arm moves at about a third of
# the right arm's amplitude.
#
# Pass via:
#   --modality-config-path agents/rldx-trainbot/harness/configs/openarm_o6_modality.py \
#   --embodiment-tag GENERAL_EMBODIMENT
#
# RLDX-1 ships MODALITY_CONFIGS entries only for unitree_g1, libero_panda, oxe_widowx,
# oxe_google and behavior_r1_pro — there is NO general_embodiment entry, even though
# GENERAL_EMBODIMENT is the tag upstream reserves for downstream finetuning. So one has to be
# registered, and rldx/experiment/utils.py:load_modality_config() does that simply by
# importing this file, hence the module-level register call below.
#
# See openarm_o6_modalities.py for the shared builder, the dataset layout, and the
# action-representation rationale (arms RELATIVE, hands ABSOLUTE).

from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag

from openarm_o6_modalities import CANONICAL_LIMB_ORDER, build_openarm_o6_config

OPENARM_O6_CONFIG = build_openarm_o6_config(CANONICAL_LIMB_ORDER)

register_modality_config(OPENARM_O6_CONFIG, EmbodimentTag.GENERAL_EMBODIMENT)
