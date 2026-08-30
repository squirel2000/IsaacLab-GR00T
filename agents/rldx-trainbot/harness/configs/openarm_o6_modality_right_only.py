# RLDX-1 modality config: OpenArm-O6 RIGHT ARM + RIGHT HAND ONLY (13 dims).
#
# This is the setting to compare against GR00T N1.7: all nine existing N1.7 baselines were
# trained with MODE=right_only, so it is what the team actually deploys. It also matches the
# data — measured from meta/stats.json, left_hand action is exactly constant across all 2000
# episodes (every statistic 0.0000) and left_arm moves at roughly a third of the right arm's
# amplitude, so the left side carries little to no signal for can-sorting.
#
# Registered keys are a subset of the dataset's 26-dim bimanual layout; the loader reads only
# the listed limbs, so no dataset change is needed.
#
# Pass via:
#   --modality-config-path agents/rldx-trainbot/harness/configs/openarm_o6_modality_right_only.py \
#   --embodiment-tag GENERAL_EMBODIMENT
#
# See openarm_o6_modalities.py for the shared builder and the action-representation rationale
# (arms RELATIVE, hands ABSOLUTE). Bimanual counterpart: openarm_o6_modality.py.

from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag

from openarm_o6_modalities import build_openarm_o6_config

OPENARM_O6_RIGHT_ONLY_CONFIG = build_openarm_o6_config(["right_arm", "right_hand"])

register_modality_config(OPENARM_O6_RIGHT_ONLY_CONFIG, EmbodimentTag.GENERAL_EMBODIMENT)
