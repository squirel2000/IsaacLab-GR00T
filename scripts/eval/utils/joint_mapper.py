import numpy as np

# To avoid the import problems described below.
try:
    import carb
    carb_settings_iface = carb.settings.get_settings()
    robot_type = carb_settings_iface.get("/current_env/robot_type")
except:
    # only be triggered when executing 'generate_dataset_metadata.py', because it did not launch IsaacSim.
    robot_type=""
    pass

"""
This method cannot be used; 
When executing 'generate_dataset_metadata.py' it will import joint_mapper to get joint list, 
an initialization import failure will occur because IsaacSim is not enabled.
Therefore, not get the joint list by import from cfg for the time being, define it manually again.
"""
# from isaaclab_tasks.manager_based.manipulation.playground_openarm.dexhand_bimanual.config.openarm_robot_cfg import (  # isort: skip
#     OPENARM_JOINT,
#     LEAPHAND_RIGHT_JOINT,
#     LINKERHAND_O6_JOINT,
# )

OPENARM_JOINT = [
    "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
    "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6", "openarm_left_joint7",
    "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
    "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6", "openarm_right_joint7",
]
LEAPHAND_RIGHT_JOINT = [
    'index_mcp_forward', 'middle_mcp_forward', 'ring_mcp_forward', 'thumb_mcp_side', 
    'index_mcp_side', 'middle_mcp_side', 'ring_mcp_side', 'thumb_mcp_forward', 
    'index_pip', 'middle_pip', 'ring_pip', 'thumb_pip_joint', 
    'index_dip', 'middle_dip', 'ring_dip', 'thumb_dip_joint',
]
LINKERHAND_O6_JOINT = [
    'L_index_mcp_pitch', 'L_middle_mcp_pitch', 'L_pinky_mcp_pitch', 'L_ring_mcp_pitch',
    'L_thumb_cmc_pitch', 'L_thumb_cmc_yaw',
    'R_index_mcp_pitch', 'R_middle_mcp_pitch', 'R_pinky_mcp_pitch', 'R_ring_mcp_pitch',
    'R_thumb_cmc_pitch', 'R_thumb_cmc_yaw',
]

class JointMapper:
    """
    Handles the mapping of joint observations and actions between IsaacSim
    and the GR00T model.
    """
    # GR00T model joint names in their canonical order (based on info.json from GR00T dataset)
    TRIHAND_GR00T_MODEL_JOINT_NAMES = [
        "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
        "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
        "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
        "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
        "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
        "kLeftHandMiddle0", "kLeftHandMiddle1",
        "kLeftHandIndex0", "kLeftHandIndex1",
        "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
        "kRightHandIndex0", "kRightHandIndex1",
        "kRightHandMiddle0", "kRightHandMiddle1"
    ]

    # Corresponding IsaacSim joint names. The order MUST match GR00T_MODEL_JOINT_NAMES.
    TRIHAND_ISAACSIM_EQUIVALENT_NAMES_FOR_GR00T_JOINTS = [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint", "left_hand_middle_1_joint",
        "left_hand_index_0_joint", "left_hand_index_1_joint",
        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        "right_hand_index_0_joint", "right_hand_index_1_joint",
        "right_hand_middle_0_joint", "right_hand_middle_1_joint"
    ]
    
    INSPIRE_GR00T_MODEL_JOINT_NAMES = [
        "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
        "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
        "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
        "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
        "L_index_proximal_joint", "L_middle_proximal_joint", "L_pinky_proximal_joint", "L_ring_proximal_joint",
        "L_thumb_proximal_yaw_joint",
        "L_index_intermediate_joint", "L_middle_intermediate_joint", "L_pinky_intermediate_joint", "L_ring_intermediate_joint",
        "L_thumb_proximal_pitch_joint", "L_thumb_intermediate_joint", "L_thumb_distal_joint",
        "R_index_proximal_joint", "R_middle_proximal_joint", "R_pinky_proximal_joint", "R_ring_proximal_joint",
        "R_thumb_proximal_yaw_joint",
        "R_index_intermediate_joint", "R_middle_intermediate_joint", "R_pinky_intermediate_joint", "R_ring_intermediate_joint",
        "R_thumb_proximal_pitch_joint", "R_thumb_intermediate_joint", "R_thumb_distal_joint"
    ]

    # Corresponding IsaacSim joint names. The order MUST match GR00T_MODEL_JOINT_NAMES.
    INSPIRE_ISAACSIM_EQUIVALENT_NAMES_FOR_GR00T_JOINTS = [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        "L_index_proximal_joint", "L_middle_proximal_joint", "L_pinky_proximal_joint", "L_ring_proximal_joint",
        "L_thumb_proximal_yaw_joint",
        "L_index_intermediate_joint", "L_middle_intermediate_joint", "L_pinky_intermediate_joint", "L_ring_intermediate_joint",
        "L_thumb_proximal_pitch_joint", "L_thumb_intermediate_joint", "L_thumb_distal_joint",
        "R_index_proximal_joint", "R_middle_proximal_joint", "R_pinky_proximal_joint", "R_ring_proximal_joint",
        "R_thumb_proximal_yaw_joint",
        "R_index_intermediate_joint", "R_middle_intermediate_joint", "R_pinky_intermediate_joint", "R_ring_intermediate_joint",
        "R_thumb_proximal_pitch_joint", "R_thumb_intermediate_joint", "R_thumb_distal_joint"
    ]

    INSPIRE_GR00T_MODEL_ACTION_JOINT_NAMES = [
        "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
        "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
        "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
        "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
        "L_index_proximal_joint", "L_middle_proximal_joint", "L_pinky_proximal_joint", "L_ring_proximal_joint",
        "L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint",
        "R_index_proximal_joint", "R_middle_proximal_joint", "R_pinky_proximal_joint", "R_ring_proximal_joint",
        "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint"
    ]

    OPENARM_LEAPHAND_RIGHT_JOINT_NAMES = OPENARM_JOINT + LEAPHAND_RIGHT_JOINT

    OPENARM_LINKERHAND_O6_JOINT_NAMES = OPENARM_JOINT + LINKERHAND_O6_JOINT

    # Structure for GR00T observation state keys and action keys, using GR00T joint names.
    if robot_type == "g1_trihand":
        GR00T_MODEL_JOINT_NAMES = TRIHAND_GR00T_MODEL_JOINT_NAMES
        ISAACSIM_EQUIVALENT_NAMES_FOR_GR00T_JOINTS = TRIHAND_ISAACSIM_EQUIVALENT_NAMES_FOR_GR00T_JOINTS
        GR00T_LIMB_STATE_NAMES_STRUCTURE = {
            "left_arm": GR00T_MODEL_JOINT_NAMES[0:7],
            "right_arm": GR00T_MODEL_JOINT_NAMES[7:14],
            "left_hand": GR00T_MODEL_JOINT_NAMES[14:21],
            "right_hand": GR00T_MODEL_JOINT_NAMES[21:28],
        }
        GR00T_LIMB_ACTION_NAMES_STRUCTURE = GR00T_LIMB_STATE_NAMES_STRUCTURE

    elif robot_type == "g1_inspire":
        GR00T_MODEL_JOINT_NAMES = INSPIRE_GR00T_MODEL_JOINT_NAMES
        ISAACSIM_EQUIVALENT_NAMES_FOR_GR00T_JOINTS = INSPIRE_ISAACSIM_EQUIVALENT_NAMES_FOR_GR00T_JOINTS
        GR00T_LIMB_STATE_NAMES_STRUCTURE = {
            "left_arm": GR00T_MODEL_JOINT_NAMES[0:7],
            "right_arm": GR00T_MODEL_JOINT_NAMES[7:14],
            "left_hand": GR00T_MODEL_JOINT_NAMES[14:26],
            "right_hand": GR00T_MODEL_JOINT_NAMES[26:38],
        }
        GR00T_LIMB_ACTION_NAMES_STRUCTURE = {
            "left_arm": INSPIRE_GR00T_MODEL_ACTION_JOINT_NAMES[0:7],
            "right_arm": INSPIRE_GR00T_MODEL_ACTION_JOINT_NAMES[7:14],
            "left_hand": INSPIRE_GR00T_MODEL_ACTION_JOINT_NAMES[14:20],
            "right_hand": INSPIRE_GR00T_MODEL_ACTION_JOINT_NAMES[20:26],
        }

    elif robot_type == "openarm_leaphand_right":
        GR00T_MODEL_JOINT_NAMES = OPENARM_LEAPHAND_RIGHT_JOINT_NAMES
        ISAACSIM_EQUIVALENT_NAMES_FOR_GR00T_JOINTS = OPENARM_LEAPHAND_RIGHT_JOINT_NAMES
        GR00T_LIMB_STATE_NAMES_STRUCTURE = {
            "left_arm": GR00T_MODEL_JOINT_NAMES[0:7],
            "right_arm": GR00T_MODEL_JOINT_NAMES[7:14],
            "right_hand": GR00T_MODEL_JOINT_NAMES[14:30],
        }
        GR00T_LIMB_ACTION_NAMES_STRUCTURE = GR00T_LIMB_STATE_NAMES_STRUCTURE

    #elif robot_type == "openarm_linkerhand_o6":
    else:
        GR00T_MODEL_JOINT_NAMES = OPENARM_LINKERHAND_O6_JOINT_NAMES
        ISAACSIM_EQUIVALENT_NAMES_FOR_GR00T_JOINTS = OPENARM_LINKERHAND_O6_JOINT_NAMES
        GR00T_LIMB_STATE_NAMES_STRUCTURE = {
            "left_arm": GR00T_MODEL_JOINT_NAMES[0:7],
            "right_arm": GR00T_MODEL_JOINT_NAMES[7:14],
            "left_hand": GR00T_MODEL_JOINT_NAMES[14:20],
            "right_hand": GR00T_MODEL_JOINT_NAMES[20:26],
        }
        GR00T_LIMB_ACTION_NAMES_STRUCTURE = GR00T_LIMB_STATE_NAMES_STRUCTURE

    GR00T_TO_ISAACSIM_JOINT_NAME_MAP = dict(zip(GR00T_MODEL_JOINT_NAMES, ISAACSIM_EQUIVALENT_NAMES_FOR_GR00T_JOINTS))
    ISAACSIM_TO_GR00T_JOINT_NAME_MAP = dict(zip(ISAACSIM_EQUIVALENT_NAMES_FOR_GR00T_JOINTS, GR00T_MODEL_JOINT_NAMES))


    def __init__(self, env_cfg, robot_articulation):
        """
        Initializes the JointMapper.

        Args:
            env_cfg: The environment configuration object from IsaacLab.
            robot_articulation: The robot articulation object from IsaacSim scene.
        """
        self.env_cfg = env_cfg
        self.robot_articulation = robot_articulation

        self.isaacsim_env_action_joint_names = self._get_isaacsim_action_joint_names()
        self.num_isaacsim_action_joints = len(self.isaacsim_env_action_joint_names)

        # self.isaacsim_full_obs_joint_names = self._get_isaacsim_full_obs_joint_names()
        self.isaacsim_full_obs_joint_names = self._get_isaacsim_action_joint_names()
        
        self._validate_joint_names()
        # self.print_joint_name_info()

    def _get_isaacsim_action_joint_names(self):
        """Retrieves the order of joint names expected by the IsaacSim environment for actions."""
        try:
            if hasattr(self.env_cfg.actions, 'arm_action_cfg') and hasattr(self.env_cfg.actions.arm_action_cfg, 'joint_names'):
                 action_joint_names = self.env_cfg.actions.arm_action_cfg.joint_names
                 if action_joint_names is None:
                     print("Warning: env_cfg.actions.arm_action_cfg.joint_names is None. Falling back to robot_articulation.joint_names for action mapping.")
                     action_joint_names = list(self.robot_articulation.joint_names)
                 return list(action_joint_names) # Ensure it's a list
            else:
                print("Warning: Could not find env_cfg.actions.arm_action_cfg.joint_names. Using robot_articulation.joint_names for action mapping.")
                return list(self.robot_articulation.joint_names)
        except AttributeError as e:
            print(f"Error accessing action joint names from env_cfg: {e}. Using robot_articulation.joint_names as fallback.")
            return list(self.robot_articulation.joint_names)

    def _get_isaacsim_full_obs_joint_names(self):
        """Retrieves the full list of joint names from the IsaacSim robot articulation."""
        try:
            return list(self.robot_articulation.joint_names) # Ensure it's a list
        except AttributeError:
            print("Error: Could not get joint names from robot_articulation.joint_names.")
            return []

    def _validate_joint_names(self):
        """Validates that all GR00T-equivalent IsaacSim joints are present in the IsaacSim definitions."""
        print("\n--- Validating Joint Names ---")
        # Validate observation joints
        if not self.isaacsim_full_obs_joint_names:
            print("CRITICAL WARNING: IsaacSim full observation joint names list is EMPTY. Observation mapping will likely fail.")
        else:
            missing_obs_joints = [name for name in self.ISAACSIM_EQUIVALENT_NAMES_FOR_GR00T_JOINTS if name not in self.isaacsim_full_obs_joint_names]
            if missing_obs_joints:
                print(f"ERROR: The following IsaacSim joints (mapped from GR00T) are MISSING from the robot's full observation joint list ({len(self.isaacsim_full_obs_joint_names)} total): {missing_obs_joints}")

        # Validate action joints
        if not self.isaacsim_env_action_joint_names:
            print("CRITICAL WARNING: IsaacSim environment action joint names list is EMPTY. Action mapping will likely fail.")
        else:
            missing_action_joints = [name for name in self.ISAACSIM_EQUIVALENT_NAMES_FOR_GR00T_JOINTS if name not in self.isaacsim_env_action_joint_names]
            if missing_action_joints:
                print(f"WARNING: The following IsaacSim joints (mapped from GR00T) are MISSING from the environment's action joint list ({len(self.isaacsim_env_action_joint_names)} total): {missing_action_joints}. Actions for these GR00T joints will be ignored.")
        
        unmapped_env_action_joints = [name for name in self.isaacsim_env_action_joint_names if name not in self.ISAACSIM_EQUIVALENT_NAMES_FOR_GR00T_JOINTS]
        if unmapped_env_action_joints:
            print(f"INFO: The following IsaacSim action joints are defined in the environment but are NOT TARGETED by GR00T actions (will receive 0.0 or default action): {unmapped_env_action_joints}")
        print("--- Validation Complete ---")

    def print_joint_name_info(self):
        """Prints detailed information about the joint names for debugging and verification."""
        print("\n--- Joint Mapper: Joint Name Information ---")
        print(f"Number of GR00T model joints defined: {len(self.GR00T_MODEL_JOINT_NAMES)}")

        print(f"\nIsaacSim Full Observation Joint Names (Total: {len(self.isaacsim_full_obs_joint_names)}):")
        for i, name in enumerate(self.isaacsim_full_obs_joint_names): print(f"  [{i:02d}] {name}")

        print(f"\nIsaacSim Environment Action Joint Names (Total: {len(self.isaacsim_env_action_joint_names)}):")
        for i, name in enumerate(self.isaacsim_env_action_joint_names): print(f"  [{i:02d}] {name}")
        
        print("\nMapping: GR00T Name -> IsaacSim Name (IsaacSim Obs Idx | IsaacSim Act Idx)")
        for gr00t_name in self.GR00T_MODEL_JOINT_NAMES:
            isaac_name = self.GR00T_TO_ISAACSIM_JOINT_NAME_MAP.get(gr00t_name, "NOT_MAPPED")
            obs_idx_str = f"{self.isaacsim_full_obs_joint_names.index(isaac_name):02d}" if isaac_name in self.isaacsim_full_obs_joint_names else "N/A"
            act_idx_str = f"{self.isaacsim_env_action_joint_names.index(isaac_name):02d}" if isaac_name in self.isaacsim_env_action_joint_names else "N/A"
            print(f"  {gr00t_name:<20} -> {isaac_name:<30} (Obs: {obs_idx_str} | Act: {act_idx_str})")
        print("--- End of Joint Name Information ---\n")

    def map_isaac_obs_to_gr00t_state(self, isaac_robot_joint_pos_flat: np.ndarray) -> dict:
        """
        Maps flat IsaacSim robot joint positions to a GR00T-structured state dictionary.
        Args:
            isaac_robot_joint_pos_flat: A 1D numpy array of current joint positions
                                        for all robot joints from IsaacSim observation.
        Returns:
            A dictionary for GR00T state observations (e.g., {"state.left_arm": ...}).
        """
        if not self.isaacsim_full_obs_joint_names:
            print("ERROR CRITICAL: isaacsim_full_obs_joint_names is empty. Cannot map observations. Returning empty state.")
            return {f"state.{limb_key}": np.array([[]], dtype=np.float64) for limb_key in self.GR00T_LIMB_STATE_NAMES_STRUCTURE}

        current_isaac_joint_states = dict(zip(self.isaacsim_full_obs_joint_names, isaac_robot_joint_pos_flat))
        gr00t_state_obs = {}

        for limb_key, gr00t_joint_name_list_for_limb in self.GR00T_LIMB_STATE_NAMES_STRUCTURE.items():
            limb_states = []
            for gr00t_joint_name in gr00t_joint_name_list_for_limb:
                isaac_joint_name = self.GR00T_TO_ISAACSIM_JOINT_NAME_MAP.get(gr00t_joint_name)
                if isaac_joint_name is None or isaac_joint_name not in current_isaac_joint_states:
                    print(f"Warning: Joint '{gr00t_joint_name}' (Isaac: '{isaac_joint_name}') not found or not mapped in current IsaacSim observation. Using 0.0 for this joint.")
                    limb_states.append(0.0)
                else:
                    limb_states.append(current_isaac_joint_states[isaac_joint_name])
            gr00t_state_obs[f"state.{limb_key}"] = np.array([limb_states], dtype=np.float64)
        return gr00t_state_obs

    def map_gr00t_action_to_isaac_action(self, gr00t_action_dict: dict) -> np.ndarray:
        """
        Maps a GR00T action dictionary to a flat numpy array for IsaacSim environment.
        Args:
            gr00t_action_dict: Action dictionary from GR00T server.
        Returns:
            A 2D numpy array of sixteen time steps for IsaacSim environment.
        """
        env_action_values_fully_step = np.zeros((16, self.num_isaacsim_action_joints), dtype=np.float32)
        if not self.isaacsim_env_action_joint_names:
            print("ERROR CRITICAL: isaacsim_env_action_joint_names is empty. Cannot map actions. Returning zero actions.")
            return env_action_values_fully_step

        for limb_key_base, gr00t_limb_joint_names_list in self.GR00T_LIMB_ACTION_NAMES_STRUCTURE.items():
            gr00t_action_key = f"action.{limb_key_base}"
            if gr00t_action_key not in gr00t_action_dict:
                # print(f"Warning: Action key '{gr00t_action_key}' not in GR00T action dict. Skipping.")
                continue

            # Take predicted action from the 16 (default) sequences
            action_values_for_limb_step = gr00t_action_dict[gr00t_action_key]

            for i, gr00t_joint_name in enumerate(gr00t_limb_joint_names_list):
                isaac_joint_name = self.GR00T_TO_ISAACSIM_JOINT_NAME_MAP.get(gr00t_joint_name)
                if isaac_joint_name and isaac_joint_name in self.isaacsim_env_action_joint_names:
                    target_idx = self.isaacsim_env_action_joint_names.index(isaac_joint_name)
                    env_action_values_fully_step[:, target_idx] = action_values_for_limb_step[:, i]
                # Warnings for unmapped/missing joints are covered by _validate_joint_names and print_joint_name_info
        return env_action_values_fully_step

      

# --- Validating Joint Names ---
# --- Validation Complete ---

# --- Joint Mapper: Joint Name Information ---
# Number of GR00T model joints defined: 28

# IsaacSim Full Observation Joint Names (Total: 43):
#   [00] left_hip_pitch_joint
#   [01] right_hip_pitch_joint
#   [02] waist_yaw_joint
#   [03] left_hip_roll_joint
#   [04] right_hip_roll_joint
#   [05] waist_roll_joint
#   [06] left_hip_yaw_joint
#   [07] right_hip_yaw_joint
#   [08] waist_pitch_joint
#   [09] left_knee_joint
#   [10] right_knee_joint
#   [11] left_shoulder_pitch_joint
#   [12] right_shoulder_pitch_joint
#   [13] left_ankle_pitch_joint
#   [14] right_ankle_pitch_joint
#   [15] left_shoulder_roll_joint
#   [16] right_shoulder_roll_joint
#   [17] left_ankle_roll_joint
#   [18] right_ankle_roll_joint
#   [19] left_shoulder_yaw_joint
#   [20] right_shoulder_yaw_joint
#   [21] left_elbow_joint
#   [22] right_elbow_joint
#   [23] left_wrist_roll_joint
#   [24] right_wrist_roll_joint
#   [25] left_wrist_pitch_joint
#   [26] right_wrist_pitch_joint
#   [27] left_wrist_yaw_joint
#   [28] right_wrist_yaw_joint
#   [29] left_hand_index_0_joint
#   [30] left_hand_middle_0_joint
#   [31] left_hand_thumb_0_joint
#   [32] right_hand_index_0_joint
#   [33] right_hand_middle_0_joint
#   [34] right_hand_thumb_0_joint
#   [35] left_hand_index_1_joint
#   [36] left_hand_middle_1_joint
#   [37] left_hand_thumb_1_joint
#   [38] right_hand_index_1_joint
#   [39] right_hand_middle_1_joint
#   [40] right_hand_thumb_1_joint
#   [41] left_hand_thumb_2_joint
#   [42] right_hand_thumb_2_joint

# IsaacSim Environment Action Joint Names (Total: 28):
#   [00] left_shoulder_pitch_joint
#   [01] right_shoulder_pitch_joint
#   [02] left_shoulder_roll_joint
#   [03] right_shoulder_roll_joint
#   [04] left_shoulder_yaw_joint
#   [05] right_shoulder_yaw_joint
#   [06] left_elbow_joint
#   [07] right_elbow_joint
#   [08] left_wrist_roll_joint
#   [09] right_wrist_roll_joint
#   [10] left_wrist_pitch_joint
#   [11] right_wrist_pitch_joint
#   [12] left_wrist_yaw_joint
#   [13] right_wrist_yaw_joint
#   [14] left_hand_index_0_joint
#   [15] left_hand_middle_0_joint
#   [16] left_hand_thumb_0_joint
#   [17] right_hand_index_0_joint
#   [18] right_hand_middle_0_joint
#   [19] right_hand_thumb_0_joint
#   [20] left_hand_index_1_joint
#   [21] left_hand_middle_1_joint
#   [22] left_hand_thumb_1_joint
#   [23] right_hand_index_1_joint
#   [24] right_hand_middle_1_joint
#   [25] right_hand_thumb_1_joint
#   [26] left_hand_thumb_2_joint
#   [27] right_hand_thumb_2_joint

# Mapping: GR00T Name -> IsaacSim Name (IsaacSim Obs Idx | IsaacSim Act Idx)
#   kLeftShoulderPitch   -> left_shoulder_pitch_joint      (Obs: 11 | Act: 00)
#   kLeftShoulderRoll    -> left_shoulder_roll_joint       (Obs: 15 | Act: 02)
#   kLeftShoulderYaw     -> left_shoulder_yaw_joint        (Obs: 19 | Act: 04)
#   kLeftElbow           -> left_elbow_joint               (Obs: 21 | Act: 06)
#   kLeftWristRoll       -> left_wrist_roll_joint          (Obs: 23 | Act: 08)
#   kLeftWristPitch      -> left_wrist_pitch_joint         (Obs: 25 | Act: 10)
#   kLeftWristYaw        -> left_wrist_yaw_joint           (Obs: 27 | Act: 12)
#   kRightShoulderPitch  -> right_shoulder_pitch_joint     (Obs: 12 | Act: 01)
#   kRightShoulderRoll   -> right_shoulder_roll_joint      (Obs: 16 | Act: 03)
#   kRightShoulderYaw    -> right_shoulder_yaw_joint       (Obs: 20 | Act: 05)
#   kRightElbow          -> right_elbow_joint              (Obs: 22 | Act: 07)
#   kRightWristRoll      -> right_wrist_roll_joint         (Obs: 24 | Act: 09)
#   kRightWristPitch     -> right_wrist_pitch_joint        (Obs: 26 | Act: 11)
#   kRightWristYaw       -> right_wrist_yaw_joint          (Obs: 28 | Act: 13)
#   kLeftHandThumb0      -> left_hand_thumb_0_joint        (Obs: 31 | Act: 16)
#   kLeftHandThumb1      -> left_hand_thumb_1_joint        (Obs: 37 | Act: 22)
#   kLeftHandThumb2      -> left_hand_thumb_2_joint        (Obs: 41 | Act: 26)
#   kLeftHandMiddle0     -> left_hand_middle_0_joint       (Obs: 30 | Act: 15)
#   kLeftHandMiddle1     -> left_hand_middle_1_joint       (Obs: 36 | Act: 21)
#   kLeftHandIndex0      -> left_hand_index_0_joint        (Obs: 29 | Act: 14)
#   kLeftHandIndex1      -> left_hand_index_1_joint        (Obs: 35 | Act: 20)
#   kRightHandThumb0     -> right_hand_thumb_0_joint       (Obs: 34 | Act: 19)
#   kRightHandThumb1     -> right_hand_thumb_1_joint       (Obs: 40 | Act: 25)
#   kRightHandThumb2     -> right_hand_thumb_2_joint       (Obs: 42 | Act: 27)
#   kRightHandIndex0     -> right_hand_index_0_joint       (Obs: 32 | Act: 17)
#   kRightHandIndex1     -> right_hand_index_1_joint       (Obs: 38 | Act: 23)
#   kRightHandMiddle0    -> right_hand_middle_0_joint      (Obs: 33 | Act: 18)
#   kRightHandMiddle1    -> right_hand_middle_1_joint      (Obs: 39 | Act: 24)
# --- End of Joint Name Information ---

