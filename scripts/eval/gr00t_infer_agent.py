# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Gr00t agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Pour-Water-OpenArm-DexHand-v0",
    help="Name of the task. Options: 'Isaac-Cabinet-Pour-G1-Abs-v0', 'Isaac-Can-Sorting-OpenArm-DexHand-v0', 'Isaac-Pour-Water-OpenArm-DexHand-v0'."
)
parser.add_argument("--policy", type=str, default="gr00t", help="Policy backend to use. Examples: gr00t, starvla.")
parser.add_argument("--policy_config", type=str, default="", help="Optional JSON/YAML policy backend config.")
parser.add_argument("--port", type=int, help="Port number for the policy server.", default=5555)
parser.add_argument("--host", type=str, help="Host address for the policy server.", default="localhost")
parser.add_argument("--gr00t_ver", type=str, default="N1.5", choices=["N1.5", "N1.6", "N1.7"], help="GR00T inference server version.", )

parser.add_argument("--save_video", action="store_true", default=False, help="Save the data from camera RGB image.")
parser.add_argument("--save_dir", type=str, default="output/infer_record/openarm_cansorting_movebasket_n1.5_200k_ds4", help="Folder path for saving video and image.")
parser.add_argument("--max_eps_num", type=int, default=100, help="Max number of inference episodes.")
parser.add_argument("--filter", action="store_true", default=False, help="Use filters to process prediction results.")
parser.add_argument("--g1_hand_type", default="inspire", choices=["trihand", "inspire"], help="DexHands type of G1.")
parser.add_argument("--openarm_hand_type", default="linkerhand_o6", choices=["leaphand_right", "linkerhand_o6"], help="DexHands type of OpenArm.")
parser.add_argument("--pov_list", nargs="+", type=str, default=["head"], choices=["head", "wrist_R", "wrist_L"], help="Camera perspective list.")
parser.add_argument("--multitask", action="store_true", default=True, help="Whether to use the multitask version of the environment (only for can sorting task).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# Force enable cameras for this script by modifying the parsed arguments
args_cli.enable_cameras = True

##########
# Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
# pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
import pinocchio  # noqa: F401
##########

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from typing import cast
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

# PLACEHOLDER: Extension template (do not remove this comment)
"""Data collection setup"""
import os
import numpy as np
import time
from datetime import datetime

"""gr00t integration"""
import carb
carb_settings_iface = carb.settings.get_settings()
carb_settings_iface.set_bool("/current_env/use_joint_space", True)

if "G1" in args_cli.task:
    ROBOT_TYPE = "g1_"+ args_cli.g1_hand_type
elif "OpenArm" in args_cli.task:
    ROBOT_TYPE = "openarm_"+ args_cli.openarm_hand_type
else:
    raise NotImplementedError("Currently only for G1 or OpenArm.")
carb_settings_iface.set_string("/current_env/robot_type", ROBOT_TYPE)

from utils.policy_client_factory import build_policy_client
from utils.joint_mapper import JointMapper
from utils.filter import LowPassFilter, MovingAverageFilter
from utils.episode_data_saver import EpisodeDataSaver
from utils.run_manifest import build_initial_manifest, finalize_manifest, write_manifest


TASK_SCENES = ["Cabinet-Pour", "Can-Sorting", "Cube-Stack", "Pour-Water"]
# Determine TASK_DESCRIPTION based on the selected task
if "Can-Sorting" in args_cli.task or "PickPlace-Only" in args_cli.task:
    TASK_DESCRIPTION = ["pick and sort a red or blue can"]
    if "OpenArm" in args_cli.task:
        from isaaclab_tasks.manager_based.manipulation.playground_openarm.dexhand_bimanual.task_scenes.can_sorting.mdp.terminations import task_done
    elif "G1" in args_cli.task:
        from isaaclab_tasks.manager_based.manipulation.playground_g1.task_scenes.can_sorting.mdp.terminations import task_done
elif "Cube-Stack" in args_cli.task:
    TASK_DESCRIPTION = ["stack the cubes in the order of red, green and blue."]
    if "OpenArm" in args_cli.task:
        from isaaclab_tasks.manager_based.manipulation.playground_openarm.dexhand_bimanual.task_scenes.cube_stack.mdp.terminations import task_done
    elif "G1" in args_cli.task:
        from isaaclab_tasks.manager_based.manipulation.playground_g1.task_scenes.cube_stack.mdp.terminations import task_done
elif "Cabinet-Pour" in args_cli.task:
    TASK_DESCRIPTION = ["open the drawer, take the mug on the mug mat, and pour water from the bottle into the mug."]
    if "OpenArm" in args_cli.task:
        from isaaclab_tasks.manager_based.manipulation.playground_openarm.dexhand_bimanual.task_scenes.cabinet_pour.mdp.terminations import task_done
    elif "G1" in args_cli.task:
        from isaaclab_tasks.manager_based.manipulation.playground_g1.task_scenes.cabinet_pour.mdp.terminations import task_done
elif "Pour-Water" in args_cli.task:
    TASK_DESCRIPTION = ["take the mug on the mug mat, and pour water from the bottle into the mug."]
    if "OpenArm" in args_cli.task:
        from isaaclab_tasks.manager_based.manipulation.playground_openarm.dexhand_bimanual.task_scenes.pour_water.mdp.terminations import task_done
elif "Playground" in args_cli.task:
    TASK_DESCRIPTION = ["Perform the default behavior"]
    carb_settings_iface.set("/gr00t_infer/current_task", TASK_SCENES[0])
else:
    TASK_DESCRIPTION = ["Perform the default behavior"]

STABILIZATION_STEPS = 5

CAMERA_OBS = {
    "head": "rgb_image",
    "wrist_L": "wrist_L_image",
    "wrist_R": "wrist_R_image",
}

# Map the numeric target object color ID in observation to a readable name.
TARGET_COLOR_ID_TO_NAME = {
    0: "orange",
    1: "green",
}

def run_stabilization(env, idle_actions_tensor):
    """
    Runs stabilization steps by holding a default joint pose and returns the final observation.
    """
    print(f"\n[INFO] Stabilizing robot to default joint positions for {STABILIZATION_STEPS} steps...")
    for _ in range(STABILIZATION_STEPS):
        obs, _, _, _, _ = env.step(idle_actions_tensor)
    print("[INFO] Stabilization complete.")
    
    return obs

def main():
    """GR00T actions agent with Isaac Lab environment."""

    """gr00t inference client"""
    policy_client = build_policy_client(args_cli)

    # Set numpy print options to display floats with 3 decimal places
    np.set_printoptions(precision=3, suppress=True, floatmode='fixed')

    # create environment with configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    
    env_cfg.terminations.success = None # Judge through task_done()
    
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg).unwrapped)
    
    # Per-step wall/sim time (also doubles as 1/video_fps).
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    if args_cli.save_video:
        video_fps = 1.0 / step_dt

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    obs, _ = env.reset()

    # Initialize teleop interface (just for env_reset, not for robot eef control)
    do_env_reset = False
    def reset_env_and_episode():
        nonlocal do_env_reset
        do_env_reset = True
    
    task_description = TASK_DESCRIPTION
    current_scene_idx = 0
    def switch_task_scene():
        nonlocal current_scene_idx, task_description

        if args_cli.task == "Isaac-Playground-G1-Abs-v0":
            current_scene_idx+=1
            if current_scene_idx>=len(TASK_SCENES): current_scene_idx = 0
            carb_settings_iface.set("/current_env/task_scene", TASK_SCENES[current_scene_idx])
        
            if current_scene_idx==0:
                task_description = ["open the drawer, take the mug on the mug mat, and pour water from the bottle into the mug."]
            elif current_scene_idx==1:
                task_description = ["pick and sort a red or blue can."]
            elif current_scene_idx==2:
                task_description = ["stack the cubes in the order of red, green and blue."]
            else:
                task_description = ["Perform the default behavior."]
        
        reset_env_and_episode()
        print(f"\n[INFO] Change the current task scene to {TASK_SCENES[current_scene_idx]}: {task_description}")
        
    teleop_interface = Se3Keyboard(Se3KeyboardCfg())
    teleop_interface.add_callback("R", reset_env_and_episode)
    teleop_interface.add_callback("M", switch_task_scene)

    print("\n==== Teleoperation Interface Controls ====")
    print("  R: Reset the environment.")
    print("  M: Switch task scene and reset environment.", "Scenes list:",TASK_SCENES)
    print("========================================\n")
    teleop_interface.reset()
    
    
    # Initialize the JointMapper
    robot_articulation = env.scene.articulations["robot"]
    joint_mapper = JointMapper(env_cfg=env_cfg, robot_articulation=robot_articulation)

    # Create the default joint action tensor for stabilization ( env_cfg.actions.arm_action_cfg is JointPositionActionCfg due to /current_env/use_joint_space = True )
    action_joint_names_list = env_cfg.actions.arm_action_cfg.joint_names
    default_joint_positions_dict = env_cfg.scene.robot.init_state.joint_pos
    default_joint_action_np = np.zeros(len(action_joint_names_list), dtype=np.float32)
    for i, joint_name in enumerate(action_joint_names_list):
        default_joint_action_np[i] = default_joint_positions_dict.get(joint_name, 0.0)
    default_idle_actions_tensor = torch.tensor(default_joint_action_np, dtype=torch.float32, device=env.device).unsqueeze(0)

    # Initial stabilization run
    obs = run_stabilization(env, default_idle_actions_tensor)
    episode_start_sim_time = env.sim.current_time # Initialize after first stabilization

    # simulate environment
    episode_counter, step_counter, success_counter = 1, 0, 0
    terminated_counter, truncated_counter = 0, 0
    video_writer = None
    image_list = []
    action_list = []
    joint_pos_list = []
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    output_dir = os.path.join(args_cli.save_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    data_saver = None
    if args_cli.save_video:
        data_saver = EpisodeDataSaver(output_dir, action_joint_names_list, video_fps, step_dt, task_description)

    # Run-level manifest: checkpoint identity, env config, git SHA. Written
    # once now so a crash still leaves identity; overwritten at end with results.
    manifest = build_initial_manifest(
        args_cli=args_cli,
        robot_type=ROBOT_TYPE,
        task_description=task_description,
        step_dt=step_dt,
        action_joint_names=action_joint_names_list,
    )
    write_manifest(output_dir, manifest)

    MAX_EPS_NUM = args_cli.max_eps_num
    infer_time_per_epi = []
    all_inference_times = []

    filter = LowPassFilter(alpha=0.3)
    #filter = MovingAverageFilter(window_size=3)

    while simulation_app.is_running():
        
        # run everything in inference mode
        with torch.inference_mode():
            if do_env_reset:
                obs, _ = env.reset()
                teleop_interface.reset()
                print("[INFO] The environment was reset due to detection of [R/M] being pressed.")
                do_env_reset = False
                obs = run_stabilization(env, default_idle_actions_tensor) # Run stabilization again
                episode_start_sim_time = env.sim.current_time # Reset episode start time
                episode_counter += 1
                step_counter = 0      # Reset step_counter for the new episode
                image_list = []
                action_list = []
                joint_pos_list = []
                episode_result = ""

            # --- 1. Process observations ---
            # obs tensors have a batch dim of 1, even with unwrapped env
            robot_joint_pos = obs["robot_obs"]["robot_joint_pos"].cpu().numpy().astype(np.float64)
            isaac_robot_joint_pos_flat = robot_joint_pos[0]  # Index to get 1D array (num_joints,)
            
            # Select the correct camera observation(s) for GR00T
            selected_cameras = args_cli.pov_list
            camera_obs_keys = []
            for camera_name in selected_cameras:
                camera_key = CAMERA_OBS.get(camera_name)
                if camera_key not in obs["robot_obs"]:
                    raise KeyError(f"Selected camera key '{camera_key}' not found in obs['robot_obs'].")
                camera_obs_keys.append(camera_key)

            camera_images = {
                camera_name: obs["robot_obs"][camera_key].cpu().numpy().astype(np.uint8)
                for camera_name, camera_key in zip(selected_cameras, camera_obs_keys)
            }

            # --- 2. Prepare GR00T observation ---
            if args_cli.multitask:
                target_object_color_id = int(obs["scene_obs"]["target_object_color"].cpu().numpy().reshape(-1)[0])
                target_object_color = TARGET_COLOR_ID_TO_NAME.get(target_object_color_id, "unknown")
                if target_object_color == "orange":
                    task_description = ["place the can on the orange plate"]
                elif target_object_color == "green":
                    task_description = ["place the can on the green plate"]

            gr00t_state_obs = joint_mapper.map_isaac_obs_to_gr00t_state(isaac_robot_joint_pos_flat)
            gr00t_obs = {
                "annotation.human.task_description": task_description,
                **gr00t_state_obs
            }
            # Add all selected cameras to GR00T obs (N1.5 format)
            for camera_name, camera_image in camera_images.items():
                if camera_name == "head":   # For backward compatibility with N1.5, keep the original key "video.camera" for the head camera
                    gr00t_obs["video.camera"] = camera_image
                else:
                    gr00t_obs[f"video.camera_{camera_name}"] = camera_image

            # --- 3. Query GR00T policy server ---
            time_start = time.time()
            gr00t_action = policy_client.get_action(gr00t_obs)
            get_action_time = time.time() - time_start

            # --- 4. Map GR00T action to Isaac action gr00t_action is a dict, e.g., {"action.left_arm": (prediction_horizon, 7), ...} ---
            env_action_values_fully_step = joint_mapper.map_gr00t_action_to_isaac_action(gr00t_action)
            actions_seqs = torch.tensor(env_action_values_fully_step, dtype=torch.float32, device=env.device).unsqueeze(1) # (16, 1, 28)
            
            if args_cli.filter: 
                smoothed_actions = filter.filter(actions_seqs)
                actions_seqs=smoothed_actions

            # --- 5. Step environment ---
            for action in actions_seqs: # Step every predicted action
                obs, _, terminated, truncated, _ = env.step(action)    # (obs, reward, terminated, truncated, info)
                success = task_done(env).cpu().numpy()[0] if args_cli.task != "Isaac-Playground-G1-Abs-v0" else False
                camera_image = obs["robot_obs"][camera_obs_keys[0]].cpu().numpy().astype(np.uint8)
                image_list.append(camera_image[0])
                action_list.append(action.cpu().numpy()[0])
                joint_pos_list.append(obs["robot_obs"]["robot_joint_pos"].cpu().numpy()[0])
                step_counter += 1
                # Interrupt action sequence step
                if terminated or truncated or success: break

  
            current_sim_time = env.sim.current_time
            relative_episode_time = current_sim_time - episode_start_sim_time

            # print(f"Ep {episode_counter} | Step {step_counter} | SimTime {relative_episode_time:.2f}s: Inference: {get_action_time:.3f}s, "
            #     f"Right EE Pos/Quat: {right_eef_pos}, {right_eef_quat}, Object Pos: {target_object_pos}")
            print(f"Ep {episode_counter} | Step {step_counter} | SimTime {relative_episode_time:.2f}s: Inference: {get_action_time:.3f}s")
            infer_time_per_epi.append(get_action_time)
            all_inference_times.append(get_action_time)

            # --- 6. Check for termination and reset if necessary ---
            if terminated or truncated or success:
                print(f"Episode {episode_counter} finished after {step_counter} steps (Success: {success}, Terminated: {terminated}, Truncated: {truncated}).")
                if success: 
                    success_counter+=1
                    episode_result = "success"
                elif terminated:
                    terminated_counter+=1
                    episode_result = "terminated"
                elif truncated:
                    truncated_counter+=1
                    episode_result = "truncated"

                print(f"Success rate: {(success_counter/episode_counter)*100}% ({success_counter}/{episode_counter})")
                print(f"Inference time cost- "
                      f"Avg:{sum(infer_time_per_epi)/len(infer_time_per_epi):.3f}s, "
                      f"Max:{max(infer_time_per_epi):.3f}s, "
                      f"Min:{min(infer_time_per_epi):.3f}s "
                      f"({len(infer_time_per_epi)} infer nums)")
                
                if data_saver: # and (terminated or truncated): # or other condition (currently only records truncated/terminated)
                    data_saver.save_episode(episode_counter, image_list, action_list, joint_pos_list, episode_result)

                obs, _ = env.reset()  # Reset the environment
                obs = run_stabilization(env, default_idle_actions_tensor) # Run stabilization again
                episode_start_sim_time = env.sim.current_time # Reset episode start time
                episode_counter += 1
                step_counter = 0      # Reset step_counter for the new episode
                image_list = []
                action_list = []
                joint_pos_list = []
                infer_time_per_epi = []

            if episode_counter>MAX_EPS_NUM: 
                print(f"\n***The maximum number of episodes has been reached.***")
                print(f"***Success rate: {(success_counter/MAX_EPS_NUM)*100}% ({success_counter}/{MAX_EPS_NUM})***")
                break

    # Overwrite run manifest with final results so it's queryable after the run.
    finalize_manifest(
        manifest,
        episodes_total=MAX_EPS_NUM,
        success=success_counter,
        terminated=terminated_counter,
        truncated=truncated_counter,
        infer_times=all_inference_times,
    )
    write_manifest(output_dir, manifest)

    # Write simulation note
    with open(os.path.join(output_dir, "simulation_note.txt"), "a") as f:
        f.write(f"=== Simulation Result ===\n")
        f.write(f"Success rate: {(success_counter/MAX_EPS_NUM)*100:.2f}% ({success_counter}/{MAX_EPS_NUM}))\n")
        f.write(f"  Terminated: {terminated_counter}\n")
        f.write(f"  Truncated: {truncated_counter}\n")
        if all_inference_times:
            f.write(f"Inference time cost:\n")
            f.write(f"  Avg: {sum(all_inference_times)/len(all_inference_times):.3f}s\n")
            f.write(f"  Max: {max(all_inference_times):.3f}s\n")
            f.write(f"  Min: {min(all_inference_times):.3f}s\n")
        else:
            f.write("Inference time cost: N/A\n")
        
    # close the simulator
    if video_writer is not None: # Release writer if simulation ends mid-episode
        video_writer.release()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
