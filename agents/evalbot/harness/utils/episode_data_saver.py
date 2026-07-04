import json
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class EpisodeDataSaver:
    def __init__(self, output_dir, action_joint_names_list, video_fps, step_dt, task_description):
        self.output_dir = output_dir
        self.action_joint_names_list = action_joint_names_list
        self.video_fps = video_fps
        self.step_dt = float(step_dt)
        self.task_description = list(task_description) if task_description else [""]
        # Cumulative across all episodes in this run — matches LeRobot's global `index` column.
        self._global_index = 0

    def save_episode(self, episode_counter, image_list, action_list, joint_pos_list, episode_result):
        """Save video and action data for the episode."""
        # Determine category folder based on termination reason
        category_folder = episode_result

        # Create a dedicated folder for the episode's data
        episode_dir = os.path.join(self.output_dir, category_folder, f"episode_{episode_counter:03d}")
        os.makedirs(episode_dir, exist_ok=True)

        video_path = os.path.join(episode_dir, f"episode_{episode_counter:03d}.mp4")
        fourcc = cv2.VideoWriter.fourcc(*'mp4v') # Codec for .mp4

        # `gr00t_infer_agent.py` calls `image_list.append(camera_image[0])`,
        # which strips the batch dim — so each element here is (H, W, C).
        # An older comment claimed (1, H, W, C); reading `shape[1], shape[2]`
        # under that assumption gave `(W, 3)` and produced a 3-pixel-wide
        # VideoWriter, which silently failed every `write()` call as
        # `FFmpeg: Failed to write frame`. Squeeze defensively in case a
        # future caller leaves the leading batch dim in place.
        first = image_list[0]
        if first.ndim == 4 and first.shape[0] == 1:
            image_list = [im[0] if im.ndim == 4 and im.shape[0] == 1 else im for im in image_list]
            first = image_list[0]
        if first.ndim != 3:
            raise ValueError(
                f"EpisodeDataSaver expected frames of shape (H, W, C); got {first.shape}"
            )
        frame_height, frame_width = first.shape[0], first.shape[1]
        video_writer = cv2.VideoWriter(video_path, fourcc, self.video_fps, (frame_width, frame_height))

        for img in image_list:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(img_bgr)
        video_writer.release()

        # Plot and save actions to CSV
        episode_actions = np.array(action_list)
        episode_joint_pos = np.array(joint_pos_list)
        csv_path = os.path.join(episode_dir, f"actions_ep{episode_counter:03d}.csv")
        header = ",".join([f"action_{name}" for name in self.action_joint_names_list] +
                          [f"state_{name}" for name in self.action_joint_names_list])
        np.savetxt(csv_path, np.hstack((episode_actions, episode_joint_pos)), delimiter=",", header=header, comments="")

        # LeRobot-style parquet (loose layout — sits inside the existing episode_NNN folder).
        self._save_episode_parquet(episode_dir, episode_counter, episode_actions, episode_joint_pos)
        self._save_episode_meta(episode_dir, episode_counter, episode_result, len(action_list))

        num_joints = episode_actions.shape[1]
        num_plots_per_fig = 4
        num_figs = (num_joints + num_plots_per_fig - 1) // num_plots_per_fig

        for fig_idx in range(num_figs):
            fig, axs = plt.subplots(num_plots_per_fig, 1, figsize=(10, 12), sharex=True)
            if num_plots_per_fig == 1: axs = np.array([axs])
            elif not isinstance(axs, np.ndarray): axs = np.array([axs])
            axs = axs.flatten()

            start_idx = fig_idx * num_plots_per_fig
            for i in range(num_plots_per_fig):
                joint_idx = start_idx + i
                if joint_idx >= num_joints:
                    axs[i].axis('off')
                    continue
                
                joint_name = self.action_joint_names_list[joint_idx]
                axs[i].plot(episode_actions[:, joint_idx], label="Action")
                axs[i].plot(episode_joint_pos[:, joint_idx], label="State", linestyle="--", alpha=0.7)
                
                axs[i].set_title(joint_name)
                axs[i].legend(fontsize='x-small')
                axs[i].grid(True)
            
            axs[min(num_plots_per_fig, num_joints - start_idx) - 1].set_xlabel("Steps")
            plt.tight_layout()
            plt.savefig(os.path.join(episode_dir, f"actions_ep{episode_counter:03d}_part{fig_idx+1}.png"))
            plt.close(fig)

    def _save_episode_parquet(self, episode_dir, episode_counter, episode_actions, episode_joint_pos):
        """Write a LeRobot-v2-style parquet (loose layout — same schema, kept inside episode_NNN/)."""
        actions = np.asarray(episode_actions, dtype=np.float64)
        states = np.asarray(episode_joint_pos, dtype=np.float64)
        num_steps = actions.shape[0]

        episode_index = episode_counter - 1  # 0-based to match dataset convention
        frame_index = np.arange(num_steps, dtype=np.int64)
        global_index = np.arange(self._global_index, self._global_index + num_steps, dtype=np.int64)
        self._global_index += num_steps

        df = pd.DataFrame({
            "observation.state": list(states),
            "action": list(actions),
            "timestamp": (frame_index * self.step_dt).astype(np.float64),
            "frame_index": frame_index,
            "episode_index": np.full(num_steps, episode_index, dtype=np.int64),
            "index": global_index,
            "task_index": np.zeros(num_steps, dtype=np.int64),
        })
        df.to_parquet(os.path.join(episode_dir, f"episode_{episode_counter:03d}.parquet"), index=False)

    def _save_episode_meta(self, episode_dir, episode_counter, episode_result, num_steps):
        meta = {
            "episode_index": episode_counter - 1,
            "episode_counter": episode_counter,
            "length": int(num_steps),
            "result": episode_result,
            "task_index": 0,
            "task": self.task_description[0] if self.task_description else "",
        }
        with open(os.path.join(episode_dir, f"episode_{episode_counter:03d}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)