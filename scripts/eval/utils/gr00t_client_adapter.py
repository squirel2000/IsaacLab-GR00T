"""
gr00t_client_adapter.py

A compatibility wrapper for connecting IsaacLab to different GR00T policy server versions (e.g. N1.5, N1.6).

This class hides API and import differences between GR00T versions and exposes
a unified `get_action(obs)` interface to the caller.
"""

import time

import numpy as np

class Gr00tClientAdapter:
    """
    Gr00tClientAdapter is a compatibility adapter for GR00T policy servers.

    It allows IsaacLab to connect to different GR00T versions (e.g. N1.5, N1.6) using a unified interface, hiding version-specific import paths and API differences.
    Including observation and action formatting.

    Parameters
    ----------
    version : str
        GR00T version identifier (e.g. "N1.5", "N1.6").
    host : str
        Policy server host address (default: "localhost").
    port : int
        Policy server port (default: 5555).
    """

    def __init__(self, version: str, host="localhost", port=5555, connect_timeout=300.0):
        self.version = version.lower()

        try:
            if self.version in ("n1.5", "1.5", "v1.5"):
                from gr00t.eval.robot import RobotInferenceClient
                self._client = RobotInferenceClient(host=host, port=port)
                self._mode = "N1.5"
            elif self.version in ("n1.6", "1.6", "v1.6", "n1.7", "1.7", "v1.7"):
                from gr00t.policy.server_client import PolicyClient
                self._client = PolicyClient(host=host, port=port)
                self._mode = "N1.6"
            else:
                raise ValueError(f"Unsupported gr00t version: {version}")
            
        except ModuleNotFoundError as e:
            raise ImportError(
                f"\n[GR00T VERSION MISMATCH]\n"
                f"You requested GR00T {self.version} client, but the client implementation "
                f"cannot be found in the current GR00T environment.\n"
                f"Please check that:\n"
                f"  - Your gr00t server environment is checked out to the {self.version} branch\n"
                f"  - OR your Python environment has GR00T {self.version} installed\n"
            ) from e
        
        print(f"\nTrying to connect to the GR00T {self._mode} policy server.")
        # The server may still be loading its model (the N1.7 1B+ DiT / Qwen3VL takes a
        # while) while the client boots IsaacSim in parallel, so poll instead of failing on
        # the first ping. N1.7's ping() self-times-out per attempt; N1.6's blocks until reply.
        deadline = time.monotonic() + connect_timeout
        while not self._client.ping():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Cannot connect to the gr00t policy server at {host}:{port} "
                    f"after {connect_timeout:.0f}s"
                )
            print("  ...server not ready yet, retrying")
            time.sleep(2)
        print(f"\nSuccessfully connect to the GR00T {self._mode} policy server.")

        modality_configs = self._client.get_modality_config()
        print(f"Retrieved modality keys: {list(modality_configs.keys())}")

    # ---------------------------
    # Private helpers
    # ---------------------------
    def _format_obs(self, N15_obs):
        """
        Convert GR00T N1.5 observation dict to GR00T N1.6 observation dict.

        N15_obs = {
            "video.camera_name": np.ndarray,  # rgb_image, Shape (1, H, W, C)
            "state.left_arm": np.ndarray,  # robot_state
            "state.right_arm": np.ndarray,
            ...
            "annotation.human.task_description": list[str], # ['do something']
        }
        """

        # N1.6 expects image in (B, T, H, W, C) format
        gr00t_video_obs_n16 = {}
        for key, value in N15_obs.items():
            if key.startswith("video."):
                camera_name = key.split(".", 1)[1]
                image_n16 = np.expand_dims(value, axis=1)  # Shape (1, 1, H, W, C)
                gr00t_video_obs_n16[camera_name] = image_n16
        
        # N1.6 expects state in (B, T, D) format
        gr00t_state_obs_n16 = {}
        for key, value in N15_obs.items():
            if "state" in key:
                value_float32 = value.astype(np.float32) if value.dtype != np.float32 else value
                # value shape is originally (D,), reshape to (1, 1, D)
                gr00t_state_obs_n16[key.replace("state.", "")] = value_float32.reshape(1, 1, -1)
        
        # N1.6 observation format
        N16_obs = {
            "video": gr00t_video_obs_n16,
            "state": gr00t_state_obs_n16,  # Each entry: (1, 1, D)
            "language": {
                "annotation.human.task_description": [N15_obs["annotation.human.task_description"]]  # [[str]] format
            }
        }

        return N16_obs

    def _reformat_action(self, gr00t_action):
        """
        Convert GR00T N1.6 action to GR00T N1.5 format (remove batch, add prefixes).
        """
        # N1.6 action format: {"left_arm": (1, 16, 7), "right_arm": (1, 16, 7), ...}
        # Need to convert to the format expected by joint_mapper(N1.5)
        # Remove batch dimension and convert to the format: {"action.left_arm": (16, 7), ...}
        gr00t_action_reformatted = {}
        for key, value in gr00t_action.items():
            # Remove batch dimension: (1, T, D) -> (T, D)
            action_squeezed = value.squeeze(0) if value.shape[0] == 1 else value
            # Add "action." prefix to match expected format
            gr00t_action_reformatted[f"action.{key}"] = action_squeezed

        return gr00t_action_reformatted


    def get_action(self, obs):
        if self._mode == "N1.5":
            action = self._client.get_action(obs)
            return action

        elif self._mode == "N1.6":
            action, info = self._client.get_action(self._format_obs(obs))
            return self._reformat_action(action)
