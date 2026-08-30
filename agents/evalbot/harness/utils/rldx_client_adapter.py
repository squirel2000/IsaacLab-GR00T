"""
rldx_client_adapter.py

Adapter that lets the IsaacLab rollout loop drive an RLWRLD RLDX-1 policy server.

RLDX-1 is not GR00T-derived (Qwen3-VL-8B backbone + Multi-Stream Action Transformer), but
its wire contract is close to GR00T N1.6's: ZeroMQ REQ/REP with msgpack, nested observation
dicts, and 16-step action chunks. So this adapter is a translation layer, not a new
harness — the comparison runs through the same run_eval.py, task and metrics as the GR00T
runs, which is part of what makes it fair.

Observation format RLDX-1 validates (rldx/policy/observation_validator.py):
    {
      "video":    {key: np.ndarray[np.uint8,   (B, T, H, W, C)]},
      "state":    {key: np.ndarray[np.float32, (B, T, D)]},
      "language": {key: list[list[str]]},                  # shape (B, T)
    }
and it asserts ``video.shape[1] == len(modality_config["video"].delta_indices)``.

Single-frame only: we run RLDX-1-PT-IMG with ``video_length 1`` so the video modality has
``delta_indices=[0]`` and T is always 1. That keeps the comparison symmetric with the
single-frame GR00T baselines and means no client-side frame buffering. If a multi-frame
variant is ever run, the client — not the server — must buffer past frames at the
action-step offsets in ``delta_indices`` and send them oldest-first; the validator says so
explicitly and will reject a T mismatch.

Action returned by the server is ``{group: (B, 16, D)}``; the rollout loop wants
``{"action.<group>": (16, D)}``, matching what joint_mapper already consumes from the GR00T
adapter.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


class RldxClientAdapter:
    """Unified ``get_action(obs)`` over an RLDX-1 policy server.

    Parameters
    ----------
    host, port:
        Where the RLDX-1 server is listening.
    timeout_ms:
        ZeroMQ receive timeout for a single call.
    api_token:
        Optional shared secret; the server rejects calls without it when configured.
    connect_timeout:
        How long to keep polling ``ping()`` while the server loads its checkpoint. A 6.9B
        bf16 model takes ~30 s to become ready, and the IsaacSim client boots in parallel,
        so polling beats failing on the first ping.
    session_id:
        Only needed for the stateful memory module. Left unset for can-sorting, where memory
        is disabled — see the note on ``options`` in :meth:`get_action`.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = 30000,
        api_token: str | None = None,
        connect_timeout: float = 900.0,
        session_id: str | None = None,
    ) -> None:
        try:
            from rldx.policy.server_client import PolicyClient
        except ModuleNotFoundError as e:
            raise ImportError(
                "\n[RLDX-1 CLIENT NOT FOUND]\n"
                "Could not import rldx.policy.server_client. The rollout process needs the "
                "RLDX-1 package importable — add the RLDX-1 checkout to PYTHONPATH via the "
                "policy config's 'client_pythonpath', as the N1.7 config does.\n"
            ) from e

        # strict=False: PolicyClient.check_observation/check_action deliberately raise
        # NotImplementedError, so strict mode is unusable from the client side. The server
        # validates the observation anyway.
        self._client = PolicyClient(
            host=host, port=port, timeout_ms=timeout_ms, api_token=api_token, strict=False
        )
        self._session_id = session_id
        self._reset_pending = True

        print(f"\nTrying to connect to the RLDX-1 policy server at {host}:{port}.")
        deadline = time.monotonic() + connect_timeout
        while not self._client.ping():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Cannot connect to the RLDX-1 policy server at {host}:{port} "
                    f"after {connect_timeout:.0f}s"
                )
            print("  ...server not ready yet, retrying")
            time.sleep(2)
        print("\nSuccessfully connected to the RLDX-1 policy server.")

        try:
            modality_configs = self._client.get_modality_config()
            print(f"Retrieved modality keys: {list(modality_configs.keys())}")
            self._video_keys = list(getattr(modality_configs.get("video"), "modality_keys", []) or [])
        except Exception as e:  # noqa: BLE001 - informative, not fatal
            print(f"  (could not read modality config: {e})")
            self._video_keys = []

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _to_nested_obs(flat: dict[str, Any]) -> dict[str, Any]:
        """Flat rollout obs -> RLDX-1's nested (B, T, ...) observation.

        Incoming (as the rollout loop and the GR00T adapter both use it):
            "video.<camera>"                     (H, W, C) or (1, H, W, C) uint8
            "state.<group>"                      (D,)
            "annotation.human.task_description"  str
        """
        video: dict[str, np.ndarray] = {}
        state: dict[str, np.ndarray] = {}

        for key, value in flat.items():
            if key.startswith("video."):
                arr = np.asarray(value)
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8)
                # Normalise to (H, W, C), then expand to (B=1, T=1, H, W, C).
                if arr.ndim == 4:
                    arr = arr[0]
                if arr.ndim != 3:
                    raise ValueError(
                        f"Video '{key}' must reduce to (H, W, C); got {np.asarray(value).shape}"
                    )
                video[key.split(".", 1)[1]] = arr[None, None, ...]

            elif key.startswith("state."):
                arr = np.asarray(value, dtype=np.float32).reshape(1, 1, -1)
                state[key.split(".", 1)[1]] = arr

        # language: list[list[str]] with shape (B, T)
        lang_key = "annotation.human.task_description"
        text = flat.get(lang_key, "")
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        language = {lang_key: [[str(text)]]}

        return {"video": video, "state": state, "language": language}

    @staticmethod
    def _to_flat_action(action: dict[str, Any]) -> dict[str, np.ndarray]:
        """{group: (B, T, D)} -> {"action.<group>": (T, D)}."""
        out: dict[str, np.ndarray] = {}
        for key, value in action.items():
            arr = np.asarray(value)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            out[f"action.{key}"] = arr
        return out

    # ------------------------------------------------------------------- public
    def reset(self) -> None:
        """Mark an episode boundary. Call at the start of each episode."""
        self._reset_pending = True

    def get_action(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        # `options` carries memory bookkeeping (reset_memory / session_ids). With the memory
        # module disabled it is inert, so we send it only when a session id was configured —
        # sending nothing is the lower-risk default and keeps the payload identical to what
        # a stateless policy expects.
        options: dict[str, Any] | None = None
        if self._session_id is not None:
            options = {"session_ids": [self._session_id], "reset_memory": [self._reset_pending]}
        self._reset_pending = False

        action, _info = self._client.get_action(self._to_nested_obs(obs), options)
        return self._to_flat_action(action)
