"""Checkpoint download module — reuses run_finetune.download (resumable + sha256).

``run_finetune.download`` already streams the server-side ``.zip`` with HTTP Range resume
and verifies both size and sha256, so this module only wraps it with an outer retry loop
and records the verified local path/size into the pipeline state for the report stage.
"""
from __future__ import annotations

import time
from pathlib import Path

import run_finetune as rf
import pipeline_config as pc
from gpu_monitor import connect_with_retry
from training_monitor import build_finetune_cfg
from pipeline_logging import get_logger

log = get_logger("download")

RETRY_MAX = 3


def run(config: dict, state) -> Path:
    """DOWNLOADING stage entry point: fetch + verify the finished run's zip locally."""
    _, profile = pc.resolve_profile(config, state.profile)
    cfg = build_finetune_cfg(config, profile, state.get("gpu_id") or 0)
    info = rf.load_state(cfg)                       # run_finetune's own <run>.run.json
    s = connect_with_retry(config)

    last = None
    for attempt in range(1, RETRY_MAX + 1):
        try:
            local = rf.download(s, cfg, info)       # resumable + size + sha256 verified
            size = Path(local).stat().st_size
            state.record_output("zip_local", str(local))
            state.record_output("zip_size", size)
            log.info("Downloaded + verified %s (%d bytes).", local, size)
            return Path(local)
        except SystemExit as e:                     # verification/status error
            last = e
            log.warning("download attempt %d/%d failed: %s", attempt, RETRY_MAX, e)
            time.sleep(10)
        except Exception as e:                      # noqa: BLE001 - network etc.: reconnect
            last = e
            log.warning("download attempt %d/%d errored (%s); reconnecting",
                        attempt, RETRY_MAX, type(e).__name__)
            s = connect_with_retry(config)
            time.sleep(10)
    raise RuntimeError(f"download failed after {RETRY_MAX} attempts: {last}")
