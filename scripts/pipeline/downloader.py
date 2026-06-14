"""Checkpoint download module — reuses run_finetune.download (resumable + sha256).

``run_finetune.download`` already streams the server-side ``.zip`` with HTTP Range resume
and verifies both size and sha256, so this module only wraps it with an outer retry loop
and records the verified local path/size into the pipeline state for the report stage.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import run_finetune as rf
import pipeline_config as pc
import pipeline_progress as pp
from gpu_monitor import connect_with_retry
from training_monitor import build_finetune_cfg, info_from_state
from pipeline_logging import get_logger

log = get_logger("download")


def _size(p: Path) -> int:
    """Current byte size of a (possibly absent) file — one stat, no race."""
    try:
        return p.stat().st_size
    except OSError:
        return 0


def run(config: dict, state) -> Path:
    """DOWNLOADING stage entry point: fetch + verify the finished run's zip locally."""
    retry_max = int(config["training"]["connect_retry_max"])
    _, profile = pc.resolve_profile(config, state.profile)
    cfg = build_finetune_cfg(config, profile, state.get("gpu_id") or 0)
    info = info_from_state(cfg, state)             # reconstructed from pipeline state
    s = connect_with_retry(config)
    local_path = Path(cfg["local_dir"]) / info["zip_name"]
    try:
        total = int(rf.read_text(s, f"{info['state_rel']}/zip.size"))
    except Exception:                              # noqa: BLE001
        total = None

    # Side thread reports live download % by watching the growing partial file (rf.download
    # has no progress hook; this avoids modifying it).
    stop = threading.Event()

    def _feeder():
        t_last, b_last = time.time(), _size(local_path)
        while not stop.is_set():
            cur, now = _size(local_path), time.time()
            rate = (cur - b_last) / (now - t_last) if now > t_last else 0.0
            t_last, b_last = now, cur
            pp.update("download", done=cur, total=total,
                      pct=(round(100 * cur / total, 1) if total else None),
                      rate_mbps=round(rate / 1e6, 1),
                      eta_sec=(round((total - cur) / rate) if total and rate > 0 else None))
            stop.wait(1.5)

    th = threading.Thread(target=_feeder, daemon=True)
    th.start()

    last = None
    try:
        for attempt in range(1, retry_max + 1):
            try:
                got = rf.download(s, cfg, info)     # resumable + size + sha256 verified
                size = _size(Path(got))
                state.record_outputs(zip_local=str(got), zip_size=size)
                pp.update("download", done=size, total=total or size, pct=100,
                          rate_mbps=0, eta_sec=0)
                log.info("Downloaded + verified %s (%d bytes).", got, size)
                return Path(got)
            except SystemExit as e:                 # verification/status error
                last = e
                log.warning("download attempt %d/%d failed: %s", attempt, retry_max, e)
                time.sleep(10)
            except Exception as e:                  # noqa: BLE001 - network etc.: reconnect
                last = e
                log.warning("download attempt %d/%d errored (%s); reconnecting",
                            attempt, retry_max, type(e).__name__)
                s = connect_with_retry(config)
                time.sleep(10)
        raise RuntimeError(f"download failed after {retry_max} attempts: {last}")
    finally:
        stop.set()
        th.join(timeout=2)
