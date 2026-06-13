"""Tiny logging helper: one rotating log file per module under ``logs/`` + console.

    from pipeline_logging import get_logger
    log = get_logger("gpu_monitor")   # -> logs/gpu_monitor.log + stderr
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from pipeline_paths import LOGS_DIR

_FMT = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to ``logs/<name>.log`` (rotating) and the console.

    Idempotent: calling it twice for the same name reuses the handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:                      # already configured
        return logger
    logger.setLevel(level)
    LOGS_DIR.mkdir(exist_ok=True)
    fh = RotatingFileHandler(LOGS_DIR / f"{name}.log", maxBytes=2_000_000,
                             backupCount=3, encoding="utf-8")
    fh.setFormatter(_FMT)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(_FMT)
    logger.addHandler(ch)
    logger.propagate = False
    return logger
