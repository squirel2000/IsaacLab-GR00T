"""Single retry helper for the pipeline's connect/transfer loops.

Replaces the near-identical ``for attempt in range(...)`` loops that were copy-pasted
across gpu_monitor / deployer. Call ``retry(fn, ...)``; ``on_error(attempt, exc)`` runs
after each failure (e.g. to close a dead client) and ``interval`` seconds are slept
between attempts.
"""
from __future__ import annotations

import time

from pipeline_logging import get_logger

log = get_logger("retry")


def retry(fn, attempts: int = 3, interval: int = 10, label: str = "operation",
          on_error=None, exceptions: tuple = (Exception,)):
    """Call ``fn`` up to ``attempts`` times; return its result, or raise after the last."""
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except exceptions as e:                  # noqa: BLE001 - caller picks exception set
            last = e
            log.warning("%s failed (attempt %d/%d): %s", label, attempt, attempts, e)
            if on_error is not None:
                on_error(attempt, e)
            if attempt < attempts:
                time.sleep(interval)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last}")
