"""Network helpers shared by the deployer and the runner: Wi-Fi switch + ping/wait.

Wi-Fi switching reuses ``wifi_switch.py`` (netsh). Ping is cross-platform: Windows uses
``-n <count> -w <ms>``, POSIX uses ``-c <count> -W <sec>``.
"""
from __future__ import annotations

import subprocess
import sys
import time

import wifi_switch
from pipeline_logging import get_logger

log = get_logger("network")

_IS_WINDOWS = sys.platform.startswith("win")


def _ping_args(host: str, count: int, timeout_ms: int) -> list[str]:
    """Build the platform-correct ping command (pure; unit-tested)."""
    if _IS_WINDOWS:
        return ["ping", "-n", str(count), "-w", str(timeout_ms), host]
    secs = max(1, round(timeout_ms / 1000))
    return ["ping", "-c", str(count), "-W", str(secs), host]


def ping(host: str, count: int = 1, timeout_ms: int = 1000) -> bool:
    """Return True if ``host`` answers a ping within the timeout."""
    try:
        r = subprocess.run(_ping_args(host, count, timeout_ms),
                           capture_output=True, text=True)
        return r.returncode == 0
    except Exception as e:                       # noqa: BLE001
        log.warning("ping %s errored: %s", host, e)
        return False


def wait_for_host(host: str, timeout_sec: int = 60, interval: int = 3) -> bool:
    """Poll ``host`` until it answers a ping or ``timeout_sec`` elapses."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if ping(host):
            log.info("%s is reachable.", host)
            return True
        time.sleep(interval)
    return ping(host)


def switch_wifi(key_or_ssid: str, timeout: int = 20) -> bool:
    """Switch the Wi-Fi adapter to a network given by key ('cj'/'omap') or full SSID."""
    ssid = wifi_switch.NETWORKS.get(key_or_ssid, key_or_ssid)
    cur = wifi_switch.current_ssid()
    if cur == ssid:
        log.info("Wi-Fi already on %s.", ssid)
        return True
    log.info("Switching Wi-Fi: %s -> %s", cur or "(none)", ssid)
    ok = wifi_switch.connect(ssid, timeout=timeout)
    if ok:
        log.info("Wi-Fi connected: %s", ssid)
    else:
        log.warning("Wi-Fi switch to %s FAILED (now on %s).",
                    ssid, wifi_switch.current_ssid() or "(none)")
    return ok
