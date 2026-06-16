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


def ensure_reachable(config: dict) -> bool:
    """Ensure the internet is reachable for the Pegasus stages.

    The external SSID (CJ86GJI4_5G) auto-connects at the system level, and a manual
    ``netsh wlan connect`` to it needs elevation (``netsh`` also can't read the SSID
    without Location access). So we DON'T force a switch when already online — we only
    fall back to an explicit switch if the internet is actually down.
    """
    ip = str(config["wifi"]["internet_check_ip"])
    if ping(ip):
        return True                              # already online — don't disturb Wi-Fi
    log.info("Internet unreachable (ping %s failed); switching Wi-Fi to external.", ip)
    switch_wifi(config["wifi"]["external"])
    return ping(ip)


def disconnect_wifi() -> bool:
    """Drop the current Wi-Fi association (``netsh wlan disconnect``).

    Used to leave the LAN (OMAP-Motion-5G) after a deploy: dropping it lets the
    auto-connect enterprise SSID (CJ86GJI4_5G) re-associate WITHOUT a manual
    ``netsh wlan connect`` (which needs elevation). Disconnect itself does not.
    """
    try:
        r = subprocess.run(["netsh", "wlan", "disconnect", f"interface={wifi_switch.ADAPTER}"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        log.info("Wi-Fi disconnect (rc=%d)", r.returncode)
        return r.returncode == 0
    except Exception as e:                        # noqa: BLE001
        log.warning("disconnect errored: %s", e)
        return False


def ensure_external(config: dict) -> bool:
    """收尾: get back on the internet after the LAN deploy, confirmed by ping.

    The deploy LAN (OMAP-Motion-5G) has no internet, so after deploy we must leave it. Two
    best-effort moves, because a direct ``netsh connect`` to the enterprise SSID needs
    elevation/Location and silently fails (that's why a prior run stayed stuck on the LAN):

      1. drop the LAN so an auto-connect external SSID can re-associate on its own, and
      2. also attempt an explicit switch back (works if elevated / the profile allows).

    Truth is ping reachability, NOT netsh's SSID read. If still offline after the timeout we
    log a LOUD warning telling the user to switch manually (rather than failing silently).
    """
    ip = str(config["wifi"]["internet_check_ip"])
    # STABLE check (not a single ping): right after a deploy the Wi-Fi can be mid-re-association
    # and a lone ping may transiently succeed, fooling us into skipping the real switch (which is
    # exactly how a finalize once left the laptop off CJ). Require several spaced probes.
    if _online_stable(ip):
        log.info("Already online (stable ping %s); external Wi-Fi is up.", ip)
        return True
    ext = config["wifi"]["external"]
    log.info("Offline/unstable after deploy; restoring internet (drop LAN + try switch to %s).", ext)
    disconnect_wifi()                                # let an auto-connect SSID take over
    try:
        switch_wifi(ext, retries=2)                  # best-effort explicit connect
    except Exception as e:                           # noqa: BLE001
        log.warning("explicit switch to %s failed (%s); relying on auto-reconnect.", ext, e)
    # Wait for STABLE connectivity, not just one ping that might be a re-association blip.
    deadline = time.time() + int(config["wifi"].get("net_ready_timeout_sec", 60))
    while time.time() < deadline:
        if _online_stable(ip, checks=2, gap=2):
            log.info("Internet restored (stable) on the external network.")
            return True
        time.sleep(3)
    log.warning("⚠ COULD NOT restore internet automatically — still off the internet (deploy LAN %s). "
                "Connecting to the enterprise SSID via netsh needs elevation/Location, so "
                "please switch Wi-Fi back to %s MANUALLY.",
                config["wifi"]["local"], wifi_switch.NETWORKS.get(ext, ext))
    return False


def _online_stable(ip: str, checks: int = 3, gap: int = 2) -> bool:
    """True only if ``ping`` succeeds on every one of ``checks`` probes spaced ``gap`` seconds
    apart — so a transient blip during Wi-Fi re-association doesn't read as 'online'."""
    for i in range(checks):
        if not ping(ip):
            return False
        if i < checks - 1:
            time.sleep(gap)
    return True


def switch_wifi(key_or_ssid: str, timeout: int = 20, retries: int = 3,
                retry_wait: int = 4) -> bool:
    """Switch the Wi-Fi adapter to a network given by key ('cj'/'omap') or full SSID.

    Retries a few times: right after dropping the other network the target profile can be
    momentarily "not available to connect" (and an auto-connect SSID may re-associate on
    its own), so a single attempt can spuriously report failure.
    """
    ssid = wifi_switch.NETWORKS.get(key_or_ssid, key_or_ssid)
    if wifi_switch.current_ssid() == ssid:
        log.info("Wi-Fi already on %s.", ssid)
        return True
    for attempt in range(1, retries + 1):
        log.info("Switching Wi-Fi -> %s (attempt %d/%d)", ssid, attempt, retries)
        if wifi_switch.connect(ssid, timeout=timeout) or wifi_switch.current_ssid() == ssid:
            log.info("Wi-Fi connected: %s", ssid)
            return True
        if attempt < retries:
            time.sleep(retry_wait)
            if wifi_switch.current_ssid() == ssid:   # auto-reconnect may have caught up
                log.info("Wi-Fi connected: %s (auto)", ssid)
                return True
    log.warning("Wi-Fi switch to %s FAILED after %d attempts (now on %s).",
                ssid, retries, wifi_switch.current_ssid() or "(none)")
    return False
