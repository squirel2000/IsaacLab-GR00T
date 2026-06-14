#!/usr/bin/env python3
"""
wifi_switch.py - connect exactly ONE Wi-Fi network at a time on a single adapter.

Switches the built-in Intel adapter ("Wi-Fi") between two 5 GHz networks:
    CJ86GJI4_5G   and   OMAP-Motion-5G

A single Wi-Fi adapter can associate with only one access point at a time, so
connecting to one network automatically drops the other -- "one at a time" is
guaranteed by the hardware, no explicit disconnect needed.

Note: both target SSIDs are 5 GHz, so the switch must run on the Intel AX211
built-in adapter. A 2.4 GHz-only USB dongle cannot reach either network.

Usage:
    python wifi_switch.py omap      # connect OMAP-Motion-5G
    python wifi_switch.py cj        # connect CJ86GJI4_5G
    python wifi_switch.py toggle    # switch to whichever is NOT current
    python wifi_switch.py status    # show current connection (read-only)
"""

import argparse
import subprocess
import sys
import time

ADAPTER = "Wi-Fi"  # Intel AX211 built-in (the only 5 GHz-capable adapter)

# friendly key -> exact Wi-Fi profile / SSID name
NETWORKS = {
    "cj":   "CJ86GJI4_5G",
    "omap": "OMAP-Motion-5G",
}


def _run(args):
    # errors="replace": netsh on zh-TW Windows emits cp950 bytes that would
    # otherwise crash the default decoder; we never parse its body, only rc.
    return subprocess.run(args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def current_ssid():
    """Real SSID currently connected on ADAPTER (or '' if none).

    Parses `netsh wlan show interfaces` for the true SSID. This is the reliable
    source: Get-NetConnectionProfile.Name returns the Windows NLA *network*
    name (e.g. "corpnet.asus" for the SSID "CJ86GJI4_5G"), which does NOT equal
    the SSID. netsh requires Location access to be enabled for desktop apps.
    """
    r = _run(["netsh", "wlan", "show", "interfaces"])
    name = None
    for line in r.stdout.splitlines():
        s = line.strip()
        if s.startswith("Name") and ":" in s:
            name = s.split(":", 1)[1].strip()
        # "SSID" line (the "AP BSSID" line starts with "AP", so no clash)
        elif s.startswith("SSID") and ":" in s and name == ADAPTER:
            return s.split(":", 1)[1].strip()
    return ""


def connect(ssid, timeout=15):
    """Connect ADAPTER to ssid; poll until connected or timeout. Returns bool."""
    r = _run(["netsh", "wlan", "connect", f"name={ssid}", f"interface={ADAPTER}"])
    if r.returncode != 0:
        print(f"  netsh error: {(r.stdout + r.stderr).strip()}")
        return False
    deadline = time.time() + timeout
    while time.time() < deadline:
        if current_ssid() == ssid:
            return True
        time.sleep(1)
    return current_ssid() == ssid


def main():
    ap = argparse.ArgumentParser(
        description="Switch one Wi-Fi network at a time (CJ86GJI4_5G / OMAP-Motion-5G).")
    ap.add_argument("action", choices=list(NETWORKS) + ["toggle", "status"])
    args = ap.parse_args()

    cur = current_ssid()

    if args.action == "status":
        print(f"Current on {ADAPTER}: {cur or '(not connected)'}")
        return

    if args.action == "toggle":
        key = "cj" if cur == NETWORKS["omap"] else "omap"
    else:
        key = args.action

    target = NETWORKS[key]
    if cur == target:
        print(f"Already connected to {target}.")
        return

    print(f"Switching {ADAPTER}: {cur or '(none)'} -> {target} ...")
    if connect(target):
        print(f"OK: connected to {target}.")
    else:
        print(f"FAILED to reach {target} (in range? profile exists?). "
              f"Now on: {current_ssid() or '(none)'}")
        sys.exit(1)


if __name__ == "__main__":
    main()
