#!/usr/bin/env python3
"""wifi_switch.py - inspect / switch the Wi-Fi network on the built-in adapter.

This machine has ONE Wi-Fi adapter (Intel Wi-Fi 6E AX211, interface alias "Wi-Fi"); it is
associated with one network at a time and moved between:
    DIT_MB         (2.4 GHz - internet / GitHub push-pull)
    CJ86GJI4_5G    (5 GHz   - company MIS / remote H100 "Pegasus")
    OMAP-Motion-5G (5 GHz   - lab LAN / asus-4090)

STATUS works WITHOUT admin: it reads `Get-NetConnectionProfile` (the Windows NLA "network
name") and maps it to the SSID. We do NOT rely on `netsh wlan show interfaces` because on
this box it needs BOTH Location permission AND admin elevation, and otherwise returns
"error 5" — which is why the old version (netsh-only) wrongly reported "(not connected)".

SWITCHING (`netsh wlan connect`) REQUIRES admin elevation here. Without it, netsh returns
"error 5"; this script then tells you to switch manually (Wi-Fi flyout) or re-run elevated.

Usage:
    python wifi_switch.py status      # show current network   (no admin needed)
    python wifi_switch.py dit         # connect DIT_MB          (needs admin)
    python wifi_switch.py cj          # connect CJ86GJI4_5G     (needs admin)
    python wifi_switch.py omap        # connect OMAP-Motion-5G  (needs admin)
    python wifi_switch.py toggle      # CJ86GJI4_5G <-> OMAP-Motion-5G (needs admin)
"""

import argparse
import subprocess
import sys
import time

ADAPTER = "Wi-Fi"  # Intel AX211 built-in (the only Wi-Fi adapter on this box)

# friendly key -> exact Wi-Fi profile / SSID name
NETWORKS = {
    "dit": "DIT_MB",
    "cj": "CJ86GJI4_5G",
    "omap": "OMAP-Motion-5G",
}

# Windows NLA "network name" (Get-NetConnectionProfile.Name) -> true SSID.
# Get-NetConnectionProfile works without elevation/Location but returns the NLA name
# (e.g. a domain network shows as "corpnet.asus", not the SSID). Map the ones we know;
# unknown names are returned verbatim so `status` is still informative.
NLA_TO_SSID = {
    "corpnet.asus": "CJ86GJI4_5G",
}


def _run(args):
    # errors="replace": netsh/powershell on zh-TW Windows may emit cp950 bytes that would
    # otherwise crash the default decoder.
    return subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def _netsh_ssid():
    """True SSID via netsh — only works if Location access AND elevation are available;
    returns '' otherwise (the common case on this box)."""
    r = _run(["netsh", "wlan", "show", "interfaces"])
    blob = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "error" in blob.lower() or "location" in blob.lower():
        return ""
    name = None
    for line in r.stdout.splitlines():
        s = line.strip()
        if ":" not in s:
            continue
        key, _, val = s.partition(":")
        key, val = key.strip(), val.strip()
        if key in ("Name", "名稱"):  # interface name line (English / zh-TW)
            name = val
        elif key == "SSID" and name == ADAPTER and val:  # not "BSSID" (starts with B)
            return val
    return ""


def _nla_name():
    """NLA network name on ADAPTER via Get-NetConnectionProfile (no admin/Location needed)."""
    r = _run(
        [
            "powershell", "-NoProfile", "-Command",
            f"(Get-NetConnectionProfile -InterfaceAlias '{ADAPTER}' -ErrorAction SilentlyContinue).Name",
        ]
    )
    return (r.stdout or "").strip()


def current_ssid():
    """Best-effort current SSID/network on ADAPTER WITHOUT requiring admin or Location.
    Prefers the true SSID (netsh, if available); else maps the NLA name to an SSID;
    else returns the raw NLA name. Returns '' only if genuinely not connected."""
    ssid = _netsh_ssid()
    if ssid:
        return ssid
    nla = _nla_name()
    if not nla:
        return ""
    return NLA_TO_SSID.get(nla, nla)


def connect(ssid, timeout=15):
    """Connect ADAPTER to ssid; poll until connected or timeout. Returns bool.
    netsh wlan connect needs admin elevation on this box; on "error 5" we report clearly."""
    r = _run(["netsh", "wlan", "connect", f"name={ssid}", f"interface={ADAPTER}"])
    blob = ((r.stdout or "") + (r.stderr or "")).strip()
    if r.returncode != 0 or "error 5" in blob.lower() or "elevation" in blob.lower():
        print("  cannot switch automatically: `netsh wlan connect` needs ADMIN elevation on this box.")
        print(f"  -> switch manually to '{ssid}' (Wi-Fi flyout), or re-run this script as Administrator.")
        if blob:
            print(f"  netsh said: {blob.splitlines()[0]}")
        return False
    deadline = time.time() + timeout
    while time.time() < deadline:
        if current_ssid() == ssid:
            return True
        time.sleep(1)
    return current_ssid() == ssid


def main():
    """CLI entry point: parse arguments, then show or switch the Wi-Fi network."""
    ap = argparse.ArgumentParser(
        description="Inspect/switch Wi-Fi on the AX211 (status needs no admin; switching needs admin)."
    )
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
        sys.exit(1)


if __name__ == "__main__":
    main()
