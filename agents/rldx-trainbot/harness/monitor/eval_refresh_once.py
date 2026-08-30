#!/usr/bin/env python3
"""One-shot refresh of the eval dashboard's status.json -- for when the Phase-0 sweep has
already finished and there's no point starting the polling loop (eval_poller.py) just to
fetch one snapshot.

Promoted from tmp/refresh_once.py (openspec task 2.5) -- same ROOT/workspace.yaml resolution
fix as eval_poller.py, so the output path no longer depends on this file's own location.
"""
import json, pathlib, socket, sys, datetime

socket.setdefaulttimeout(180)

HERE = pathlib.Path(__file__).resolve().parent


def _root() -> pathlib.Path:
    for d in (HERE, *HERE.parents):
        if (d / "workspace.yaml").is_file():
            return d
    return HERE.parents[3]


ROOT = _root()
sys.path.insert(0, str(ROOT / "agents" / "tools" / "common"))
import pegasus  # noqa: E402

PEGASUS_WS = "/data/VLA/tingying"
STATE = f"{PEGASUS_WS}/pegasus_runs/phase0"
TALLY_PY = f"{PEGASUS_WS}/pegasus_runs/tally_phase0.py"   # see eval_poller.py's NOTE on this path

PROBE = (
    f"cat {STATE}/status.json 2>/dev/null; echo '<<<SPLIT>>>'; "
    f"tail -25 {STATE}/supervisor.log 2>/dev/null; echo '<<<SPLIT>>>'; "
    f"tail -6 {STATE}/server.log 2>/dev/null; echo '<<<SPLIT>>>'; "
    f"[ -f {STATE}/supervisor.finished ] && echo FINISHED || echo RUNNING; echo '<<<SPLIT>>>'; "
    f"ls -1 {STATE}/markers/*.fail 2>/dev/null | head -20; echo '<<<SPLIT>>>'; "
    f"python3 {TALLY_PY} 2>/dev/null"
)


def jload(t):
    try:
        return json.loads(t.strip())
    except Exception:
        return {}


def main() -> int:
    s = pegasus.connect()
    raw, _ = pegasus.sh(s, PROBE, timeout=180)
    parts = raw.split("<<<SPLIT>>>")

    payload = {
        "status": jload(parts[0]),
        "tally": jload(parts[5]) if len(parts) > 5 else {},
        "supervisor_log": parts[1].strip().splitlines() if len(parts) > 1 else [],
        "server_log": parts[2].strip().splitlines() if len(parts) > 2 else [],
        "finished": "FINISHED" in parts[3] if len(parts) > 3 else False,
        "failed_markers": [p.split("/")[-1] for p in parts[4].strip().splitlines() if p.strip()]
        if len(parts) > 4 else [],
        "link": {"ok": True, "error": None},
        "polled_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    out = ROOT / "tmp" / "dashboard" / "status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    o = payload["tally"].get("overall", {})
    print("finished:", payload["finished"], "| phase:", payload["status"].get("phase"))
    print("suite_mean:", o.get("suite_mean"), "| gate:", o.get("gate"),
          "| final:", o.get("suite_mean_is_final"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
