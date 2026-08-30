#!/usr/bin/env python3
"""Poll the Pegasus Phase-0 LIBERO supervisor and mirror its state into a local status.json for
the eval dashboard (a separate view from the training monitor's index.html in the same output
directory -- this one tracks eval_supervisor.sh's sweep, not a training run).

One long-lived Pegasus session is reused for every poll (repeated logins can trip a
temporary 401 lockout). Transient failures keep the last good payload and surface a
`link` field so the dashboard can show connection state instead of going blank.

Promoted from tmp/dashboard_poller.py (openspec task 2.5) -- the fix that task asked for is
here: the output path resolves against the repo root (found by walking up for workspace.yaml,
same mechanism agents/evalbot/harness/pegasus_launch.py uses) instead of a path relative to
this file's own location, so it no longer matters what cwd this is launched from.
"""
import json, pathlib, socket, sys, time, datetime, traceback

# Bound EVERY blocking socket op. pegasus.login() and the kernel websocket handshake
# have no explicit timeouts, so a wedged server silently hangs the poller forever and
# the dashboard just goes stale with no error shown.
socket.setdefaulttimeout(180)

HERE = pathlib.Path(__file__).resolve().parent


def _root() -> pathlib.Path:
    for d in (HERE, *HERE.parents):
        if (d / "workspace.yaml").is_file():
            return d
    return HERE.parents[3]   # agents/rldx-trainbot/harness/monitor -> repo root, if no marker


ROOT = _root()
sys.path.insert(0, str(ROOT / "agents" / "tools" / "common"))
import pegasus  # noqa: E402

# Pegasus-side state: same absolute root eval_supervisor.sh itself uses (WS in that script).
# Not resolved via workspace.yaml -- that mechanism is for THIS repo's local checkout layout,
# and has no equivalent meaning on the remote box's own filesystem.
PEGASUS_WS = "/data/VLA/tingying"
STATE = f"{PEGASUS_WS}/pegasus_runs/phase0"
# NOTE: this is a remote copy uploaded by hand during the original Phase-0 run, kept separate
# from the promoted source of truth at agents/rldx-trainbot/harness/tally.py (task 2.4). There
# is no deploy script for this harness yet (unlike agents/evalbot/harness/pegasus_deploy.py) --
# if tally.py changes, re-upload it to this path too, or this poller silently drifts from it.
TALLY_PY = f"{PEGASUS_WS}/pegasus_runs/tally_phase0.py"

OUT = ROOT / "tmp" / "dashboard" / "status.json"
INTERVAL = 20

# One shell round-trip fetches everything the UI needs.
PROBE = (
    f"cat {STATE}/status.json 2>/dev/null; "
    f"echo '<<<SPLIT>>>'; "
    f"tail -25 {STATE}/supervisor.log 2>/dev/null; "
    f"echo '<<<SPLIT>>>'; "
    f"tail -6 {STATE}/server.log 2>/dev/null; "
    f"echo '<<<SPLIT>>>'; "
    f"[ -f {STATE}/supervisor.finished ] && echo FINISHED || echo RUNNING; "
    f"echo '<<<SPLIT>>>'; "
    f"ls -1 {STATE}/markers/*.fail 2>/dev/null | head -20; "
    f"echo '<<<SPLIT>>>'; "
    # Authoritative per-episode tally from simulation_results.csv. The supervisor's own
    # status.json counts video filenames, which overshoot the episode target and are not
    # deduped -- good enough for a progress bar, wrong for the gate number.
    f"python3 {TALLY_PY} 2>/dev/null"
)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s = pegasus.connect()
    last = {}
    consecutive_errors = 0

    while True:
        payload = dict(last)
        try:
            raw, rc = pegasus.sh(s, PROBE, timeout=120)
            parts = raw.split("<<<SPLIT>>>")
            status = {}
            if parts and parts[0].strip():
                try:
                    status = json.loads(parts[0].strip())
                except json.JSONDecodeError:
                    status = last.get("status", {})
            tally = {}
            if len(parts) > 5 and parts[5].strip():
                try:
                    tally = json.loads(parts[5].strip())
                except json.JSONDecodeError:
                    tally = last.get("tally", {})
            payload = {
                "status": status,
                "tally": tally,
                "supervisor_log": (parts[1].strip().splitlines() if len(parts) > 1 else []),
                "server_log": (parts[2].strip().splitlines() if len(parts) > 2 else []),
                "finished": ("FINISHED" in parts[3] if len(parts) > 3 else False),
                "failed_markers": (
                    [p.split("/")[-1] for p in parts[4].strip().splitlines() if p.strip()]
                    if len(parts) > 4 else []
                ),
                "link": {"ok": True, "error": None},
            }
            consecutive_errors = 0
            last = payload
        except Exception as e:  # noqa: BLE001 - keep the poller alive across any transport fault
            consecutive_errors += 1
            payload = dict(last)
            payload["link"] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                               "consecutive": consecutive_errors}
            # Session may have expired -> rebuild it.
            if consecutive_errors in (2, 6, 12):
                try:
                    s = pegasus.connect()
                    payload["link"]["error"] += " (reconnected)"
                except Exception:
                    traceback.print_exc()

        payload["polled_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        tmp = OUT.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(OUT)
        print(f"[{payload['polled_at']}] link={payload['link']['ok']} "
              f"phase={payload.get('status', {}).get('phase')} "
              f"done={payload.get('status', {}).get('tasks', {}).get('done')}", flush=True)

        if payload.get("finished") and payload.get("status", {}).get("phase") in ("done", "partial", "failed"):
            print("supervisor finished; poller exiting", flush=True)
            break
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
