#!/usr/bin/env python3
"""Read-only live web dashboard for the automation pipeline.

Serves a single auto-refreshing page that polls ``/api/status`` (built from
``pipeline_state.json`` + ``logs/progress.json`` + ``logs/metrics.jsonl``) and shows the
stage timeline, the active phase's %/rate/ETA, a live loss chart, and run outputs.

    python dashboard.py                 # http://localhost:8770  (and LAN IP for phones)
    python dashboard.py --port 9000 --host 127.0.0.1

Read-only: it never writes pipeline state and never exposes secrets (the config password
is not part of the status payload).
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pipeline_config as pc
import pipeline_progress as pp
import report_generator as rg
from pipeline_state import ORDER
from pipeline_paths import STATE_PATH, METRICS_PATH, CHARTJS_PATH, DASHBOARD_HTML

_MAX_POINTS = 400


# --------------------------------------------------------------------------- #
#  Status assembly (pure — unit-tested)
# --------------------------------------------------------------------------- #
def _downsample(seq: list, limit: int = _MAX_POINTS) -> list:
    if len(seq) <= limit:
        return seq
    k = len(seq) // limit + 1
    return seq[::k] + [seq[-1]]


def _series(metrics: list, key: str) -> dict:
    """{steps, vals, segs} for one metric, keeping only records that have it (downsampled).

    ``segs`` is the per-point resume-segment index (0 = first launch, 1+ = after each
    ``--resume`` relaunch) so the dashboard can color the curve at resume boundaries.
    """
    pts = [(m.get("step"), m[key], m.get("seg", 0)) for m in metrics if m.get(key) is not None]
    return {"steps": _downsample([s for s, _, _ in pts]),
            "vals": _downsample([v for _, v, _ in pts]),
            "segs": _downsample([g for _, _, g in pts])}


def build_status(state_dict: dict, progress: dict, metrics: list,
                 partial_bytes: int | None = None, wifi: str | None = None) -> dict:
    """Assemble the JSON the dashboard polls (pure)."""
    data = state_dict.get("data", {})
    stages = state_dict.get("stages", {})
    current = state_dict.get("current", "IDLE")
    stage_list = [{"name": s.value, **stages.get(s.value, {"status": "pending"})}
                  for s in ORDER]
    if current == "DONE":                        # terminal stage has no record of its own
        for s in stage_list:
            if s["name"] == "DONE":
                s["status"] = "done"
    train_loss = _series(metrics, "loss")
    return {
        "current": current,
        "profile": state_dict.get("profile"),
        "stages": stage_list,
        "data": data,
        "progress": progress,
        "loss": train_loss,                       # back-compat: training-loss curve
        "series": {                               # per-metric series for the dashboard plots
            "train_loss": train_loss,
            "eval_loss": _series(metrics, "eval_loss"),
            "lr": _series(metrics, "lr"),
            "grad_norm": _series(metrics, "grad_norm"),
        },
        "partial_download_bytes": partial_bytes,
        "wifi": wifi,
        "updated_at": progress.get("updated_at"),
    }


# --------------------------------------------------------------------------- #
#  Live data gathering
# --------------------------------------------------------------------------- #
_metrics_cache: dict = {"key": None, "data": []}


def _cached_metrics() -> list:
    """Parse metrics.jsonl via report_generator.load_metrics, cached by (mtime, size).

    The dashboard is polled ~every 1.5s by every client; re-parsing the whole (growing)
    file each time is the one cost that scales with run length, so we only re-parse when
    the file actually changes.
    """
    try:
        st = METRICS_PATH.stat()
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        return []
    if key != _metrics_cache["key"]:
        _metrics_cache["key"] = key
        _metrics_cache["data"] = rg.load_metrics(METRICS_PATH)
    return _metrics_cache["data"]


class _WifiCache:
    """Cache the SSID (netsh is slow) and refresh at most every 10s."""
    def __init__(self):
        self.value = None
        self.ts = 0.0

    def get(self):
        if time.time() - self.ts > 10:
            try:
                import wifi_switch
                self.value = wifi_switch.current_ssid() or None
            except Exception:                    # noqa: BLE001
                self.value = None
            self.ts = time.time()
        return self.value


def _status_payload(config: dict, wifi_cache: _WifiCache) -> dict:
    state_dict = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}
    progress = pp.read()
    metrics = _cached_metrics()
    # Live partial-download size (works even for an in-flight run that predates progress.json).
    partial = None
    try:
        if state_dict.get("current") == "DOWNLOADING":
            zip_name = state_dict.get("data", {}).get("zip_name")
            if zip_name:
                f = Path(config["local"]["download_dir"]) / zip_name
                if f.exists():
                    partial = f.stat().st_size
    except Exception:                            # noqa: BLE001
        partial = None
    return build_status(state_dict, progress, metrics, partial, wifi_cache.get())


# --------------------------------------------------------------------------- #
#  HTTP server
# --------------------------------------------------------------------------- #
def make_handler(config: dict):
    wifi_cache = _WifiCache()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):              # silence per-request console spam
            pass

        def _send(self, body: bytes, ctype: str, cache: bool = False):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if cache:
                self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                if self.path.startswith("/api/status"):
                    payload = json.dumps(_status_payload(config, wifi_cache)).encode("utf-8")
                    self._send(payload, "application/json")
                elif self.path.startswith("/vendor/chart"):
                    self._send(CHARTJS_PATH.read_bytes(), "application/javascript", cache=True)
                elif self.path in ("/", "/index.html"):
                    self._send(DASHBOARD_HTML.read_bytes(), "text/html; charset=utf-8")
                else:
                    self.send_error(404)
            except BrokenPipeError:
                pass
            except Exception as e:               # noqa: BLE001 - never crash the server
                self.send_error(500, str(e))

        def do_POST(self):
            try:
                if self.path.startswith("/api/stop"):
                    import training_monitor as tm
                    from pipeline_state import PipelineState
                    state = PipelineState.load()
                    msg = tm.stop_run(config, state)
                    state.stop(state.current)    # mark state so --status is honest (matches --stop)
                    self._send(json.dumps({"ok": True, "message": msg}).encode("utf-8"),
                               "application/json")
                else:
                    self.send_error(404)
            except Exception as e:               # noqa: BLE001
                self._send(json.dumps({"ok": False, "message": str(e)}).encode("utf-8"),
                           "application/json")

    return Handler


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:                            # noqa: BLE001 - offline (e.g. on LAN)
        return "127.0.0.1"
    finally:
        s.close()


def serve(host: str = "0.0.0.0", port: int = 8770, config_path: str | None = None) -> None:
    """Start the read-only dashboard server (reused by the CLI and gr00t_pipeline.py)."""
    config = pc.load_config(config_path)
    httpd = ThreadingHTTPServer((host, port), make_handler(config))
    print(f"Dashboard:  http://localhost:{port}")
    if host == "0.0.0.0":
        print(f"On LAN/phone:  http://{_lan_ip()}:{port}")
    print("Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0", help="bind host (default 0.0.0.0 for LAN)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--config", help="path to config.yaml")
    args = ap.parse_args()
    serve(args.host, args.port, args.config)


if __name__ == "__main__":
    main()
