"""Deploy the verified checkpoint to the asus-4090 sim box over SSH (paramiko).

DEPLOYING stage sequence (each step must succeed; transient failures retry up to
``retry_max``):
  1. switch Wi-Fi to the LAN network (OMAP-Motion-5G) that reaches asus-4090,
  2. wait until the box answers a ping,
  3. SSH in (password auth), ensure the deploy dir exists,
  4. SFTP-upload the run zip,
  5. remotely ``unzip -o`` it into the deploy dir,
  6. verify the unzipped run directory exists and is non-empty.

asus-4090 is reached by SSH (unlike Pegasus), so we use paramiko here — password auth,
no sshpass needed, works the same on Windows and Linux.
"""
from __future__ import annotations

import posixpath
import shlex
import time
from pathlib import Path

import paramiko

import net_util
import pipeline_config as pc
import pipeline_progress as pp
from pipeline_retry import retry
from pipeline_logging import get_logger

log = get_logger("deploy")


def _default_client() -> "paramiko.SSHClient":
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    return c


def connect_ssh(host: str, user: str, password: str, retry_max: int = 3,
                interval: int = 10, connect_timeout: int = 20, client_factory=None):
    """Open an SSH connection, retrying ``retry_max`` times. ``client_factory`` is
    injectable for tests."""
    factory = client_factory or _default_client
    holder = {}

    def _connect():
        client = holder["c"] = factory()
        client.connect(hostname=host, username=user, password=password,
                       timeout=connect_timeout, look_for_keys=False, allow_agent=False)
        log.info("SSH connected to %s@%s.", user, host)
        return client

    def _close_failed(_attempt, _exc):
        try:
            holder["c"].close()
        except Exception:                        # noqa: BLE001
            pass

    return retry(_connect, attempts=retry_max, interval=interval,
                 label=f"SSH connect to {host}", on_error=_close_failed)


def run_ssh(ssh, cmd: str, timeout: int = 1800):
    """Run a remote command; return (combined_output, exit_code)."""
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    rc = stdout.channel.recv_exit_status()
    out = (stdout.read().decode("utf-8", "replace")
           + stderr.read().decode("utf-8", "replace"))
    return out, rc


def sftp_upload(ssh, local: str, remote: str, retry_max: int = 3, interval: int = 10):
    """Upload one file via SFTP, retrying on transient failure."""
    size = Path(local).stat().st_size

    def _put():
        sftp = ssh.open_sftp()
        try:
            start = time.time()
            t_log = [start]
            t_prog = [start]

            def _cb(done, total, _tl=t_log, _tp=t_prog):
                now = time.time()
                if now - _tp[0] >= 1.5:            # progress.json for the dashboard
                    _tp[0] = now
                    rate = done / max(now - start, 1e-3)
                    pp.update("deploy", done=done, total=total,
                              pct=round(100 * done / max(total, 1), 1),
                              rate_mbps=round(rate / 1e6, 1),
                              eta_sec=(round((total - done) / rate) if rate > 0 else None))
                if now - _tl[0] >= 5:              # throttle log to ~every 5s
                    _tl[0] = now
                    log.info("  upload %.1f/%.1f MB (%.0f%%)",
                             done / 1e6, total / 1e6, 100 * done / max(total, 1))

            sftp.put(local, remote, callback=_cb)
        finally:
            sftp.close()
        pp.update("deploy", done=size, total=size, pct=100, rate_mbps=0, eta_sec=0)
        log.info("Uploaded %s -> %s (%d bytes).", local, remote, size)

    retry(_put, attempts=retry_max, interval=interval, label="SFTP upload")


def run(config: dict, state) -> str:
    """DEPLOYING stage entry point. Returns the remote unzipped directory path."""
    a = config["asus4090"]
    host, user = a["host"], a["user"]
    password = pc.get_secret("ASUS4090_PASSWORD", a.get("password"))
    if not password:
        raise SystemExit("no asus-4090 password (set ASUS4090_PASSWORD or config.yaml)")
    deploy_dir = str(a["deploy_dir"]).rstrip("/")
    retry_max = int(a.get("retry_max", 3))

    zip_local = state.get("zip_local")
    if not zip_local or not Path(zip_local).exists():
        raise SystemExit("no downloaded zip in state; DOWNLOADING must run first")
    zip_name = Path(zip_local).name
    run_name = Path(zip_name).stem

    # 1. switch to the LAN that reaches asus-4090
    net_util.switch_wifi(config["wifi"]["local"])
    # 2. wait for the box
    if not net_util.wait_for_host(host, int(config["wifi"]["net_ready_timeout_sec"])):
        raise RuntimeError(f"{host} not reachable after Wi-Fi switch")

    # 3. ssh in
    ssh = connect_ssh(host, user, password,
                      retry_max=int(config["training"]["connect_retry_max"]),
                      interval=int(config["training"]["connect_retry_interval_sec"]))
    try:
        out, rc = run_ssh(ssh, f"mkdir -p {shlex.quote(deploy_dir)}")
        if rc != 0:
            raise RuntimeError(f"mkdir {deploy_dir} failed (rc={rc}): {out.strip()}")

        # 4. upload
        remote_zip = posixpath.join(deploy_dir, zip_name)
        sftp_upload(ssh, zip_local, remote_zip, retry_max=retry_max)

        # 5. unzip
        out, rc = run_ssh(ssh, f"cd {shlex.quote(deploy_dir)} && "
                               f"unzip -o {shlex.quote(zip_name)}")
        if rc != 0:
            raise RuntimeError(f"remote unzip failed (rc={rc}): {out.strip()}")

        # 6. verify non-empty
        unzipped = posixpath.join(deploy_dir, run_name)
        out, rc = run_ssh(ssh, f'test -d {shlex.quote(unzipped)} && '
                               f'[ -n "$(ls -A {shlex.quote(unzipped)})" ] && echo OK')
        if rc != 0 or "OK" not in out:
            raise RuntimeError(f"deploy verify failed: {unzipped} missing or empty")

        state.record_output("deploy_remote", unzipped)
        state.record_output("deploy_host", f"{user}@{host}")
        log.info("Deployed checkpoint to %s@%s:%s", user, host, unzipped)
        return unzipped
    finally:
        ssh.close()
