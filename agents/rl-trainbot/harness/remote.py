"""Thin remote-exec layer for the RL Phase-1 tooling, built on agents/tools/common/pegasus.py.

Gives launch.py / poll.py / report.py a small, shared surface for talking to the
Pegasus Jupyter host without each re-implementing the kernel/websocket plumbing:

    connect()                       -> authenticated session (uses PEGASUS_PASSWORD)
    sh(s, cmd)                      -> (stdout, returncode), blocking, captured
    run_detached(s, cmd, log, pid)  -> start cmd under setsid+nohup, return its PID
    pid_alive(s, pid)               -> bool
    tail(s, path, n)                -> last n lines of a remote file
    fetch(s, remote, local)         -> download a remote file/dir

`run_detached` uses `setsid nohup ... </dev/null &` so the training process lands in
its own session: deleting the transient Jupyter kernel (which pegasus.sh does on every
call) cannot SIGHUP/SIGTERM it, and a dropped control connection cannot kill it.
"""
from __future__ import annotations

import pathlib
import shlex
import sys

# Resolve shared tools through the workspace.yaml root marker, so this keeps working
# regardless of the caller's cwd *and* survives directory moves (see agents/tools/
# workspace_paths.py). ROOT is re-exported for consumers that need repo-relative paths.
def _ws_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        if (d / "workspace.yaml").is_file():
            return d
    raise FileNotFoundError("workspace.yaml not found walking up from " + str(here))


ROOT = _ws_root()
if str(ROOT / "agents" / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "agents" / "tools"))

from workspace_paths import ws_path  # noqa: E402

_COMMON = str(ws_path("common_tools"))
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

import pegasus  # noqa: E402


def connect():
    """Authenticated Pegasus session (reads PEGASUS_PASSWORD or prompts)."""
    return pegasus.connect()


def sh(s, cmd, timeout=300):
    """Run a bash command on Pegasus, return (stdout_text, returncode)."""
    return pegasus.sh(s, cmd, timeout=timeout)


def run_detached(s, cmd, log_path, pidfile, cwd=None, env=None, timeout=120):
    """Start `cmd` fully detached on Pegasus; return (launch_rc, pid_str_or_None).

    stdout+stderr -> log_path, PID -> pidfile. The process is put in its own
    session (setsid) and ignores SIGHUP (nohup), so it outlives both the Jupyter
    kernel and the control connection.
    """
    parts = []
    if cwd:
        parts.append(f"cd {shlex.quote(cwd)}")
    parts.append(f'mkdir -p "$(dirname {shlex.quote(log_path)})"')
    parts.append(f'mkdir -p "$(dirname {shlex.quote(pidfile)})"')

    env_prefix = ""
    if env:
        env_prefix = "env " + " ".join(
            f"{k}={shlex.quote(str(v))}" for k, v in env.items()
        ) + " "

    # setsid -> new session; nohup -> ignore HUP; </dev/null -> no stdin tie.
    # The background job is wrapped in its own ( ... ) subshell so the trailing `&`
    # unambiguously scopes to just this job (bash's `&` binds looser than `&&`, so an
    # unparenthesized "cd X && ... && setsid ... &" backgrounds the WHOLE cd/mkdir/setsid
    # chain, not just setsid -- $! then names that wrapper, which is fine 99% of the time
    # but leaves no safety net if the final echo/cat blips). pgrep on the (unique) log
    # path is a fallback PID source if the pidfile write itself ever fails transiently.
    launch = (
        f"( setsid nohup {env_prefix}bash -lc {shlex.quote(cmd)} "
        f"> {shlex.quote(log_path)} 2>&1 < /dev/null & echo $! > {shlex.quote(pidfile)} ); "
        f"sleep 1; cat {shlex.quote(pidfile)} 2>/dev/null "
        f"|| pgrep -f {shlex.quote(log_path)} | head -1"
    )
    parts.append(launch)
    full = " && ".join(parts)

    out, rc = pegasus.sh(s, full, timeout=timeout)
    pid = None
    for line in reversed(out.strip().splitlines()):
        line = line.strip()
        if line.isdigit():
            pid = line
            break
    if pid is None:
        # Last-resort fallback: the job may have launched fine even if every local
        # PID-capture step above blipped (transient FS/websocket hiccup).
        out2, _ = pegasus.sh(s, f"pgrep -f {shlex.quote(log_path)} | head -1", timeout=30)
        out2 = out2.strip()
        pid = out2 if out2.isdigit() else None
    return rc, pid


def pid_alive(s, pid, timeout=60):
    """True if `pid` is still running on Pegasus."""
    if not pid:
        return False
    out, _ = pegasus.sh(s, f"kill -0 {int(pid)} 2>/dev/null && echo ALIVE || echo DEAD",
                        timeout=timeout)
    return "ALIVE" in out


def tail(s, path, n=40, timeout=120):
    """Return the last `n` lines of a remote text file (empty string if absent)."""
    out, _ = pegasus.sh(s, f"tail -n {int(n)} {shlex.quote(path)} 2>/dev/null", timeout=timeout)
    return out


def exists(s, path, timeout=60):
    """True if a remote path exists."""
    out, _ = pegasus.sh(s, f"test -e {shlex.quote(path)} && echo YES || echo NO", timeout=timeout)
    return "YES" in out


def fetch(s, remote_path, local_path):
    """Download a remote file or directory to local (resumable + verified)."""
    pegasus.get(s, remote_path, str(local_path))
