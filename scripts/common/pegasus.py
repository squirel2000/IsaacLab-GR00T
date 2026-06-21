#!/usr/bin/env python3
"""Automation client for the Pegasus Jupyter server (service-cpu-bdml.asus.com/pegasus).

Library + CLI for upload / download / remote-exec against the password-protected
Jupyter server, so the web UI is no longer the only way to move data or run scripts.

Auth:   set the password in the environment (never hard-code it):
            PowerShell:  $env:PEGASUS_PASSWORD = 'eksncl#...'
        otherwise it is prompted interactively.

TLS:    the server uses an internal "ASUSTEK 2016 Service CA1" certificate that lives
        in the Windows trust store.  This client uses `truststore` to validate against
        that store, so verified TLS works without --insecure.   pip install truststore

Paths:  the Jupyter root maps to the server's /data, so a server file
        /data/VLA/tingying/foo.zip is addressed here as "VLA/tingying/foo.zip".
        Helpers accept either form (a leading "/data/" is stripped automatically).

CLI:
    python pegasus.py ls   [REMOTE_DIR]
    python pegasus.py put  LOCAL REMOTE          # file or directory (recursive)
    python pegasus.py get  REMOTE LOCAL          # file or directory (resumable + verified)
    python pegasus.py rm   REMOTE                # recursive
    python pegasus.py mkdir REMOTE_DIR
    python pegasus.py run  "shell command ..."   # streams stdout/stderr live
    python pegasus.py run  --file local_script.sh
"""
import argparse, base64, hashlib, json, os, ssl, sys, time, uuid, pathlib

BASE = os.environ.get("PEGASUS_BASE", "https://service-cpu-bdml.asus.com/pegasus")
PEGASUS_PASSWORD = os.environ.get("PEGASUS_PASSWORD", "eksncl#20260410_PE")

# --------------------------------------------------------------------------- #
#  Session / auth
# --------------------------------------------------------------------------- #
def make_session(insecure=False):
    """Create a requests.Session with TLS validated via the OS trust store
    (truststore); falls back to unverified TLS only if truststore is missing."""
    if not insecure:
        try:
            import truststore
            truststore.inject_into_ssl()          # validate via OS trust store (has ASUS CA)
        except ImportError:
            sys.stderr.write("warning: `truststore` not installed; using --insecure.\n"
                             "         pip install truststore   (for verified TLS)\n")
            insecure = True
    import requests, urllib3
    s = requests.Session()
    if insecure:
        urllib3.disable_warnings()
        s.verify = False
    s._insecure = insecure
    s._password = None
    return s


def login(s, password):
    """Two-step Jupyter login: GET to obtain the _xsrf cookie, then POST the password."""
    s.get(f"{BASE}/login").raise_for_status()
    r = s.post(f"{BASE}/login",
               data={"_xsrf": s.cookies.get("_xsrf"), "password": password},
               allow_redirects=False)            # keep the 302 so we can check it
    if r.status_code not in (302, 200) or s.cookies.get("username-service-cpu-bdml-asus-com") is None:
        raise SystemExit("login failed: check PEGASUS_PASSWORD")
    s._password = password                        # cached so relogin() can re-auth later
    return s


def connect(insecure=False, password=None):
    """One-shot: build a session and log in using PEGASUS_PASSWORD (or prompt)."""
    import getpass
    pw = password or PEGASUS_PASSWORD or getpass.getpass("Pegasus password: ")
    return login(make_session(insecure), pw)


def relogin(s):
    """Re-authenticate with the cached password after the session cookie expires."""
    if not s._password:
        raise SystemExit("session expired and no cached password to re-login")
    return login(s, s._password)


def _xsrf(s):
    """Header dict carrying the XSRF token, required on every PUT/POST/DELETE."""
    return {"X-XSRFToken": s.cookies.get("_xsrf")}


def relpath(p):
    """Normalise a server path to one relative to the Jupyter root (/data)."""
    p = str(p).replace("\\", "/").lstrip("/")
    if p.startswith("data/"):
        p = p[len("data/"):]
    return p


def files_url(rel):
    """Build the raw /files download URL (Range-capable) for a server path."""
    return f"{BASE}/files/{relpath(rel)}"


# --------------------------------------------------------------------------- #
#  Contents API:  ls / mkdir / rm / upload
# --------------------------------------------------------------------------- #
def ls(s, path=""):
    """Print a directory listing (directories first) for a server path."""
    r = s.get(f"{BASE}/api/contents/{relpath(path)}"); r.raise_for_status()
    for c in sorted(r.json()["content"], key=lambda x: (x["type"] != "directory", x["name"])):
        size = "" if c["size"] is None else f"{c['size']:>14,}"
        print(f"{'d' if c['type']=='directory' else '-'} {size}  {c['path']}")


def mkdir(s, path):
    """Create a single directory on the server."""
    r = s.put(f"{BASE}/api/contents/{relpath(path)}", headers=_xsrf(s),
              json={"type": "directory"}); r.raise_for_status()
    print("mkdir", r.json()["path"])


def rm(s, path, _verbose=True):
    """Delete a server file or directory, recursing first (the API refuses non-empty dirs)."""
    path = relpath(path)
    info = s.get(f"{BASE}/api/contents/{path}")
    if info.status_code == 200 and info.json()["type"] == "directory":
        for c in info.json()["content"]:                 # API refuses non-empty dirs
            if c["type"] == "directory":
                rm(s, c["path"], _verbose=False)
            else:
                s.delete(f"{BASE}/api/contents/{relpath(c['path'])}", headers=_xsrf(s)).raise_for_status()
    r = s.delete(f"{BASE}/api/contents/{path}", headers=_xsrf(s))
    if r.status_code not in (204, 200):
        r.raise_for_status()
    if _verbose:
        print("removed", path)


def _ensure_remote_dir(s, path):
    """Create every missing parent directory along a server path."""
    path = relpath(path)
    cur = ""
    for part in [p for p in path.split("/") if p]:
        cur = f"{cur}/{part}".lstrip("/")
        s.put(f"{BASE}/api/contents/{cur}", headers=_xsrf(s), json={"type": "directory"})


def put_file(s, local, remote):
    """Upload one local file to the server (text, or base64 if it is binary)."""
    remote = relpath(remote)
    data = pathlib.Path(local).read_bytes()
    try:
        body = {"type": "file", "format": "text", "content": data.decode("utf-8")}
    except UnicodeDecodeError:                            # not UTF-8 -> send as base64
        body = {"type": "file", "format": "base64",
                "content": base64.b64encode(data).decode("ascii")}
    _ensure_remote_dir(s, "/".join(remote.split("/")[:-1]))
    s.put(f"{BASE}/api/contents/{remote}", headers=_xsrf(s), json=body).raise_for_status()
    print(f"put  {local}  ->  {remote}  ({len(data):,} B)")


def put(s, local, remote):
    """Upload a single file, or recursively upload a whole local directory."""
    p = pathlib.Path(local)
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file():
                put_file(s, str(f), f"{relpath(remote).rstrip('/')}/{f.relative_to(p).as_posix()}")
    else:
        put_file(s, local, remote)


# --------------------------------------------------------------------------- #
#  Resumable, verified download via the raw /files endpoint (HTTP Range)
# --------------------------------------------------------------------------- #
def remote_size(s, rel, _max_tries=6):
    """Return a server file's total byte size, probed with a Range 0-0 request.
    Retries transient network errors (e.g. ReadTimeout) with backoff — otherwise a single
    server hiccup on one file's probe would abort a whole recursive directory download."""
    import requests

    rel = relpath(rel)
    for attempt in range(1, _max_tries + 1):
        try:
            r = s.get(files_url(rel), params={"_xsrf": s.cookies.get("_xsrf")},
                      headers={"Range": "bytes=0-0"}, stream=True, timeout=(30, 60))
            if r.status_code in (401, 403):
                r.close(); relogin(s); continue
            if r.status_code == 206:
                total = int(r.headers["Content-Range"].split("/")[-1])   # "bytes 0-0/TOTAL"
            elif r.status_code == 200:
                total = int(r.headers.get("Content-Length", 0))
            else:
                r.close(); r.raise_for_status()
            r.close()
            return total
        except (requests.exceptions.RequestException, ssl.SSLError, OSError) as e:
            if attempt >= _max_tries:
                raise
            wait = min(30, 2 ** attempt)
            sys.stdout.write(f"\n  ! size probe interrupted ({type(e).__name__}); "
                             f"retry in {wait}s (try {attempt})\n")
            sys.stdout.flush()
            time.sleep(wait)


def sha256_file(path, chunk=1 << 20):
    """Compute the SHA-256 of a local file, reading it in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def download_resumable(s, remote, local, expected_sha256=None,
                       chunk=1 << 20, progress=True):
    """Download remote (server path) -> local, resuming on disconnect, then verify
    size and (optionally) sha256.  Survives broken connections."""
    import requests
    rel = relpath(remote)
    local = pathlib.Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(s, rel)
    # A leftover/corrupt local file LARGER than the remote can never match by appending
    # (append-only resume would just keep growing it) — discard it and re-download clean.
    if local.exists() and local.stat().st_size > total:
        sys.stdout.write(f"  discarding oversized local partial "
                         f"({local.stat().st_size:,} > remote {total:,}); re-downloading\n")
        local.unlink()
    attempt = 0
    t0 = time.time()
    while True:
        pos = local.stat().st_size if local.exists() else 0   # bytes already on disk
        if pos >= total:
            break
        try:
            r = s.get(files_url(rel), params={"_xsrf": s.cookies.get("_xsrf")},
                      headers={"Range": f"bytes={pos}-"}, stream=True, timeout=(30, 120))
            if r.status_code in (401, 403):
                r.close(); relogin(s); continue           # session expired -> re-auth
            r.raise_for_status()
            done = pos
            with open(local, "ab") as f:                  # append: keep what we already have
                for blk in r.iter_content(chunk):
                    if blk:
                        if done + len(blk) > total:        # never write past the expected size
                            blk = blk[:total - done]
                        f.write(blk)
                        done += len(blk)
                        if done >= total:
                            break
                        if progress:
                            rate = done / max(time.time() - t0, 1e-3) / 1e6
                            sys.stdout.write(f"\r  {done/1e9:6.2f} / {total/1e9:.2f} GB "
                                             f"({100*done/total:5.1f}%)  {rate:6.1f} MB/s   ")
                            sys.stdout.flush()
            r.close()
            attempt = 0
        except (requests.exceptions.RequestException, ssl.SSLError, OSError) as e:
            attempt += 1
            wait = min(30, 2 ** attempt)                  # exponential backoff, capped at 30s
            sys.stdout.write(f"\n  ! transfer interrupted ({type(e).__name__}); "
                             f"resume in {wait}s (try {attempt})\n")
            time.sleep(wait)
    if progress:
        sys.stdout.write("\n")
    actual = local.stat().st_size
    if actual != total:                                   # ① size check
        raise SystemExit(f"size mismatch: got {actual:,}, expected {total:,}")
    if expected_sha256:                                   # ② hash check
        got = sha256_file(local)
        if got.lower() != expected_sha256.lower():
            raise SystemExit(f"sha256 mismatch:\n  got      {got}\n  expected {expected_sha256}")
        print(f"  sha256 OK  {got}")
    print(f"  verified {actual:,} bytes  ->  {local}")
    return actual


def get_file(s, remote, local):
    """Download a single server file (resumable + size-verified)."""
    download_resumable(s, remote, local, progress=True)


def get(s, remote, local):
    """Download a server file, or recursively a whole server directory."""
    remote = relpath(remote)
    r = s.get(f"{BASE}/api/contents/{remote}")
    if r.status_code == 200 and r.json()["type"] == "directory":
        for c in r.json()["content"]:
            get(s, c["path"], os.path.join(local, c["name"]))
    else:
        get_file(s, remote, local)


# --------------------------------------------------------------------------- #
#  Remote execution via a transient kernel websocket
# --------------------------------------------------------------------------- #
def _open_ws(s, kid, timeout):
    """Open an authenticated WebSocket to a kernel's message channels."""
    from websocket import create_connection
    cookie = "; ".join(f"{c.name}={c.value}" for c in s.cookies)
    sslopt = {"cert_reqs": ssl.CERT_NONE} if getattr(s, "_insecure", False) else None
    return create_connection(BASE.replace("https://", "wss://") + f"/api/kernels/{kid}/channels",
                             header=[f"Cookie: {cookie}", f"X-XSRFToken: {s.cookies.get('_xsrf')}"],
                             timeout=timeout, sslopt=sslopt)


def _exec(s, code, timeout, on_stream=None):
    """Run `code` in a fresh kernel, gather its stream output, then delete the kernel."""
    r = s.post(f"{BASE}/api/kernels", headers=_xsrf(s), json={"name": "python3"})
    r.raise_for_status()
    kid = r.json()["id"]
    ws = _open_ws(s, kid, timeout)
    mid = uuid.uuid4().hex
    ws.send(json.dumps({
        "header": {"msg_id": mid, "username": "auto", "session": uuid.uuid4().hex,
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {},
        "content": {"code": code, "silent": False, "store_history": False,
                    "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
        "channel": "shell"}))
    buf = []
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = json.loads(ws.recv())
            if msg.get("parent_header", {}).get("msg_id") != mid:   # ignore other messages
                continue
            mt, c = msg["msg_type"], msg["content"]
            if mt == "stream":                          # stdout / stderr chunks
                buf.append(c["text"])
                if on_stream:
                    on_stream(c["text"])
            elif mt == "error":
                tb = "\n".join(c["traceback"])
                buf.append(tb)
                if on_stream:
                    on_stream(tb + "\n")
            elif mt == "status" and c["execution_state"] == "idle":  # execution finished
                break
    finally:
        ws.close()
        s.delete(f"{BASE}/api/kernels/{kid}", headers=_xsrf(s))      # always free the kernel
    return "".join(buf)


def _parse_rc(out):
    """Split __PEGASUS_RC__ sentinel from captured output; return (clean_text, exit_code)."""
    rc, lines = 0, []
    for line in out.splitlines(keepends=True):
        if line.startswith("__PEGASUS_RC__="):
            rc = int(line.split("=", 1)[1])
        else:
            lines.append(line)
    return "".join(lines), rc


def run(s, command, timeout=86400):
    """Run a bash command on the server, streaming stdout/stderr live. Returns exit code."""
    code = ("import subprocess,sys\n"
            "p=subprocess.Popen(['bash','-lc'," + repr(command) + "],"
            "stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\n"
            "for l in p.stdout: print(l,end='',flush=True)\n"      # flush per line -> live
            "print('\\n__PEGASUS_RC__=%d'%p.wait())\n")            # sentinel carries exit code

    def emit(t):                                           # stream live, but hide the sentinel
        if "__PEGASUS_RC__=" in t:
            t = t.split("__PEGASUS_RC__=")[0].rstrip("\n")
        if t:
            sys.stdout.write(t); sys.stdout.flush()

    _, rc = _parse_rc(_exec(s, code, timeout, on_stream=emit))
    return rc


def sh(s, command, timeout=300):
    """Run a bash command, capture output. Returns (stdout_text, exit_code). Non-streaming."""
    code = ("import subprocess\n"
            "r=subprocess.run(['bash','-lc'," + repr(command) + "],"
            "capture_output=True,text=True)\n"
            "print(r.stdout,end='')\n"
            "print(r.stderr,end='')\n"
            "print('__PEGASUS_RC__=%d'%r.returncode)\n")
    return _parse_rc(_exec(s, code, timeout))


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main():
    """CLI entry point: parse arguments, log in, then dispatch the chosen command."""
    ap = argparse.ArgumentParser(description="Pegasus Jupyter automation client")
    ap.add_argument("--insecure", action="store_true",
                    help="skip TLS verification (only if truststore is unavailable)")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    p = sub.add_parser("ls", help="list a remote directory")
    p.add_argument("path", nargs="?", default="",
                   help="remote dir relative to /data (default: root)")

    p = sub.add_parser("put", help="upload a file or directory (recursive)")
    p.add_argument("local", help="local file or folder to upload")
    p.add_argument("remote", help="destination path on the server")

    p = sub.add_parser("get", help="download a file or directory (resumable + verified)")
    p.add_argument("remote", help="remote path on the server")
    p.add_argument("local", help="local destination path")

    p = sub.add_parser("rm", help="recursively delete a remote path")
    p.add_argument("path", help="remote file or directory to delete")

    p = sub.add_parser("mkdir", help="create a remote directory")
    p.add_argument("path", help="remote directory to create")

    p = sub.add_parser("run", help="run a bash command on the server (live output)")
    p.add_argument("command", nargs="?", help="shell command string to run")
    p.add_argument("--file", help="run the contents of a local script file instead")

    args = ap.parse_args()

    s = connect(args.insecure)
    if args.cmd == "ls":
        ls(s, args.path)
    elif args.cmd == "put":
        put(s, args.local, args.remote)
    elif args.cmd == "get":
        get(s, args.remote, args.local)
    elif args.cmd == "rm":
        rm(s, args.path)
    elif args.cmd == "mkdir":
        mkdir(s, args.path)
    elif args.cmd == "run":
        cmd = pathlib.Path(args.file).read_text(encoding="utf-8") if args.file else args.command
        if not cmd:
            ap.error("run needs a command or --file")
        sys.exit(run(s, cmd))


if __name__ == "__main__":
    main()
