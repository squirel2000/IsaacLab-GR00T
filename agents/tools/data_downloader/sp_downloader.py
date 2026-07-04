#!/usr/bin/env python3
"""
SharePoint / OneDrive downloader (reuses your browser-login cookies).

Usage:
    python3 sp_downloader.py ls [folder]
            List one folder (sub-folders + files).

    python3 sp_downloader.py get <path1> [path2 ...] [--flat] [-d dest_dir]
            Download one or many paths. Each path may be a file or a folder;
            folders are downloaded recursively. Pass --flat (-1) to download
            only a folder's top-level files. Pass -d to force the destination.

Paths accept two forms:
    1. Relative to "Shared archives", e.g. "Checkpoints" or "finetune record.xlsx"
    2. Server-absolute, e.g. "/personal/ming_li_asus_com/Documents/..."

Paths with spaces:
    Quote the shell argument, e.g. "Checkpoints/GR00T N1.7/..."
    or use %20 in place of spaces.

Cookies:
    Expected at <script folder>/cookies.txt (keep only sharepoint.com cookies).
    Override with SP_COOKIES. When cookies expire, re-export cookies.txt.

Default destinations (used when dest_dir is omitted):
    - anything under Checkpoints -> ~/Gits/IsaacLab-GR00T/artifacts/checkpoints/gr00t
    - anything under Datasets    -> ~/Gits/IsaacLab-GR00T/datasets
    - everything else            -> ./downloads
A dest_dir argument always overrides. Override defaults with SP_CKPT_DIR / SP_DATA_DIR.

Automation behavior (get):
    - shows a live progress meter per file (%, size, speed, time left/ETA)
    - verifies each file by size after download
    - retries and resumes interrupted downloads (SP_RETRIES / SP_RETRY_DELAY)
    - auto-extracts .zip files after verified download (set SP_AUTO_UNZIP=0 to disable)
    - exits 0 only when everything is complete and verified

Example (recursive folder download with spaces):
    python3 sp_downloader.py get \
        "Checkpoints/GR00T N1.7/openarm_linkerhando6_multitask_N17_150k_dataset_0403_no_tune_visual"
"""
import sys
import os
import json
import time
import subprocess
import urllib.parse
import zipfile

# --- Configuration ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")

# cookies.txt lives next to this script (travels with it when copied)
COOKIES = os.environ.get("SP_COOKIES", os.path.join(SCRIPT_DIR, "cookies.txt"))
SITE = "https://asus-my.sharepoint.com/personal/ming_li_asus_com"
ROOT = "/personal/ming_li_asus_com/Documents/Humannoid Robot/Shared archives"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36")

CKPT_DIR = os.environ.get(
    "SP_CKPT_DIR", os.path.join(HOME, "Gits/IsaacLab-GR00T/artifacts/checkpoints/gr00t"))
DATA_DIR = os.environ.get(
    "SP_DATA_DIR", os.path.join(HOME, "Gits/IsaacLab-GR00T/datasets"))
DOWNLOAD_DIR = "./downloads"

RETRIES = int(os.environ.get("SP_RETRIES", "30"))        # max attempts per file
RETRY_DELAY = int(os.environ.get("SP_RETRY_DELAY", "15"))  # seconds between attempts
AUTO_UNZIP = os.environ.get("SP_AUTO_UNZIP", "1").strip().lower() not in {
    "0", "false", "no", "off"
}


# --- Helpers ---------------------------------------------------------------
def resolve(path):
    """Turn user input into a server-absolute path."""
    path = urllib.parse.unquote(path.strip())
    if path.startswith("/personal/"):
        return path.rstrip("/")
    return (ROOT + "/" + path.strip("/")).rstrip("/")


def enc(path):
    return urllib.parse.quote(path, safe="/")


def default_dest(server_path):
    if "/Checkpoints" in server_path:
        return CKPT_DIR
    if "/Datasets" in server_path:
        return DATA_DIR
    return DOWNLOAD_DIR


def api_json(api_path):
    """Call the SharePoint REST API and return parsed JSON."""
    url = f"{SITE}/_api/web/{api_path}"
    body = subprocess.run(
        ["curl", "-s", "-b", COOKIES, "-A", UA,
         "-H", "Accept: application/json;odata=nometadata", url],
        capture_output=True, text=True).stdout
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        sys.exit("Auth failed -- cookies have likely expired. "
                 "Re-export cookies.txt and try again.")


def list_folder(server_path):
    p = enc(server_path)
    folders = api_json(f"GetFolderByServerRelativeUrl('{p}')/Folders"
                       "?$select=Name,ItemCount,ServerRelativeUrl").get("value", [])
    files = api_json(f"GetFolderByServerRelativeUrl('{p}')/Files"
                     "?$select=Name,Length,ServerRelativeUrl").get("value", [])
    return folders, files


def remote_size(server_file):
    info = api_json(f"GetFileByServerRelativeUrl('{enc(server_file)}')?$select=Length")
    return int(info["Length"]) if "Length" in info else None


def is_html(path):
    """True if the file starts like an HTML login page."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(256).lstrip().lower()
    except OSError:
        return False
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def human(n):
    return f"{n / 1048576:.2f} MB" if n < 1 << 30 else f"{n / (1 << 30):.2f} GB"


def local_path(server_file, dest_dir):
    return os.path.join(dest_dir, os.path.basename(server_file))


def unzip_marker(zip_path):
    return zip_path + ".sp_unzip_ok"


def zip_signature(zip_path):
    st = os.stat(zip_path)
    return f"{st.st_size}:{st.st_mtime_ns}"


def extract_zip_if_needed(file_path, prefix=""):
    """Extract a .zip once per file-content signature."""
    if not AUTO_UNZIP or not file_path.lower().endswith(".zip"):
        return "ok"

    marker = unzip_marker(file_path)
    signature = zip_signature(file_path)
    if os.path.exists(marker):
        try:
            with open(marker, "r", encoding="utf-8") as fh:
                if fh.read().strip() == signature:
                    print(f"  {prefix}skip unzip (already extracted): {os.path.basename(file_path)}")
                    return "ok"
        except OSError:
            pass

    print(f"  {prefix}extracting: {os.path.basename(file_path)}", flush=True)
    target_dir = os.path.dirname(file_path) or "."
    target_abs = os.path.abspath(target_dir)

    try:
        with zipfile.ZipFile(file_path) as zf:
            # Block zip-slip paths before extraction.
            for member in zf.namelist():
                member_abs = os.path.abspath(os.path.join(target_abs, member))
                if member_abs != target_abs and not member_abs.startswith(target_abs + os.sep):
                    raise ValueError(f"unsafe zip member path: {member}")
            zf.extractall(target_dir)
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write(signature)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError) as exc:
        try:
            if os.path.exists(marker):
                os.remove(marker)
        except OSError:
            pass
        print(f"  {prefix}UNZIP FAILED ({os.path.basename(file_path)}): {exc}")
        return "fail"

    print(f"    extracted -> {target_dir}")
    return "ok"


# --- Download core ---------------------------------------------------------
def attempt_download(server_file, dest_dir, size, prefix=""):
    """
    One download attempt with resume. Returns:
      "ok"    -> file verified complete
      "auth"  -> login page / 401 / 403 -> cookies expired (fatal, no point retrying)
      "retry" -> partial or transient network/server error (resumable)
    """
    os.makedirs(dest_dir, exist_ok=True)
    name = os.path.basename(server_file)
    dest = local_path(server_file, dest_dir)

    existed = os.path.exists(dest)
    if existed:
        if is_html(dest):                       # stale login page from a past failure
            os.remove(dest)
            existed = False
        elif size is not None and os.path.getsize(dest) == size:
            print(f"  {prefix}skip (already complete): {name}")
            return "ok"
        elif size is not None and os.path.getsize(dest) > size:
            os.remove(dest)                     # corrupt / oversized -> restart
            existed = False

    print(f"  {prefix}{'resuming' if existed else 'downloading'}: {name}", flush=True)
    # curl's default progress meter -> %, total/received size, avg + current
    # speed, time spent and time left (ETA). Streams to stderr.
    cmd = ["curl", "-L", "-b", COOKIES, "-A", UA,
           "-o", dest, "-w", "%{http_code}"]
    if existed:
        cmd += ["-C", "-"]                      # continue from current offset
    cmd.append(f"{SITE}/_layouts/15/download.aspx?SourceUrl={enc(server_file)}")

    # progress meter -> stderr (visible); http code captured from stdout
    code = subprocess.run(cmd, stdout=subprocess.PIPE, text=True).stdout.strip()

    got_login_page = is_html(dest)
    if code in ("401", "403") or got_login_page:
        if got_login_page:
            os.remove(dest)                     # drop the login page so resume stays clean
        return "auth"
    if code not in ("200", "206"):
        return "retry"                          # transient (e.g. 000 timeout, 5xx)
    if size is not None and os.path.getsize(dest) != size:
        return "retry"                          # incomplete -> resume next attempt
    print(f"    done: {human(os.path.getsize(dest))} -> {dest}")
    return "ok"


def fetch(server_file, dest_dir, size, prefix=""):
    """Retry/resume until complete. Returns 'ok', 'fail', or 'auth'."""
    name = os.path.basename(server_file)
    for attempt in range(1, RETRIES + 1):
        status = attempt_download(server_file, dest_dir, size, prefix)
        if status == "ok":
            return "ok"
        if status == "auth":
            print(f"  {prefix}AUTH FAILED ({name}): cookies expired -- "
                  "re-export cookies.txt.")
            return "auth"
        if attempt < RETRIES:
            print(f"  {prefix}interrupted ({name}); retry {attempt}/{RETRIES} "
                  f"in {RETRY_DELAY}s ...", flush=True)
            time.sleep(RETRY_DELAY)
    print(f"  {prefix}GAVE UP ({name}) after {RETRIES} attempts.")
    return "fail"


def download_one(server_file, dest_dir, size, prefix=""):
    """Download one file then run post-download steps."""
    status = fetch(server_file, dest_dir, size, prefix)
    if status != "ok":
        return status
    unzip_status = extract_zip_if_needed(local_path(server_file, dest_dir), prefix)
    return "ok" if unzip_status == "ok" else "fail"


def parse_args(args):
    usage = ("usage: python3 sp_downloader.py get <path1> [path2 ...] "
             "[--flat] [-d dest_dir]")
    paths, dest_dir, recursive = [], None, True
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-d", "--dest"):
            if i + 1 >= len(args):
                sys.exit(usage)
            dest_dir = os.path.expanduser(args[i + 1])
            i += 2
            continue
        if a in ("--flat", "-1"):
            recursive = False
            i += 1
            continue
        paths.append(a)
        i += 1

    if not paths:
        sys.exit(usage)
    return paths, dest_dir, recursive


def gather_items(user_path, forced_dest, recursive):
    """Resolve a path to [(server_file, dest_dir, size)] -- file or folder."""
    server_path = resolve(user_path)
    dest = forced_dest if forced_dest else default_dest(server_path)
    size = remote_size(server_path)
    if size is not None:                        # a single file
        return [(server_path, dest, size)]

    items = []                                  # a folder -> walk it, keeping its name
    def walk(spath, dpath):
        folders, files = list_folder(spath)
        for f in files:
            items.append((f["ServerRelativeUrl"], dpath, int(f["Length"])))
        if recursive:
            for d in folders:
                walk(d["ServerRelativeUrl"], os.path.join(dpath, d["Name"]))

    walk(server_path, os.path.join(dest, os.path.basename(server_path)))
    return items


# --- Commands --------------------------------------------------------------
def cmd_ls(args):
    server_path = resolve(args[0]) if args else ROOT
    folders, files = list_folder(server_path)
    print(f"\n  location: {server_path}\n")
    for f in folders:
        print(f"  [DIR]  {f['Name']}/   (items: {f.get('ItemCount', '?')})")
    for f in files:
        print(f"  {int(f['Length']) / 1048576:10.2f} MB  {f['Name']}")
    print(f"\n  {len(folders)} folder(s), {len(files)} file(s).\n")


def cmd_get(args):
    paths, forced_dest, recursive = parse_args(args)
    items = []
    for p in paths:
        items += gather_items(p, forced_dest, recursive)
    if not items:
        sys.exit("nothing to download (no matching files found).")

    total = len(items)
    print(f"\n{total} file(s), {human(sum(s for _, _, s in items))} total\n")

    done, failed = 0, []
    for i, (sf, dd, sz) in enumerate(items, 1):
        prefix = f"[{i}/{total}] " if total > 1 else ""
        status = download_one(sf, dd, sz, prefix=prefix)
        if status == "ok":
            done += 1
        elif status == "auth":
            failed.append(os.path.basename(sf))
            print("\nAborting: cookies expired. Re-export cookies.txt and re-run "
                  "(completed files are skipped).")
            break
        else:
            failed.append(os.path.basename(sf))

    if total > 1:
        print(f"\n=== complete: {done}/{total} ===")
        if failed:
            print("failed:", ", ".join(failed))
    sys.exit(0 if done == total else 1)


def main():
    cmds = {"ls": cmd_ls, "get": cmd_get}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        return
    cmds[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
