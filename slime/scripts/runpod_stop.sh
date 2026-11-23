#!/usr/bin/env bash
set -Eeuo pipefail

python3 - "$@" <<'PY'
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

args = sys.argv[1:]
if args not in (["--check"], ["--stop"]) and not (
    len(args) == 2 and args[0] in ("--watch", "--watch-training") and args[1].isdigit()
):
    raise SystemExit("Usage: runpod_stop.sh --check | --stop | --watch DOWNLOAD_PID | --watch-training TRAIN_PID")

pod_env = {}
try:
    for entry in Path("/proc/1/environ").read_bytes().split(b"\0"):
        key, separator, value = entry.partition(b"=")
        if separator and key in (b"RUNPOD_POD_ID", b"RUNPOD_API_KEY"):
            pod_env[key.decode()] = value.decode()
except OSError as exc:
    if not os.environ.get("RUNPOD_POD_ID") or not os.environ.get("RUNPOD_API_KEY"):
        raise SystemExit(f"Cannot read Runpod pod credentials: {exc}") from exc

pod_id = os.environ.get("RUNPOD_POD_ID") or pod_env.get("RUNPOD_POD_ID", "")
api_key = os.environ.get("RUNPOD_API_KEY") or pod_env.get("RUNPOD_API_KEY", "")
if pod_env.get("RUNPOD_POD_ID") and pod_id != pod_env["RUNPOD_POD_ID"]:
    raise SystemExit("RUNPOD_POD_ID does not match this pod; refusing to stop another pod.")
if not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id) or not api_key:
    raise SystemExit("Runpod pod ID or API key is missing; stop the pod manually.")

url = f"https://api.runpod.io/v2/pods/{pod_id}"


def request(action):
    headers = {"Authorization": f"Bearer {api_key}"}
    if action == "stop":
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        url + ("/action" if action == "stop" else ""),
        headers=headers,
        data=b'{"action":"stop"}' if action == "stop" else None,
        method="POST" if action == "stop" else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Runpod {action} request failed with HTTP {exc.code}; stop the pod manually.") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"Runpod {action} request failed: {exc}; stop the pod manually.") from exc


def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return stat[0], stat[19]


if args[0] in ("--check", "--watch", "--watch-training"):
    request("check")
    print("Runpod API access verified; stop permission will be checked when requested.", flush=True)

if args[0] == "--check":
    raise SystemExit(0)

if args[0] in ("--watch", "--watch-training"):
    pid = int(args[1])
    kind, script = ("Download", b"/runpod_download.sh") if args[0] == "--watch" else ("Training", b"/runpod.sh")
    identity = process_identity(pid)
    if identity is None or identity[0] == "Z":
        raise SystemExit(f"{kind} process {pid} is not running; refusing to arm the watcher.")
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError as exc:
        raise SystemExit(f"Cannot verify {kind.lower()} process {pid}: {exc}") from exc
    if not any(part.endswith(script) for part in command):
        raise SystemExit(f"Process {pid} is not the {kind.lower()} script; refusing to arm the watcher.")
    print(f"Watching {kind.lower()} process {pid}; the pod will be stopped when it exits.", flush=True)
    while True:
        current = process_identity(pid)
        if current is None or current[0] == "Z" or current[1] != identity[1]:
            break
        time.sleep(5)

print("Requesting Runpod to stop this pod; check the Runpod console to confirm.", flush=True)
status = request("stop")
print(f"Runpod accepted the stop request (HTTP {status}).", flush=True)
PY
