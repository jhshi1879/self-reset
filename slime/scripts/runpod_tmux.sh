#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
mode="${2:-preflight}"
default_auto_stop=YES
if [[ "${mode}" == "preflight" ]]; then
    default_auto_stop=NO
fi
auto_stop="${3:-${default_auto_stop}}"

if [[ "${1:-}" == "--worker" ]]; then
    if [[ "${mode}" != "preflight" && "${mode}" != "download" && "${mode}" != "train" ]]; then
        echo "Worker mode must be preflight, download, or train." >&2
        exit 1
    fi
    if [[ "${mode}" == "train" && "${4:-NO}" != "YES" ]]; then
        echo "Training worker requires explicit confirmation." >&2
        exit 1
    fi
    if [[ "${auto_stop}" != "YES" && "${auto_stop}" != "NO" ]]; then
        echo "Automatic stop must be YES or NO." >&2
        exit 1
    fi
    if [[ "${auto_stop}" == "YES" ]]; then
        stop_after_worker() {
            local exit_code=$?
            trap - EXIT
            if ! bash "${SCRIPT_DIR}/runpod_stop.sh" --stop; then
                echo "Automatic stop failed. Stop this pod manually in the Runpod console." >&2
                (( exit_code != 0 )) || exit_code=1
            fi
            exit "${exit_code}"
        }
        trap stop_after_worker EXIT
    fi
    SAVE_DIR="${SAVE_DIR:-/workspace/checkpoints/Self-RESET}"
    LOG_DIR="${SAVE_DIR}/logs"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/tmux-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
    exec > >(tee -a "${LOG_FILE}") 2>&1
    echo "tmux session output: ${LOG_FILE}"
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "HF_TOKEN is not set; gated model access requires a saved Hugging Face login or a token." >&2
    fi
    if [[ "${auto_stop}" == "YES" ]]; then
        bash "${SCRIPT_DIR}/runpod_stop.sh" --check
    fi
    status=0
    if [[ "${mode}" == "download" ]]; then
        bash "${SCRIPT_DIR}/runpod_download.sh" || status=$?
    else
        confirm_run=NO
        if [[ "${mode}" == "train" ]]; then
            confirm_run=YES
        fi
        RUNPOD_AUTO_STOP=NO RUNPOD_START_GUARD=YES CONFIRM_RUN="${confirm_run}" \
            bash "${SCRIPT_DIR}/runpod.sh" || status=$?
    fi
    if (( status == 0 )); then
        if [[ "${mode}" == "download" ]]; then
            echo "Downloads completed; use the same volume on a GPU pod for conversion and preflight."
        elif [[ "${mode}" == "preflight" ]]; then
            if [[ "${auto_stop}" == "YES" ]]; then
                echo "Preflight passed. Restart the pod and launch train mode to start training."
            else
                echo "Preflight passed. To start training in this tmux session:"
                printf 'CONFIRM_RUN=YES bash %q\n' "${SCRIPT_DIR}/runpod.sh"
            fi
        else
            echo "Training completed."
        fi
    else
        echo "${mode} failed with exit code ${status}; see the logs above." >&2
    fi
    if [[ "${auto_stop}" == "YES" ]]; then
        exit "${status}"
    fi
    echo "Detach without stopping the session using Ctrl-b, then d."
    exec bash -i
fi

session="${1:-self-reset}"
if [[ ! "${session}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "tmux session name may contain only letters, numbers, underscores, and hyphens." >&2
    exit 1
fi
if [[ "${mode}" != "preflight" && "${mode}" != "download" && "${mode}" != "train" ]]; then
    echo "Session mode must be preflight, download, or train." >&2
    exit 1
fi
if [[ "${mode}" == "train" && "${CONFIRM_RUN:-NO}" != "YES" ]]; then
    echo "Set CONFIRM_RUN=YES to start training." >&2
    exit 1
fi
if [[ "${auto_stop}" != "YES" && "${auto_stop}" != "NO" ]]; then
    echo "Automatic stop must be YES or NO." >&2
    exit 1
fi
if ! command -v tmux >/dev/null; then
    if [[ "$(id -u)" != 0 ]] || ! command -v apt-get >/dev/null; then
        echo "tmux is missing; install it on the pod and rerun the upload script." >&2
        exit 1
    fi
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tmux
fi
if tmux has-session -t "=${session}" 2>/dev/null; then
    echo "tmux session ${session} already exists; attach to it instead." >&2
    exit 1
fi
confirmation=NO
if [[ "${mode}" == "train" ]]; then
    confirmation=YES
fi
printf -v worker_command 'bash %q --worker %q %q %q' "${SCRIPT_DIR}/runpod_tmux.sh" "${mode}" "${auto_stop}" "${confirmation}"
if [[ "${mode}" == "train" ]]; then
    printf -v worker_command 'RUNPOD_TRAIN_TRIAL=%q REUSE_BASELINE_LOG=%q %s' \
        "${RUNPOD_TRAIN_TRIAL:-NO}" "${REUSE_BASELINE_LOG:-}" "${worker_command}"
fi
tmux new-session -d -s "${session}" "${worker_command}"
echo "Started tmux session: ${session}"
