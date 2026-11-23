#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPLICATION_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

: "${RUNPOD_SSH_HOST:?Set RUNPOD_SSH_HOST to the pod SSH user and host, or an SSH config alias.}"
RUNPOD_SSH_PORT="${RUNPOD_SSH_PORT:-}"
RUNPOD_SSH_HOST_KEY_ALIAS="${RUNPOD_SSH_HOST_KEY_ALIAS:-}"
RUNPOD_PROJECT_DIR="${RUNPOD_PROJECT_DIR:-/workspace/Self-ReSET}"
RUNPOD_TMUX_SESSION="${RUNPOD_TMUX_SESSION:-self-reset}"
RUNPOD_ATTACH="${RUNPOD_ATTACH:-YES}"
RUNPOD_MODE="${RUNPOD_MODE:-preflight}"
default_auto_stop=YES
if [[ "${RUNPOD_MODE}" == "preflight" ]]; then
    default_auto_stop=NO
fi
RUNPOD_AUTO_STOP="${RUNPOD_AUTO_STOP:-${default_auto_stop}}"

if [[ ! "${RUNPOD_SSH_HOST}" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.@:-]*$ ]]; then
    echo "RUNPOD_SSH_HOST must be an SSH host or user@host without shell metacharacters." >&2
    exit 1
fi
if [[ -n "${RUNPOD_SSH_PORT}" ]] && { ! [[ "${RUNPOD_SSH_PORT}" =~ ^[0-9]{1,5}$ ]] || (( 10#${RUNPOD_SSH_PORT} < 1 || 10#${RUNPOD_SSH_PORT} > 65535 )); }; then
    echo "RUNPOD_SSH_PORT must be a port number between 1 and 65535." >&2
    exit 1
fi
if [[ ! "${RUNPOD_PROJECT_DIR}" =~ ^/[a-zA-Z0-9_./-]+$ ]] || [[ "/${RUNPOD_PROJECT_DIR}/" == *"/../"* ]]; then
    echo "RUNPOD_PROJECT_DIR must be an absolute path without shell metacharacters or '..' components." >&2
    exit 1
fi
if [[ ! "${RUNPOD_TMUX_SESSION}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "RUNPOD_TMUX_SESSION may contain only letters, numbers, underscores, and hyphens." >&2
    exit 1
fi
if [[ "${RUNPOD_ATTACH}" != "YES" && "${RUNPOD_ATTACH}" != "NO" ]]; then
    echo "RUNPOD_ATTACH must be YES or NO." >&2
    exit 1
fi
if [[ "${RUNPOD_MODE}" != "preflight" && "${RUNPOD_MODE}" != "download" && "${RUNPOD_MODE}" != "train" ]]; then
    echo "RUNPOD_MODE must be preflight, download, or train." >&2
    exit 1
fi
if [[ "${RUNPOD_MODE}" == "train" && "${CONFIRM_RUN:-NO}" != "YES" ]]; then
    echo "Set CONFIRM_RUN=YES to start training." >&2
    exit 1
fi
if [[ "${RUNPOD_AUTO_STOP}" != "YES" && "${RUNPOD_AUTO_STOP}" != "NO" ]]; then
    echo "RUNPOD_AUTO_STOP must be YES or NO." >&2
    exit 1
fi

command -v ssh >/dev/null || { echo "ssh is required." >&2; exit 1; }
command -v rsync >/dev/null || { echo "rsync is required locally and on the pod." >&2; exit 1; }

ssh_command=(ssh)
if [[ -n "${RUNPOD_SSH_PORT}" ]]; then
    ssh_command+=(-p "${RUNPOD_SSH_PORT}")
fi
if [[ -n "${RUNPOD_SSH_HOST_KEY_ALIAS}" ]]; then
    alias_pattern='^(\[[a-zA-Z0-9_.:-]+\](:[0-9]+)?|[a-zA-Z0-9_.:-]+)$'
    if ! [[ "${RUNPOD_SSH_HOST_KEY_ALIAS}" =~ ${alias_pattern} ]]; then
        echo "RUNPOD_SSH_HOST_KEY_ALIAS is not a valid known_hosts alias." >&2
        exit 1
    fi
    ssh_command+=(-o StrictHostKeyChecking=yes -o "HostKeyAlias=${RUNPOD_SSH_HOST_KEY_ALIAS}")
fi

echo "Uploading replication source and data to ${RUNPOD_SSH_HOST}:${RUNPOD_PROJECT_DIR}/replication/"
"${ssh_command[@]}" "${RUNPOD_SSH_HOST}" \
    "if command -v tmux >/dev/null && tmux has-session -t '=${RUNPOD_TMUX_SESSION}' 2>/dev/null; then echo 'tmux session ${RUNPOD_TMUX_SESSION} already exists; attach to it instead of uploading again.' >&2; exit 1; fi; mkdir -p -- ${RUNPOD_PROJECT_DIR}/replication"
rsync -rlt --no-links \
    --exclude='.*' \
    --include='*/' \
    --include='*.py' --include='*.sh' --include='*.toml' \
    --include='*.txt' --include='*.md' --include='*.jsonl' --include='LICENSE' \
    --exclude='*' \
    -e "${ssh_command[*]}" \
    "${REPLICATION_ROOT}/" "${RUNPOD_SSH_HOST}:${RUNPOD_PROJECT_DIR}/replication/"
confirmation=NO
training_env=""
if [[ "${RUNPOD_MODE}" == "train" ]]; then
    confirmation=YES
    printf -v training_env 'RUNPOD_TRAIN_TRIAL=%q REUSE_BASELINE_LOG=%q ' \
        "${RUNPOD_TRAIN_TRIAL:-NO}" "${REUSE_BASELINE_LOG:-}"
fi
"${ssh_command[@]}" "${RUNPOD_SSH_HOST}" \
    "${training_env}CONFIRM_RUN=${confirmation} bash ${RUNPOD_PROJECT_DIR}/replication/slime/scripts/runpod_tmux.sh ${RUNPOD_TMUX_SESSION} ${RUNPOD_MODE} ${RUNPOD_AUTO_STOP}"
if [[ "${RUNPOD_MODE}" == "download" ]]; then
    echo "CPU downloads are running in tmux; conversion and training have not been started."
elif [[ "${RUNPOD_MODE}" == "train" ]]; then
    echo "Training is starting in tmux."
else
    echo "Preflight is running in tmux; training has not been started."
fi
if [[ "${RUNPOD_AUTO_STOP}" == "YES" ]]; then
    echo "The pod will be asked to stop when ${RUNPOD_MODE} succeeds or fails."
fi
printf 'To reconnect:'
printf ' %q' "${ssh_command[@]}" -t "${RUNPOD_SSH_HOST}" tmux attach-session -t "=${RUNPOD_TMUX_SESSION}"
printf '\n'
if [[ "${RUNPOD_ATTACH}" == "YES" && -t 0 && -t 1 ]]; then
    "${ssh_command[@]}" -t "${RUNPOD_SSH_HOST}" "tmux attach-session -t '=${RUNPOD_TMUX_SESSION}'"
fi
