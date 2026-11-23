#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPLICATION_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${REPLICATION_ROOT}/.." &>/dev/null && pwd)"
SLIME_ROOT="${REPLICATION_ROOT}/slime"
source "${SCRIPT_DIR}/runpod_util.sh"

if [[ -z "${RUNPOD_PYTHON:-}" ]]; then
    for candidate in python3 python3.12 python3.11 python3.10; do
        candidate_path="$(command -v "${candidate}" || true)"
        if [[ -n "${candidate_path}" ]] && "${candidate_path}" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
            RUNPOD_PYTHON="${candidate_path}"
            break
        fi
    done
fi
if [[ -z "${RUNPOD_PYTHON:-}" ]] || ! "${RUNPOD_PYTHON}" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 or newer is required"'; then
    echo "Python 3.10 or newer is required for CPU downloads." >&2
    exit 1
fi
echo "Using CPU download Python: ${RUNPOD_PYTHON}"

if ! "${RUNPOD_PYTHON}" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("huggingface_hub") is None)'; then
    download_venv="${RUNPOD_DOWNLOAD_VENV:-/workspace/.venvs/self-reset-download}"
    if [[ -e "${download_venv}" && ! -f "${download_venv}/.self-reset-download-venv" ]]; then
        echo "Refusing to modify an unmanaged download environment: ${download_venv}" >&2
        exit 1
    fi
    if [[ ! -f "${download_venv}/.self-reset-venv-ready" || ! -x "${download_venv}/bin/python3" ]]; then
        mkdir -p "${download_venv}"
        touch "${download_venv}/.self-reset-download-venv"
        if ! "${RUNPOD_PYTHON}" -m venv "${download_venv}"; then
            echo "Could not create the download environment; Python venv and pip are required." >&2
            exit 1
        fi
        touch "${download_venv}/.self-reset-venv-ready"
    fi
    RUNPOD_PYTHON="${download_venv}/bin/python3"
    "${RUNPOD_PYTHON}" -m pip install --only-binary=:all: 'huggingface_hub==1.9.2'
fi
export RUNPOD_PYTHON
"${RUNPOD_PYTHON}" -c 'from huggingface_hub import HfApi, hf_hub_download, snapshot_download; from huggingface_hub.errors import GatedRepoError'
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "${HF_HOME}"
runpod_set_model_paths
runpod_ensure_data
runpod_ensure_hf_models YES YES YES
echo "CPU downloads complete. Checkpoint conversion, guard startup, and training require a GPU pod."
