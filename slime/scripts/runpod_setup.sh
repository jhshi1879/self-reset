#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/runpod_util.sh"
runpod_ensure_environment
echo "Prebuilt RunPod environment is ready: ${RUNPOD_PYTHON}"
echo "Megatron-LM checkout: ${MEGATRON_PATH}"
