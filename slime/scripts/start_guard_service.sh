#!/bin/bash
# ============================================================================
# Guard Model HTTP Service Launcher
#
# Starts the guard model service before training. Set GUARD_DEVICE=cpu
# to keep the guard models off the training GPUs.
#
# Usage:
#   bash scripts/start_guard_service.sh [GPU_ID] [PORT]
#
# Example:
#   bash scripts/start_guard_service.sh 7 8100
#
# Then in your training script, add:
#   --guard-service-url http://localhost:8100
#   --actor-num-gpus-per-node 8   (use all 8 GPUs for training+rollout)
# ============================================================================

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SAVE_DIR="${SAVE_DIR:-/workspace/checkpoints/Self-RESET}"
LOG_DIR="${RUN_LOG_DIR:-${SAVE_DIR}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/guard-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
source "${SCRIPT_DIR}/runpod_util.sh"
runpod_ensure_environment
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "${HF_HOME}"
runpod_set_model_paths
runpod_ensure_hf_models NO YES NO
runpod_ensure_guard_environment
command -v curl >/dev/null || { echo "ERROR: curl is required" >&2; exit 1; }
PORT=${2:-8100}
case "${GUARD_DEVICE:-gpu}" in
    cpu)
        GUARD_CUDA_DEVICES=""
        DEVICE_LABEL=CPU
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
        ;;
    gpu)
        GPU_COUNT=$("${RUNPOD_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')
        if [ "${GPU_COUNT}" -lt 1 ]; then
            echo "ERROR: No CUDA GPUs are visible to PyTorch" >&2
            exit 1
        fi
        GPU_ID=${1:-${GUARD_GPU_ID:-$((GPU_COUNT - 1))}}
        if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]] || [ "${GPU_ID}" -ge "${GPU_COUNT}" ]; then
            echo "ERROR: GPU_ID must be between 0 and $((GPU_COUNT - 1))" >&2
            exit 1
        fi
        GUARD_CUDA_DEVICES="${GPU_ID}"
        DEVICE_LABEL="GPU ${GPU_ID}"
        ;;
    *)
        echo "ERROR: GUARD_DEVICE must be cpu or gpu" >&2
        exit 1
        ;;
esac

: "${SAFETY_STREAM_MODEL_PATH:?Set SAFETY_STREAM_MODEL_PATH to the Qwen3Guard-Stream model path or repository ID.}"
: "${SAFETY_GUARD_MODEL_PATH:?Set SAFETY_GUARD_MODEL_PATH to the Qwen3Guard-Gen model path or repository ID.}"
: "${LLAMA_GUARD_MODEL_PATH:?Set LLAMA_GUARD_MODEL_PATH to the LlamaGuard model path or repository ID.}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent output directory for guard-service logs.}"

STREAM_MODEL="${SAFETY_STREAM_MODEL_PATH}"
GUARD_MODEL="${SAFETY_GUARD_MODEL_PATH}"
EVAL_MODEL="${EVAL_MODEL_PATH:-${SAFETY_GUARD_MODEL_PATH}}"
LLAMA_GUARD_MODEL="${LLAMA_GUARD_MODEL_PATH}"

echo "Starting guard model service on ${DEVICE_LABEL}, port ${PORT}..."
echo "Guard service output will be saved to ${LOG_FILE}"

export PYTHONPATH="${SLIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${SLIME_ROOT}"
CUDA_VISIBLE_DEVICES="${GUARD_CUDA_DEVICES}" PYTHONNOUSERSITE=1 "${GUARD_PYTHON}" -m slime.services.guard_service \
    --safety-stream-model-path "${STREAM_MODEL}" \
    --safety-guard-model-path "${GUARD_MODEL}" \
    --eval-model-path "${EVAL_MODEL}" \
    --llama-guard-model-path "${LLAMA_GUARD_MODEL}" \
    --host "${GUARD_HOST:-127.0.0.1}" \
    --port "${PORT}" &

GUARD_PID=$!
echo "Guard service PID: ${GUARD_PID}"

# Wait for the service to be healthy
echo "Waiting for guard service to be ready..."
for i in $(seq 1 120); do
    if curl --fail --silent "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Guard service is ready on http://localhost:${PORT}"
        exit 0
    fi
    if ! kill -0 "${GUARD_PID}" 2>/dev/null; then
        wait "${GUARD_PID}" || true
        echo "ERROR: Guard service exited before becoming ready; see ${LOG_FILE}." >&2
        exit 1
    fi
    sleep 2
done

echo "ERROR: Guard service failed to start within 240 seconds"
kill "${GUARD_PID}"
exit 1
