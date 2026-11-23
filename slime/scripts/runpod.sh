#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RUNPOD_AUTO_STOP="${RUNPOD_AUTO_STOP:-YES}"
if [[ "${RUNPOD_AUTO_STOP}" != "YES" && "${RUNPOD_AUTO_STOP}" != "NO" ]]; then
    echo "RUNPOD_AUTO_STOP must be YES or NO." >&2
    exit 1
fi
RAY_ARCHIVE_REQUIRED=NO
finish_run() {
    local exit_code=$?
    trap - EXIT
    if [[ "${RAY_ARCHIVE_REQUIRED}" == "YES" ]]; then
        local ray_logs="${RAY_TMPDIR}/session_latest/logs"
        if [[ -d "${ray_logs}" ]]; then
            if mkdir -p "${RUN_LOG_DIR}/ray" && cp -a "${ray_logs}/." "${RUN_LOG_DIR}/ray/"; then
                echo "Archived Ray logs to ${RUN_LOG_DIR}/ray"
            else
                echo "Failed to archive Ray logs from ${ray_logs}" >&2
                (( exit_code != 0 )) || exit_code=1
            fi
        else
            echo "Ray log directory not found at ${ray_logs}" >&2
            (( exit_code != 0 )) || exit_code=1
        fi
    fi
    if [[ "${CONFIRM_RUN:-NO}" == "YES" && "${RUNPOD_AUTO_STOP}" == "YES" ]]; then
        if ! bash "${SCRIPT_DIR}/runpod_stop.sh" --stop; then
            echo "Automatic stop failed. Stop this pod manually in the Runpod console." >&2
            (( exit_code != 0 )) || exit_code=1
        fi
    fi
    exit "${exit_code}"
}
trap finish_run EXIT
if [[ "${CONFIRM_RUN:-NO}" == "YES" && "${RUNPOD_AUTO_STOP}" == "YES" ]]; then
    bash "${SCRIPT_DIR}/runpod_stop.sh" --check
fi
REPLICATION_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${REPLICATION_ROOT}/.." &>/dev/null && pwd)"
SLIME_ROOT="${REPLICATION_ROOT}/slime"
if [[ "${RUNPOD_TRAIN_TRIAL:-NO}" == "YES" ]]; then
    if [[ -n "${NUM_EPOCHS:-}" || -n "${RESUME_FROM:-}" ]]; then
        echo "The three-rollout trial requires a fresh run without NUM_EPOCHS or RESUME_FROM." >&2
        exit 1
    fi
    NUM_ROLLOUTS=3
    ROLLOUT_BATCH_SIZE=4
    N_SAMPLES_PER_PROMPT=4
    GLOBAL_BATCH_SIZE=16
    OVERSAMPLING_BATCH_SIZE=4
    ROLLOUT_MAX_RESPONSE_LEN=512
    EVAL_MAX_RESPONSE_LEN=512
    MAX_TOKENS_PER_GPU=2048
    SGLANG_MEM_FRACTION_STATIC=0.45
    SAVE_INTERVAL=1
    SAVE_DIR="${SAVE_DIR:-/workspace/checkpoints/Self-RESET-trial-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
    RUNPOD_START_GUARD=YES
    export GUARD_DEVICE=cpu
fi
SAVE_DIR="${SAVE_DIR:-/workspace/checkpoints/Self-RESET}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${SAVE_DIR}/logs/${RUN_ID}}"
mkdir -p "${SAVE_DIR}" "${RUN_LOG_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/console.log") 2>&1
source "${SCRIPT_DIR}/runpod_util.sh"
runpod_ensure_environment
runpod_ensure_data
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "${HF_HOME}"
MODEL_CONFIG="${MODEL_CONFIG:-qwen3-8B}"

case "${MODEL_CONFIG}" in
    qwen2.5-7B|qwen3-8B|llama3.1-8B-Instruct)
        source "${SCRIPT_DIR}/models/${MODEL_CONFIG}.sh"
        ;;
    *)
        echo "Unsupported MODEL_CONFIG '${MODEL_CONFIG}'. Choose qwen2.5-7B, qwen3-8B, or llama3.1-8B-Instruct." >&2
        exit 1
        ;;
esac

runpod_set_model_paths
"${RUNPOD_PYTHON}" - "${SAVE_DIR}" "${REF_LOAD}" <<'PY'
from pathlib import Path
import sys

save_dir, ref_load = (Path(value).expanduser().resolve() for value in sys.argv[1:])
if save_dir == ref_load:
    raise SystemExit("SAVE_DIR must differ from REF_LOAD so checkpoint cleanup cannot remove the reference model.")
PY
runpod_ensure_hf_models YES YES YES
runpod_create_reference_checkpoint

if [[ -n "${NUM_EPOCHS:-}" ]]; then
    RUN_LENGTH_ARGS=(--num-epoch "${NUM_EPOCHS}")
else
    RUN_LENGTH_ARGS=(--num-rollout "${NUM_ROLLOUTS:-1}")
fi

: "${MEGATRON_PATH:?Set MEGATRON_PATH to the Megatron-LM checkout.}"

if [[ -n "${RESUME_FROM:-}" && ! -f "${RESUME_FROM}/latest_checkpointed_iteration.txt" ]]; then
    echo "RESUME_FROM must contain latest_checkpointed_iteration.txt: ${RESUME_FROM}" >&2
    exit 1
fi

command -v curl >/dev/null || { echo "curl is required." >&2; exit 1; }

if [[ ! -f "${TRAIN_DATA}" ]]; then
    echo "Training data file not found: ${TRAIN_DATA}" >&2
    exit 1
fi
if [[ ! -f "${EVAL_WILDJAILBREAK}" ]]; then
    echo "WildJailbreak evaluation file not found: ${EVAL_WILDJAILBREAK}" >&2
    exit 1
fi
if [[ ! -f "${EVAL_XSTEST}" ]]; then
    echo "XSTest evaluation file not found: ${EVAL_XSTEST}" >&2
    exit 1
fi
if [[ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]]; then
    echo "REF_LOAD must be a converted Megatron checkpoint with latest_checkpointed_iteration.txt: ${REF_LOAD}" >&2
    exit 1
fi
if [[ ! -d "${MEGATRON_PATH}" ]]; then
    echo "Megatron-LM checkout not found: ${MEGATRON_PATH}" >&2
    exit 1
fi

GPU_COUNT="$("${RUNPOD_PYTHON}" - <<'PY'
import torch

count = torch.cuda.device_count()
if count < 1:
    raise SystemExit("No CUDA GPUs are visible to PyTorch in this environment.")
print(count)
PY
)"
echo "Detected ${GPU_COUNT} visible GPU(s):"
"${RUNPOD_PYTHON}" - <<'PY'
import torch

for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    memory_gib = properties.total_memory / (1024 ** 3)
    print(f"  GPU {index}: {properties.name}, {memory_gib:.1f} GiB")
PY

DEFAULT_TP_SIZE=1
if (( GPU_COUNT >= 2 && GPU_COUNT % 2 == 0 )); then
    DEFAULT_TP_SIZE=2
fi
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${DEFAULT_TP_SIZE}}"
if ! [[ "${TENSOR_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]] || (( GPU_COUNT % TENSOR_PARALLEL_SIZE != 0 )); then
    echo "TENSOR_PARALLEL_SIZE must be a positive divisor of the ${GPU_COUNT} visible GPU(s)." >&2
    exit 1
fi

GUARD_SERVICE_URL="${GUARD_SERVICE_URL:-http://127.0.0.1:8100}"
GUARD_HEALTH_URL="${GUARD_SERVICE_URL%/}/health"
if [[ "${RUNPOD_START_GUARD:-NO}" == "YES" && "${GUARD_HEALTH_URL}" == "http://127.0.0.1:8100/health" ]] &&
    ! curl --fail --silent --max-time 5 "${GUARD_HEALTH_URL}" >/dev/null; then
    bash "${SCRIPT_DIR}/start_guard_service.sh"
fi
if ! curl --fail --silent --max-time 5 "${GUARD_HEALTH_URL}" | "${RUNPOD_PYTHON}" -c '
import json
import sys

health = json.load(sys.stdin)
if health.get("status") != "ok" or not health.get("content_evaluator") or not health.get("gen_label_evaluator"):
    raise SystemExit(1)
if sys.argv[1] == "YES" and health.get("cuda_hidden") is not True:
    sys.exit("The trial requires a CPU guard. Start or restart the updated guard launcher with GUARD_DEVICE=cpu.")
' "${RUNPOD_TRAIN_TRIAL:-NO}"; then
    echo "Guard service is unavailable or missing a loaded evaluator: ${GUARD_HEALTH_URL}" >&2
    echo "Start it with scripts/start_guard_service.sh and configured model paths before training." >&2
    exit 1
fi

echo "Model config: ${MODEL_CONFIG}"
echo "Training data: ${TRAIN_DATA}"
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "Checkpoints will be written to: ${SAVE_DIR}"
echo "Note: GPU count is detected automatically; memory fit is not guaranteed. Tune batch/sequence settings for your pod."

if [[ "${CONFIRM_RUN:-NO}" != "YES" ]]; then
    echo "Preflight passed. To start training, rerun with CONFIRM_RUN=YES."
    exit 0
fi

RAY_GCS_PORT="${RAY_GCS_PORT:-6380}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
RAY_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"

if [[ "${RUNPOD_REUSE_RAY:-NO}" == "YES" ]]; then
    if ! curl --fail --silent --max-time 5 "${RAY_ADDRESS}/api/jobs/" >/dev/null; then
        echo "Cannot reuse Ray: the jobs API is unavailable at ${RAY_ADDRESS}." >&2
        exit 1
    fi
    echo "Submitting to the existing Ray head at ${RAY_ADDRESS}."
else
    if curl --fail --silent --max-time 5 "${RAY_ADDRESS}/api/version" >/dev/null; then
        echo "Ray is already running at ${RAY_ADDRESS}; set RUNPOD_REUSE_RAY=YES to use that head." >&2
        exit 1
    fi

    # Ray appends a session directory and Unix socket filenames to this path.
    if [[ -z "${RAY_TMPDIR:-}" ]]; then
        RAY_TMPDIR="$(mktemp -d /tmp/ray.XXXXXX)"
    else
        mkdir -p "${RAY_TMPDIR}"
    fi
    export RAY_TMPDIR
    RAY_ARCHIVE_REQUIRED=YES

    "${RUNPOD_RAY}" start --head \
        --node-ip-address 127.0.0.1 \
        --port "${RAY_GCS_PORT}" \
        --num-gpus "${GPU_COUNT}" \
        --disable-usage-stats \
        --dashboard-host 127.0.0.1 \
        --dashboard-port "${RAY_DASHBOARD_PORT}" \
        --temp-dir "${RAY_TMPDIR}"
    for _ in $(seq 1 60); do
        if curl --fail --silent "${RAY_ADDRESS}/api/version" >/dev/null; then
            break
        fi
        sleep 1
    done
    if ! curl --fail --silent "${RAY_ADDRESS}/api/version" >/dev/null; then
        echo "Ray dashboard did not become ready at ${RAY_ADDRESS}." >&2
        exit 1
    fi
fi

BASE_PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="${MEGATRON_PATH}:${SLIME_ROOT}${BASE_PYTHONPATH:+:${BASE_PYTHONPATH}}"
RUNTIME_ENV_JSON="$(MEGATRON_PATH="${MEGATRON_PATH}" SLIME_ROOT="${SLIME_ROOT}" BASE_PYTHONPATH="${BASE_PYTHONPATH}" "${RUNPOD_PYTHON}" - <<'PY'
import json
import os

pythonpath = ":".join(
    value for value in (
        os.environ["MEGATRON_PATH"],
        os.environ["SLIME_ROOT"],
        os.environ.get("BASE_PYTHONPATH", ""),
    ) if value
)
print(json.dumps({
    "env_vars": {
        "PYTHONPATH": pythonpath,
        "PATH": os.environ["PATH"],
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
    }
}))
PY
)"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --ref-load "${REF_LOAD}"
    --save "${SAVE_DIR}"
    --save-interval "${SAVE_INTERVAL:-10}"
)
if [[ -n "${RESUME_FROM:-}" ]]; then
    CKPT_ARGS+=(--load "${RESUME_FROM}")
fi

ROLLOUT_ARGS=(
    --guard-service-url "${GUARD_SERVICE_URL}"
    --enable-noresp
    --enable-partial-mask
    --buffer-max-length 256
    --enable-partialrollout
    --enable-safety-thinking
    --prompt-data "${TRAIN_DATA}"
    --input-key prompt
    --label-key label
    --safety-stream-model-path "${SAFETY_STREAM_MODEL_PATH}"
    --safety-guard-model-path "${SAFETY_GUARD_MODEL_PATH}"
    --llama-guard-model-path "${LLAMA_GUARD_MODEL_PATH}"
    --use-llama-guard
    --rollout-function-path slime.rollout.partialrollout.generate_rollout
    --apply-chat-template
    --rollout-shuffle
    --rm-type deepscaler
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-64}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-16}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
    --rollout-temperature 0.7
    --rollout-top-p 1.0
    --rollout-top-k -1
    --rollout-repetition-penalty 1.0
    --over-sampling-batch-size "${OVERSAMPLING_BATCH_SIZE:-64}"
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --global-batch-size "${GLOBAL_BATCH_SIZE:-1024}"
    --balance-data
)

PERF_ARGS=(
    --tensor-model-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-9216}"
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${TENSOR_PARALLEL_SIZE}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.7}"
)
if [[ "${RUNPOD_TRAIN_TRIAL:-NO}" == "YES" ]]; then
    OPTIMIZER_ARGS+=(--optimizer-cpu-offload --use-precision-aware-optimizer)
    PERF_ARGS+=(--no-offload-train)
    SGLANG_ARGS+=(
        --sglang-max-total-tokens 4096
        --sglang-disable-cuda-graph --sglang-disable-piecewise-cuda-graph
    )
fi

EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL:-5}"
    --eval-prompt-data wildjailbreak "${EVAL_WILDJAILBREAK}" XSTest "${EVAL_XSTEST}"
    --eval-function-path slime.rollout.partialrollout.generate_rollout
    --n-samples-per-eval-prompt 2
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-8000}"
    --eval-top-p 1.0
    --eval-temperature 0.0
    --eval-top-k -1
    --eval-repetition-penalty 1.2
)
if [[ -n "${REUSE_BASELINE_LOG:-}" ]]; then
    EVAL_ARGS+=(--reuse-baseline-log "${REUSE_BASELINE_LOG}")
fi

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

cd "${SLIME_ROOT}"
"${RUNPOD_RAY}" job submit --address="${RAY_ADDRESS}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${RUNPOD_PYTHON}" "${SLIME_ROOT}/train_epoch.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${GPU_COUNT}" \
    --num-gpus-per-node "${GPU_COUNT}" \
    --colocate \
    --num-workers "${NUM_WORKERS:-2}" \
    --keep-only-last-checkpoint \
    "${RUN_LENGTH_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    --custom-rm-path slime.rollout.rm_hub.safety_rm.reward_func \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${MISC_ARGS[@]}"
