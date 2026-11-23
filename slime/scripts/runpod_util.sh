#!/usr/bin/env bash

runpod_environment_ready() {
    (
        cd "${SLIME_ROOT}" || exit
        "${RUNPOD_PYTHON}" - <<'PY'
import torch
import ray
import flash_attn
import huggingface_hub
import mbridge
import slime
import datasets
import fastapi
import httpx
import omegaconf
import pylatexenc
import transformers
import uvicorn
import transformer_engine.pytorch
import torch_memory_saver
import megatron.training.arguments
import sglang_router.launch_router
import os
from pathlib import Path
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

if not Path(slime.__file__).resolve().is_relative_to(Path(os.environ["SLIME_ROOT"]).resolve()):
    raise SystemExit(f"Wrong Slime source loaded: {slime.__file__}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA is unavailable; check the NVIDIA driver and GPU.")
print(f"Environment verified: torch={torch.__version__}, CUDA={torch.version.cuda}, ray={ray.__version__}")
PY
    )
}

runpod_ensure_environment() {
    MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
    if [[ -n "${RUNPOD_VENV:-}" ]]; then
        RUNPOD_PYTHON="${RUNPOD_PYTHON:-${RUNPOD_VENV}/bin/python3}"
        RUNPOD_RAY="${RUNPOD_RAY:-${RUNPOD_VENV}/bin/ray}"
        export PATH="${RUNPOD_VENV}/bin:${PATH}"
        if [[ -d "${RUNPOD_VENV}/conda-meta" ]]; then
            export CUDA_HOME="${RUNPOD_VENV}"
            export CC="${CC:-${RUNPOD_VENV}/bin/x86_64-conda-linux-gnu-gcc}"
            export CXX="${CXX:-${RUNPOD_VENV}/bin/x86_64-conda-linux-gnu-c++}"
            export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX}}"
            export LIBRARY_PATH="${RUNPOD_VENV}/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"
            export CPATH="${RUNPOD_VENV}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
            export LD_LIBRARY_PATH="${RUNPOD_VENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        fi
    else
        RUNPOD_PYTHON="${RUNPOD_PYTHON:-/usr/bin/python3.12}"
        RUNPOD_RAY="${RUNPOD_RAY:-/usr/local/bin/ray}"
        export PATH="/usr/bin:/usr/local/bin:/usr/local/cuda/bin:${PATH}"
    fi
    export RUNPOD_PYTHON RUNPOD_RAY MEGATRON_PATH SLIME_ROOT
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    if [[ -z "${RUNPOD_VENV:-}" ]]; then
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    export PYTHONPATH="${SLIME_ROOT}:${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
    if [[ ! -x "${RUNPOD_PYTHON}" || ! -x "${RUNPOD_RAY}" || ! -d "${MEGATRON_PATH}/megatron" ]]; then
        echo "Prebuilt environment not found: python=${RUNPOD_PYTHON}, ray=${RUNPOD_RAY}, Megatron=${MEGATRON_PATH}." >&2
        echo "Use the slimerl/slime:nightly-dev-20260430b image, or explicitly configure RUNPOD_VENV, RUNPOD_PYTHON, RUNPOD_RAY and MEGATRON_PATH." >&2
        return 1
    fi
    if ! runpod_environment_ready; then
        echo "Prebuilt environment validation failed; see the error above. No GPU packages were installed or compiled." >&2
        return 1
    fi
}

runpod_ensure_guard_environment() {
    local guard_venv="${GUARD_VENV:-/workspace/.venvs/self-reset-guard}"
    local base_prefix base_env_prefix base_version base_torch
    GUARD_PYTHON="${guard_venv}/bin/python3"
    if [[ ! -x "${GUARD_PYTHON}" ]]; then
        if [[ -e "${guard_venv}" ]]; then
            echo "Guard environment exists but has no Python interpreter: ${guard_venv}" >&2
            return 1
        fi
        echo "Creating isolated guard environment at ${guard_venv}."
        "${RUNPOD_PYTHON}" -m venv --system-site-packages "${guard_venv}"
    fi
    base_prefix="$("${RUNPOD_PYTHON}" -c 'import sys; print(sys.base_prefix)')"
    base_env_prefix="$("${RUNPOD_PYTHON}" -c 'import sys; print(sys.prefix)')"
    base_version="$("${RUNPOD_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    base_torch="$("${RUNPOD_PYTHON}" -c 'import torch; print(torch.__file__)')"
    PYTHONNOUSERSITE=1 "${GUARD_PYTHON}" - "${base_prefix}" "${base_env_prefix}" "${base_version}" "${guard_venv}" <<'PY'
import os
from pathlib import Path
import sys

base_prefix, base_env_prefix, base_version, guard_venv = sys.argv[1:]
config = (Path(guard_venv) / "pyvenv.cfg").read_text().lower().splitlines()
uses_system_packages = any(
    line.split("=", 1)[0].strip() == "include-system-site-packages"
    and line.split("=", 1)[1].strip() == "true"
    for line in config if "=" in line
)
version = f"{sys.version_info.major}.{sys.version_info.minor}"
if (
    sys.prefix == sys.base_prefix
    or os.path.realpath(sys.base_prefix) != os.path.realpath(base_prefix)
    or os.path.realpath(sys.prefix) == os.path.realpath(base_env_prefix)
    or os.path.realpath(sys.prefix) != os.path.realpath(guard_venv)
    or version != base_version
    or not uses_system_packages
):
    raise SystemExit("Guard venv is incompatible with the selected image Python; use a fresh GUARD_VENV.")
PY
    if ! PYTHONNOUSERSITE=1 "${GUARD_PYTHON}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
import sys

try:
    installed = version("transformers")
except PackageNotFoundError:
    print("Guard Transformers is not installed yet.")
    raise SystemExit(1)
spec = find_spec("transformers")
if (
    installed != "4.55.0"
    or spec is None
    or spec.origin is None
    or not Path(spec.origin).resolve().is_relative_to(Path(sys.prefix).resolve())
):
    print("Guard Transformers is missing, incompatible, or outside the guard venv.")
    raise SystemExit(1)
PY
    then
        echo "Installing guard-only dependencies into ${guard_venv}."
        PYTHONNOUSERSITE=1 "${GUARD_PYTHON}" -m pip --isolated install --only-binary=:all: \
            --force-reinstall --no-deps -r "${SLIME_ROOT}/guard-requirements.txt"
        PYTHONNOUSERSITE=1 "${GUARD_PYTHON}" -m pip --isolated install --only-binary=:all: \
            -r "${SLIME_ROOT}/guard-requirements.txt"
    fi
    PYTHONNOUSERSITE=1 "${GUARD_PYTHON}" - "${base_torch}" <<'PY'
import os
from pathlib import Path
import sys
import torch
import transformers
from transformers.utils.generic import check_model_inputs
from slime.rollout.Guardforward import ContentEvaluator, GenLabelEvaluator
from slime.services.guard_service import app

if transformers.__version__ != "4.55.0":
    raise SystemExit(f"Unexpected guard Transformers version: {transformers.__version__}")
if not Path(transformers.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
    raise SystemExit("Guard Transformers must be installed inside the guard venv.")
if os.path.realpath(torch.__file__) != os.path.realpath(sys.argv[1]):
    raise SystemExit("Guard environment must use the image's existing PyTorch installation.")
print(f"Guard environment verified: transformers={transformers.__version__}, torch={torch.__version__}")
PY
}

runpod_set_model_paths() {
    RUNPOD_MODEL_DIR="${RUNPOD_MODEL_DIR:-/workspace/models}"
    local default_base_model_id
    case "${MODEL_CONFIG:-qwen3-8B}" in
        qwen3-8B) default_base_model_id="Qwen/Qwen3-8B" ;;
        qwen2.5-7B) default_base_model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" ;;
        llama3.1-8B-Instruct) default_base_model_id="meta-llama/Llama-3.1-8B-Instruct" ;;
        *)
            echo "Unsupported MODEL_CONFIG '${MODEL_CONFIG}'." >&2
            return 1
            ;;
    esac
    HF_BASE_MODEL_ID="${HF_BASE_MODEL_ID:-${default_base_model_id}}"
    HF_BASE_MODEL_REVISION="${HF_BASE_MODEL_REVISION:-main}"
    QWEN3GUARD_STREAM_MODEL_ID="${QWEN3GUARD_STREAM_MODEL_ID:-Qwen/Qwen3Guard-Stream-8B}"
    QWEN3GUARD_STREAM_MODEL_REVISION="${QWEN3GUARD_STREAM_MODEL_REVISION:-main}"
    QWEN3GUARD_GEN_MODEL_ID="${QWEN3GUARD_GEN_MODEL_ID:-Qwen/Qwen3Guard-Gen-8B}"
    QWEN3GUARD_GEN_MODEL_REVISION="${QWEN3GUARD_GEN_MODEL_REVISION:-main}"
    LLAMA_GUARD_MODEL_ID="${LLAMA_GUARD_MODEL_ID:-meta-llama/Llama-Guard-3-8B}"
    LLAMA_GUARD_MODEL_REVISION="${LLAMA_GUARD_MODEL_REVISION:-main}"

    local base_model_name="${HF_BASE_MODEL_ID##*/}"
    HF_CHECKPOINT="${HF_CHECKPOINT:-${RUNPOD_MODEL_DIR}/${base_model_name}}"
    REF_LOAD="${REF_LOAD:-/workspace/checkpoints/${base_model_name}_torch_dist}"
    SAVE_DIR="${SAVE_DIR:-/workspace/checkpoints/Self-RESET}"
    SAFETY_STREAM_MODEL_PATH="${SAFETY_STREAM_MODEL_PATH:-${RUNPOD_MODEL_DIR}/Qwen3Guard-Stream-8B}"
    SAFETY_GUARD_MODEL_PATH="${SAFETY_GUARD_MODEL_PATH:-${RUNPOD_MODEL_DIR}/Qwen3Guard-Gen-8B}"
    LLAMA_GUARD_MODEL_PATH="${LLAMA_GUARD_MODEL_PATH:-${RUNPOD_MODEL_DIR}/Llama-Guard-3-8B}"

    export RUNPOD_MODEL_DIR HF_BASE_MODEL_ID HF_BASE_MODEL_REVISION
    export QWEN3GUARD_STREAM_MODEL_ID QWEN3GUARD_STREAM_MODEL_REVISION
    export QWEN3GUARD_GEN_MODEL_ID QWEN3GUARD_GEN_MODEL_REVISION
    export LLAMA_GUARD_MODEL_ID LLAMA_GUARD_MODEL_REVISION
    export HF_CHECKPOINT REF_LOAD SAVE_DIR
    export SAFETY_STREAM_MODEL_PATH SAFETY_GUARD_MODEL_PATH LLAMA_GUARD_MODEL_PATH
}

runpod_ensure_hf_models() {
    local include_base_model="$1"
    local include_guard_models="$2"
    local prepare_reference_checkpoint="$3"
    local -a model_args=()

    if [[ "${include_base_model}" == "YES" ]]; then
        model_args+=(
            "${HF_CHECKPOINT}" "${HF_BASE_MODEL_ID}" "${HF_BASE_MODEL_REVISION}"
        )
    fi
    if [[ "${include_guard_models}" == "YES" ]]; then
        model_args+=(
            "${SAFETY_STREAM_MODEL_PATH}" "${QWEN3GUARD_STREAM_MODEL_ID}" "${QWEN3GUARD_STREAM_MODEL_REVISION}"
            "${SAFETY_GUARD_MODEL_PATH}" "${QWEN3GUARD_GEN_MODEL_ID}" "${QWEN3GUARD_GEN_MODEL_REVISION}"
            "${LLAMA_GUARD_MODEL_PATH}" "${LLAMA_GUARD_MODEL_ID}" "${LLAMA_GUARD_MODEL_REVISION}"
        )
    fi

    local reference_dir=""
    if [[ "${prepare_reference_checkpoint}" == "YES" ]]; then
        reference_dir="${REF_LOAD}"
    fi

    "${RUNPOD_PYTHON}" - "${RUNPOD_MODEL_SPACE_RESERVE_GB:-10}" "${reference_dir}" \
        "${HF_CHECKPOINT}" "${HF_BASE_MODEL_ID}" "${HF_BASE_MODEL_REVISION}" \
        "${model_args[@]}" <<'PY'
import fnmatch
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

allow_patterns = (
    "*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken",
    "*.jinja", "*.py", "*.md", "LICENSE",
)


def has_complete_weights(folder: Path) -> bool:
    if not (folder / "config.json").is_file():
        return False
    has_tokenizer = any(
        (folder / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    )
    if not has_tokenizer:
        return False
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = folder / index_name
        if index_path.is_file():
            try:
                weight_map = json.loads(index_path.read_text())["weight_map"]
            except (OSError, KeyError, json.JSONDecodeError):
                return False
            return bool(weight_map) and all((folder / name).is_file() for name in set(weight_map.values()))
    return any(folder.glob("*.safetensors")) or any(folder.glob("pytorch_model*.bin"))


def file_size(file) -> int:
    size = getattr(file, "size", None)
    if size is None:
        lfs = getattr(file, "lfs", None)
        size = getattr(lfs, "size", None) if lfs is not None else None
    return int(size or 0)


reserve_gib = float(sys.argv[1])
reference_dir = Path(sys.argv[2]) if sys.argv[2] else None
base_dir = Path(sys.argv[3])
base_repo, base_revision = sys.argv[4:6]
args = sys.argv[6:]
if len(args) % 3:
    raise SystemExit("Internal error: model paths must be supplied as path/repository/revision triples.")

models = [
    (Path(args[index]), args[index + 1], args[index + 2])
    for index in range(0, len(args), 3)
]
missing = [(path, repo, revision) for path, repo, revision in models if not has_complete_weights(path)]
need_reference = reference_dir is not None and not (reference_dir / "latest_checkpointed_iteration.txt").is_file()
if not missing and not need_reference:
    for path, _, _ in models:
        print(f"Model already present: {path}")
    raise SystemExit(0)

api = HfApi()
metadata = {}


def get_metadata(repo: str, revision: str):
    key = (repo, revision)
    if key not in metadata:
        try:
            hf_hub_download(repo_id=repo, filename="config.json", revision=revision)
            metadata[key] = api.model_info(repo, revision=revision, files_metadata=True).siblings
        except GatedRepoError as exc:
            raise SystemExit(
                f"Access to {repo} is gated. Accept its terms on Hugging Face and set HF_TOKEN "
                "in the pod environment before retrying."
            ) from exc
    return metadata[key]


def estimated_sizes(repo: str, revision: str) -> dict[str, int]:
    return {
        item.rfilename: file_size(item)
        for item in get_metadata(repo, revision)
        if any(fnmatch.fnmatchcase(item.rfilename, pattern) for pattern in allow_patterns)
    }


requirements: dict[int, tuple[int, Path]] = {}


def add_requirement(path: Path, size: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    device = path.parent.stat().st_dev
    required, device_path = requirements.get(device, (0, path.parent))
    requirements[device] = (required + size, device_path)


for target, repo, revision in missing:
    remote_sizes = estimated_sizes(repo, revision)
    if not remote_sizes or sum(remote_sizes.values()) == 0:
        raise SystemExit(f"Could not estimate downloadable files for {repo}; refusing an unbounded download.")
    present_bytes = sum(
        file_size for name, file_size in remote_sizes.items()
        if (target / name).is_file()
    )
    add_requirement(target, max(0, sum(remote_sizes.values()) - present_bytes))
    print(f"Model {repo}: up to {sum(remote_sizes.values()) / 1024**3:.2f} GiB to download.")

if need_reference:
    if has_complete_weights(base_dir):
        base_weight_bytes = sum(path.stat().st_size for path in base_dir.glob("*.safetensors"))
        if base_weight_bytes == 0:
            base_weight_bytes = sum(path.stat().st_size for path in base_dir.glob("pytorch_model*.bin"))
    else:
        base_weight_bytes = sum(
            size for name, size in estimated_sizes(base_repo, base_revision).items()
            if name.endswith((".safetensors", ".bin"))
        )
    add_requirement(reference_dir, base_weight_bytes)
    print(f"Converted reference checkpoint: reserving about {base_weight_bytes / 1024**3:.2f} GiB.")

for device, (required, path) in requirements.items():
    required += int(reserve_gib * 1024**3)
    free = shutil.disk_usage(path).free
    if free < required:
        raise SystemExit(
            f"Insufficient free space on the volume containing {path}: "
            f"need about {required / 1024**3:.1f} GiB including reserve, "
            f"have {free / 1024**3:.1f} GiB. Increase pod storage before downloading."
        )

for target, repo, revision in missing:
    print(f"Downloading {repo} ({revision}) to {target}.")
    try:
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=str(target),
            allow_patterns=allow_patterns,
        )
    except GatedRepoError as exc:
        raise SystemExit(
            f"Access to {repo} is gated. Accept its terms on Hugging Face and set HF_TOKEN "
            "in the pod environment before retrying."
        ) from exc
    if not has_complete_weights(target):
        raise SystemExit(f"Model download is incomplete: {target}")
PY
}

runpod_create_reference_checkpoint() {
    if [[ -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]]; then
        echo "Megatron reference checkpoint already exists: ${REF_LOAD}"
        return
    fi
    if [[ "${AUTO_CONVERT_CHECKPOINT:-YES}" != "YES" ]]; then
        echo "Converted checkpoint is missing; set AUTO_CONVERT_CHECKPOINT=YES or provide REF_LOAD." >&2
        return 1
    fi
    if [[ -e "${REF_LOAD}" ]]; then
        if [[ ! -d "${REF_LOAD}" ]] || [[ -n "$(find "${REF_LOAD}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
            echo "REF_LOAD exists but is incomplete: ${REF_LOAD}" >&2
            echo "Move the incomplete directory aside before retrying conversion." >&2
            return 1
        fi
    fi

    mkdir -p "$(dirname -- "${REF_LOAD}")"
    echo "Converting ${HF_CHECKPOINT} to a Megatron reference checkpoint at ${REF_LOAD}."
    (
        cd "${SLIME_ROOT}"
        PYTHONPATH="${MEGATRON_PATH}:${SLIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            "${RUNPOD_PYTHON}" tools/convert_hf_to_torch_dist.py \
            --hf-checkpoint "${HF_CHECKPOINT}" \
            --save "${REF_LOAD}" \
            "${MODEL_ARGS[@]}"
    )

    if [[ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]]; then
        echo "Checkpoint conversion finished without creating the expected tracker file in ${REF_LOAD}." >&2
        return 1
    fi
}

runpod_sha256() {
    "${RUNPOD_PYTHON}" - "$1" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

runpod_ensure_data_file() {
    local variable_name="$1"
    local file_name="$2"
    local source_relative_path="$3"
    local expected_sha256="$4"
    local repository_commit="3d2c4c64429a4f4063d7b78eaa758020bdb212c8"
    local data_dir="${RUNPOD_DATA_DIR:-/workspace/data/self-reset}"
    local destination="${data_dir}/${file_name}"
    local source_file="${PROJECT_ROOT}/${source_relative_path}"
    local download_url="https://raw.githubusercontent.com/Ing1024/Self-ReSET/${repository_commit}/${source_relative_path}"
    local configured_path="${!variable_name:-}"
    local temporary_file
    local actual_sha256

    if [[ -n "${configured_path}" ]]; then
        if [[ ! -f "${configured_path}" ]]; then
            echo "${variable_name} does not exist: ${configured_path}" >&2
            return 1
        fi
        return 0
    fi

    mkdir -p "${data_dir}"
    if [[ -e "${destination}" ]]; then
        if [[ ! -f "${destination}" ]]; then
            echo "Expected a data file at ${destination}" >&2
            return 1
        fi
        actual_sha256="$(runpod_sha256 "${destination}")"
        if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
            echo "Cached dataset checksum mismatch: ${destination}" >&2
            echo "Move it aside or set ${variable_name} to a different file." >&2
            return 1
        fi
    else
        temporary_file="$(mktemp "${destination}.tmp.XXXXXX")"
        if [[ -f "${source_file}" ]] && [[ "$(runpod_sha256 "${source_file}")" == "${expected_sha256}" ]]; then
            cp -- "${source_file}" "${temporary_file}"
        elif ! curl --fail --location --silent --show-error --retry 3 \
            "${download_url}" --output "${temporary_file}"; then
            rm -f -- "${temporary_file}"
            echo "Failed to download dataset: ${download_url}" >&2
            return 1
        fi

        actual_sha256="$(runpod_sha256 "${temporary_file}")"
        if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
            rm -f -- "${temporary_file}"
            echo "Downloaded dataset checksum mismatch: ${file_name}" >&2
            return 1
        fi
        mv -- "${temporary_file}" "${destination}"
        echo "Prepared dataset: ${destination}"
    fi

    printf -v "${variable_name}" '%s' "${destination}"
    export "${variable_name}"
}

runpod_ensure_data() {
    RUNPOD_DATA_DIR="${RUNPOD_DATA_DIR:-/workspace/data/self-reset}"
    command -v curl >/dev/null || {
        echo "curl is required to fetch datasets missing from the workspace." >&2
        return 1
    }

    runpod_ensure_data_file \
        TRAIN_DATA \
        vanilla_filter_balanced_1500.jsonl \
        data/training/vanilla_filter_balanced_1500.jsonl \
        8d9cdc01e182a3e432bce2452e8bf3704bd42fe16960d94babaa8752c74e7d6a
    runpod_ensure_data_file \
        EVAL_WILDJAILBREAK \
        wildjailbreak.jsonl \
        data/evaluation/safety/wildjailbreak.jsonl \
        14f527ecc552c47ebe53a86f96ec105d294531aa04856df039f8622e058e5065
    runpod_ensure_data_file \
        EVAL_XSTEST \
        XSTest.jsonl \
        data/evaluation/overrefusal/XSTest.jsonl \
        58b30500fd092962617f7383cbbdfb1721c95f11b323abbfb393fdf4561c5ebf
}
