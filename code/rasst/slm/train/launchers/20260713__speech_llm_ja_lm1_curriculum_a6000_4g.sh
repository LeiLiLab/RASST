#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WRAPPER="${ROOT_DIR}/slm/train/auto_train_sampling_docker.sh"
RUN_ROOT="${3:-}"
COMPUTE_LABEL="${4:-}"
DATASET_PATH="${RUN_ROOT}/data/train_s_ja_cap16_denoise_ttag_lm1x2_seed43.jsonl"
VAL_DATASET="/mnt/gemini/data1/jiaxuanluo/speech_llm_ja_cap16_denoise_budget_20260525/ja/hn1024_tau078_cap16_denoise_budget_ttag_v1/dev_s_ja_retriever_hn1024_tau078_cap16_denoise_budget_ttag_exactboundary_first355.jsonl"
MCORE_MODEL="/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-v2/"
BASE_MODEL_HOST="/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct"
SAVE_BASE="${RUN_ROOT}/checkpoints"
TRAIN_LOG_DIR="${RUN_ROOT}/logs/train"
WANDB_DIR="${RUN_ROOT}/wandb"
HF_EXPORT_STAGE_ROOT="none"
HF_EXPORT_LOCAL_CACHE_ROOT="none"
HF_EXPORT_LOCAL_LATEST_LINK="none"
NOTES_FILE="${ROOT_DIR}/../../docs/provenance/slm/20260713__speech_llm_ja_lm1_curriculum_r32a32_ep1_a6000_4g.md"

ALLOCATED_GPUS="${1:-}"
CONTAINER_NAME="${2:-}"
if [[ -z "${ALLOCATED_GPUS}" || -z "${CONTAINER_NAME}" || -z "${RUN_ROOT}" || -z "${COMPUTE_LABEL}" ]]; then
  echo "Usage: $0 <gpu_csv> <container_name> <absolute_run_root> <compute_label>" >&2
  exit 2
fi
if [[ "${RUN_ROOT}" != /* ]]; then
  echo "[ERROR] run_root must be absolute: ${RUN_ROOT}" >&2
  exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "${ALLOCATED_GPUS}"
if (( ${#GPU_ARRAY[@]} != 4 )); then
  echo "[ERROR] Expected exactly four Taurus GPUs, got: ${ALLOCATED_GPUS}" >&2
  exit 2
fi
if [[ ! "${CONTAINER_NAME}" =~ ^sglang-omni-jaxan-[0-9]{8}$ ]]; then
  echo "[ERROR] Container name must match sglang-omni-jaxan-MMDDHHMM" >&2
  exit 2
fi
LOCAL_RUN_MOUNT="$(df --output=target "${RUN_ROOT}" | tail -n 1 | xargs)"
if [[ -z "${LOCAL_RUN_MOUNT}" || "${LOCAL_RUN_MOUNT}" != /* ]]; then
  echo "[ERROR] Could not resolve filesystem mount for run_root: ${RUN_ROOT}" >&2
  exit 2
fi

for path in "${WRAPPER}" "${DATASET_PATH}" "${VAL_DATASET}" "${MCORE_MODEL}" "${BASE_MODEL_HOST}" "${NOTES_FILE}"; do
  [[ -e "${path}" ]] || { echo "[ERROR] Missing required path: ${path}" >&2; exit 3; }
done

mkdir -p "${SAVE_BASE}" "${TRAIN_LOG_DIR}" "${WANDB_DIR}"

# note (luojiaxuan): The legacy Megatron/Swift wrapper receives its complete,
# Git-tracked recipe through explicit assignments here. LOCAL_RUN_MOUNT keeps
# host-qualified /mnt/<host>/data* symlinks valid inside the training container.
BASE_MODEL_HOST="${BASE_MODEL_HOST}" \
HOST_GPU_DEVICES="${ALLOCATED_GPUS}" \
DOCKER_CONTAINER_NAME="${CONTAINER_NAME}" \
MOUNT_ROOTS="/mnt/gemini /mnt/taurus /mnt/aries ${LOCAL_RUN_MOUNT}" \
NPROC_PER_NODE=4 \
EXPERT_MODEL_PARALLEL_SIZE=2 \
TENSOR_MODEL_PARALLEL_SIZE=2 \
SEQUENCE_PARALLEL=true \
KEEP_RATIO="" \
DATASET_PATH="${DATASET_PATH}" \
VAL_DATASET="${VAL_DATASET}" \
SAVE_BASE="${SAVE_BASE}" \
TRAIN_LOG_DIR="${TRAIN_LOG_DIR}" \
MCORE_MODEL="${MCORE_MODEL}" \
LORA_RANK=32 \
LORA_ALPHA=32 \
MAX_EPOCHS=1 \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=4 \
MAX_LENGTH=3072 \
HF_EXPORT_STAGE_ROOT="${HF_EXPORT_STAGE_ROOT}" \
HF_EXPORT_MIN_FREE_GB=70 \
HF_EXPORT_LOCAL_CACHE_ROOT="${HF_EXPORT_LOCAL_CACHE_ROOT}" \
HF_EXPORT_LOCAL_LATEST_LINK="${HF_EXPORT_LOCAL_LATEST_LINK}" \
HF_EXPORT_SWIFT_EXTRA_ARGS="--device_map auto" \
ITERATIONS_PER_EPOCH=560 \
SAVE_INTERVAL=560 \
MASTER_PORT=29743 \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
NCCL_DEBUG=INFO \
TORCH_NCCL_ENABLE_MONITORING=0 \
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
WANDB_PROJECT="sst_omni" \
WANDB_MODE="offline" \
WANDB_DIR="${WANDB_DIR}" \
WANDB_EXP_PREFIX="speech-llm-ja-lm1-curriculum-r32a32-ep1-${COMPUTE_LABEL}" \
WANDB_TAGS="family:speech_llm_tcm_termmap,task:train,data:ja_lm1_curriculum,variant:lm1x2_seed43,status:running,compute:${COMPUTE_LABEL}" \
WANDB_NOTES="$(<"${NOTES_FILE}")" \
ENABLE_MD_UPDATE=0 \
bash "${WRAPPER}"
