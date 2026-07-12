#!/usr/bin/env bash
#SBATCH --job-name=ms_e2e_lm2
#SBATCH --partition=aries
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --chdir=/mnt/taurus/data2/jiaxuanluo/RASST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_ms_e2e_lm2.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_ms_e2e_lm2.err

set -euo pipefail

# End-to-end counterpart to the paper's multi-scale retrieval ablation.
# It holds the released En-Zh tagged-ACL LM=2 protocol fixed and changes only
# the retriever architecture/window policy.

RASST_ROOT="${RASST_ROOT_OVERRIDE:-/mnt/taurus/data2/jiaxuanluo/RASST}"
ACTIVE_ROOT="${RASST_ROOT}/code/rasst"
EVAL_SCRIPT="${ACTIVE_ROOT}/eval/eval_density_unified.sh"
VARIANT="${VARIANT_OVERRIDE:-largest_train}"
SCOPE="${SCOPE_OVERRIDE:-full}"
GPU_PAIR="${GPU_PAIR_OVERRIDE:-2,3}"

RUN_ROOT="${RUN_ROOT_OVERRIDE:-/mnt/aries/data6/jiaxuanluo/rasst_multiscale_e2e_20260711}"
OUTPUT_BASE="${RUN_ROOT}/${SCOPE}/${VARIANT}"
INPUT_ROOT="/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_zh_lm23_20260524T0522_tagacl_newv9_hn1024_tau078_raw_zh_lm23_aries4567/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/zh/all"
MODEL_NAME="/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/models/speech_llm_zh_cap16_denoise_budget_ttag_r32a32_ep1_taurus4_hf"
RAW_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json"
MULTISCALE_CKPT="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt"
DENSE_CKPT="/mnt/aries/data4/jiaxuanluo/train_outputs/sweep_text_pooling/sweep_tp_cls_tfpool_bs12k_best.pt"

fail() { echo "[ERROR] $*" >&2; exit 3; }
require_file() { [[ -s "$1" ]] || fail "Missing/empty required file: $1"; }

case "${VARIANT}" in
  multiscale)
    RAG_MODEL_PATH="${MULTISCALE_CKPT}"
    RAG_USE_MAXSIM=1
    RAG_MAXSIM_WINDOWS="2 3 4 5 6 7 8 10 12 16 20 24"
    RAG_TIMELINE_LOOKBACK_SEC=1.92
    ;;
  largest_infer)
    RAG_MODEL_PATH="${MULTISCALE_CKPT}"
    RAG_USE_MAXSIM=1
    RAG_MAXSIM_WINDOWS="24"
    RAG_TIMELINE_LOOKBACK_SEC=1.92
    ;;
  largest_train)
    RAG_MODEL_PATH="${DENSE_CKPT}"
    RAG_USE_MAXSIM=0
    RAG_MAXSIM_WINDOWS="24"
    # LM=2 is exactly 1.92 seconds, matching this checkpoint's train context.
    RAG_TIMELINE_LOOKBACK_SEC=0.0
    ;;
  *)
    fail "VARIANT_OVERRIDE must be multiscale, largest_infer, or largest_train; got ${VARIANT}"
    ;;
esac

for path in \
  "${EVAL_SCRIPT}" "${MODEL_NAME}/config.json" "${RAW_GLOSSARY}" \
  "${RAG_MODEL_PATH}" "${INPUT_ROOT}/source.list" "${INPUT_ROOT}/target.list" \
  "${INPUT_ROOT}/ref.txt" "${INPUT_ROOT}/source_text.txt" "${INPUT_ROOT}/audio.yaml"; do
  require_file "${path}"
done

SRC_LIST="${INPUT_ROOT}/source.list"
TGT_LIST="${INPUT_ROOT}/target.list"
REF_FILE="${INPUT_ROOT}/ref.txt"
SOURCE_TEXT_FILE="${INPUT_ROOT}/source_text.txt"
AUDIO_YAML="${INPUT_ROOT}/audio.yaml"

if [[ "${SCOPE}" == "smoke" ]]; then
  SMOKE_INPUTS="${RUN_ROOT}/smoke_inputs/first_talk"
  mkdir -p "${SMOKE_INPUTS}"
  INPUT_ROOT="${INPUT_ROOT}" SMOKE_INPUTS="${SMOKE_INPUTS}" \
    /mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python - <<'PY'
import os
from pathlib import Path
import yaml

src = Path(os.environ["INPUT_ROOT"])
dst = Path(os.environ["SMOKE_INPUTS"])
source_paths = src.joinpath("source.list").read_text(encoding="utf-8").splitlines()
target_lines = src.joinpath("target.list").read_text(encoding="utf-8").splitlines()
paper_stem = Path(source_paths[0]).stem
audio = yaml.safe_load(src.joinpath("audio.yaml").read_text(encoding="utf-8"))
refs = src.joinpath("ref.txt").read_text(encoding="utf-8").splitlines()
source_text = src.joinpath("source_text.txt").read_text(encoding="utf-8").splitlines()
if not (len(audio) == len(refs) == len(source_text)):
    raise SystemExit("audio/ref/source_text alignment mismatch")
indices = [i for i, row in enumerate(audio) if Path(str(row["wav"])).stem == paper_stem]
if not indices:
    raise SystemExit(f"no sentence rows for {paper_stem}")
dst.joinpath("source.list").write_text(source_paths[0] + "\n", encoding="utf-8")
dst.joinpath("target.list").write_text(target_lines[0] + "\n", encoding="utf-8")
dst.joinpath("ref.txt").write_text("\n".join(refs[i] for i in indices) + "\n", encoding="utf-8")
dst.joinpath("source_text.txt").write_text(
    "\n".join(source_text[i] for i in indices) + "\n", encoding="utf-8"
)
dst.joinpath("audio.yaml").write_text(
    yaml.safe_dump([audio[i] for i in indices], allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)
print(f"[SMOKE] paper={paper_stem} sentences={len(indices)}")
PY
  SRC_LIST="${SMOKE_INPUTS}/source.list"
  TGT_LIST="${SMOKE_INPUTS}/target.list"
  REF_FILE="${SMOKE_INPUTS}/ref.txt"
  SOURCE_TEXT_FILE="${SMOKE_INPUTS}/source_text.txt"
  AUDIO_YAML="${SMOKE_INPUTS}/audio.yaml"
elif [[ "${SCOPE}" != "full" ]]; then
  fail "SCOPE_OVERRIDE must be smoke or full; got ${SCOPE}"
fi

mkdir -p "${OUTPUT_BASE}" "${RUN_ROOT}/index_cache/${VARIANT}"
TMP_SLUG="${VARIANT//_/}"
EVAL_TMPDIR="/tmp/jx_ms_${TMP_SLUG:0:8}"
mkdir -p "${EVAL_TMPDIR}"

echo "[RUN] variant=${VARIANT} scope=${SCOPE} gpu_pair=${GPU_PAIR}"
echo "[RUN] checkpoint=${RAG_MODEL_PATH} use_maxsim=${RAG_USE_MAXSIM} windows=${RAG_MAXSIM_WINDOWS} lookback=${RAG_TIMELINE_LOOKBACK_SEC}"

CONDA_BASE=/mnt/taurus/home/jiaxuanluo/miniconda3 \
CONDA_ENV_NAME=spaCyEnv \
PYTHONPATH="${ACTIVE_ROOT}/eval:${ACTIVE_ROOT}:${PYTHONPATH:-}" \
RASST_ACTIVE_CODE_ROOT="${ACTIVE_ROOT}" \
RASST_ROOT="${RASST_ROOT}" \
ROOT_DIR="${ACTIVE_ROOT}" \
MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL_PATH}" \
RAG_USE_MAXSIM_OVERRIDE="${RAG_USE_MAXSIM}" \
RAG_POOLING_TYPE_OVERRIDE=transformer \
RAG_MAXSIM_WINDOWS_OVERRIDE="${RAG_MAXSIM_WINDOWS}" \
RAG_MAXSIM_STRIDE_OVERRIDE=2 \
RAG_SCORE_THRESHOLD_OVERRIDE=0.78 \
RAG_TOP_K_OVERRIDE=10 \
RAG_STREAMING_MODE_OVERRIDE=timeline \
RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="${RAG_TIMELINE_LOOKBACK_SEC}" \
RAG_GPU_OVERRIDE=cuda:1 \
RAG_DEVICE_OVERRIDE=cuda:1 \
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_PAIR}" \
VLLM_TP_SIZE_OVERRIDE=2 \
GPU_MEMORY_UTILIZATION_OVERRIDE=0.72 \
VLLM_LIMIT_AUDIO_OVERRIDE=auto \
VLLM_MAX_MODEL_LEN_OVERRIDE=12288 \
MAX_CACHE_CHUNKS_OVERRIDE=30 \
KEEP_CACHE_CHUNKS_OVERRIDE=30 \
MAX_CACHE_SECONDS_OVERRIDE=0 \
KEEP_CACHE_SECONDS_OVERRIDE=0 \
MAX_NEW_TOKENS_OVERRIDE=80 \
LATENCY_MULTIPLIER_OVERRIDE=2 \
LANG_CODE_OVERRIDE=zh \
GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
EVAL_GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
SRC_LIST_OVERRIDE="${SRC_LIST}" \
TGT_LIST_OVERRIDE="${TGT_LIST}" \
REF_FILE_OVERRIDE="${REF_FILE}" \
SOURCE_TEXT_FILE_OVERRIDE="${SOURCE_TEXT_FILE}" \
AUDIO_YAML_OVERRIDE="${AUDIO_YAML}" \
EVAL_MODE_OVERRIDE=acl6060 \
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
INDEX_CACHE_DIR_OVERRIDE="${RUN_ROOT}/index_cache/${VARIANT}" \
EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}" \
DENSITY_TAG="multiscale_e2e_${VARIANT}_${SCOPE}" \
EMPTY_TERM_MAP_POLICY_OVERRIDE=omit \
SYSTEM_PROMPT_STYLE_OVERRIDE=given_chunks \
TERM_MAP_FORMAT_OVERRIDE=plain \
STRIP_OUTPUT_TAGS_OVERRIDE=term_t \
TERM_FCR_POLICY=term_map_source_ref_negative_sentence \
FBK_FAIRSEQ_ROOT_OVERRIDE=/mnt/taurus/home/jiaxuanluo/FBK-fairseq \
MWERSEGMENTER_ROOT=/mnt/taurus/home/jiaxuanluo/mwerSegmenter \
CLEAN_OUTPUT_DIR_OVERRIDE=1 \
WANDB_LOG_OVERRIDE=0 \
bash "${EVAL_SCRIPT}"

find "${OUTPUT_BASE}" -type f \( -name instances.log -o -name eval_results.tsv \) -print
