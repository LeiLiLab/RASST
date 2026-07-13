#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RAW_INPUT="/mnt/gemini/data1/jiaxuanluo/speech_llm_deja_termmap_ablation_cap16_exactboundary_20260525/ja/retriever_hn1024_tau078_cap16_exactboundary/train_s_ja_retriever_results_hn1024_tau078.jsonl"
BASE_TRAIN="/mnt/gemini/data1/jiaxuanluo/speech_llm_ja_cap16_denoise_budget_20260525/ja/hn1024_tau078_cap16_denoise_budget_ttag_v1/train_s_ja_retriever_hn1024_tau078_cap16_denoise_budget_ttag_exactboundary.jsonl"
OUTPUT_DIR="${1:-}"
if [[ -z "${OUTPUT_DIR}" || "${OUTPUT_DIR}" != /* ]]; then
  echo "Usage: $0 <absolute_output_dir>" >&2
  exit 2
fi

SELECTED_RAW="${OUTPUT_DIR}/train_s_ja_all_lm1_raw.jsonl"
SELECT_STATS="${OUTPUT_DIR}/train_s_ja_all_lm1_select_stats.json"
STAGE1="${OUTPUT_DIR}/train_s_ja_all_lm1_denoise_seed43_stage1.jsonl"
FINAL_SUPPLEMENT="${OUTPUT_DIR}/train_s_ja_all_lm1_denoise_ttag_seed43.jsonl"
FINAL_TRAIN="${OUTPUT_DIR}/train_s_ja_cap16_denoise_ttag_lm1x2_seed43.jsonl"

for path in \
  "${ROOT_DIR}/slm/data_prep/select_latency_multiplier_rows.py" \
  "${ROOT_DIR}/slm/data_prep/rebuild_termmap_denoise_budget.py" \
  "${ROOT_DIR}/slm/train/src/wrap_assistant_term_targets.py" \
  "${ROOT_DIR}/slm/data_prep/assemble_latency_curriculum.py" \
  "${RAW_INPUT}" \
  "${BASE_TRAIN}"; do
  [[ -s "${path}" ]] || { echo "[ERROR] Missing required path: ${path}" >&2; exit 3; }
done

mkdir -p "${OUTPUT_DIR}"
for path in \
  "${SELECTED_RAW}" "${SELECT_STATS}" "${STAGE1}" "${FINAL_SUPPLEMENT}" "${FINAL_TRAIN}" \
  "${OUTPUT_DIR}/lm1_rebuild_stats.json" "${OUTPUT_DIR}/lm1_rebuild_samples.json" \
  "${OUTPUT_DIR}/lm1_wrap_stats.json" "${OUTPUT_DIR}/lm1_wrap_samples.json" \
  "${OUTPUT_DIR}/curriculum_stats.json"; do
  [[ ! -e "${path}" ]] || { echo "[ERROR] Refusing to overwrite output: ${path}" >&2; exit 4; }
done

python3 "${ROOT_DIR}/slm/data_prep/select_latency_multiplier_rows.py" \
  --input-jsonl "${RAW_INPUT}" \
  --output-jsonl "${SELECTED_RAW}" \
  --stats-json "${SELECT_STATS}" \
  --focus-multiplier 1 \
  --match-policy all \
  --expected-rows 1048

python3 "${ROOT_DIR}/slm/data_prep/rebuild_termmap_denoise_budget.py" \
  --input-jsonl "${SELECTED_RAW}" \
  --output-jsonl "${STAGE1}" \
  --stats-json "${OUTPUT_DIR}/lm1_rebuild_stats.json" \
  --sample-json "${OUTPUT_DIR}/lm1_rebuild_samples.json" \
  --target-lang ja \
  --budget-choices "6,8,10" \
  --budget-weights "0.45,0.35,0.20" \
  --no-gt-max-terms 4 \
  --no-gt-empty-prob 0.35 \
  --low-score-cutoff 0.82 \
  --mid-score-cutoff 0.85 \
  --low-score-keep-prob 0.25 \
  --mid-score-keep-prob 0.60 \
  --high-score-keep-prob 0.90 \
  --supported-non-gt-keep-prob 0.85 \
  --missing-score-keep-prob 0.50 \
  --min-target-chars 2 \
  --seed 43 \
  --sample-count 200

python3 "${ROOT_DIR}/slm/train/src/wrap_assistant_term_targets.py" \
  --input-jsonl "${STAGE1}" \
  --output-jsonl "${FINAL_SUPPLEMENT}" \
  --stats-json "${OUTPUT_DIR}/lm1_wrap_stats.json" \
  --sample-json "${OUTPUT_DIR}/lm1_wrap_samples.json" \
  --lang-code ja \
  --tag-template '<t>{translation}</t>' \
  --min-target-chars 2 \
  --max-tags-per-row 16 \
  --missing-gt-policy error \
  --exclude-source-tokens "this,that,these,those,his,her,hers,him,he,she,it,its,they,them,their,theirs,you,your,yours,we,our,ours,i,me,my,mine,myself,yourself,himself,herself,itself,ourselves,yourselves,themselves,what,which,who,whom,whose,someone,somebody,something,anyone,anybody,anything,everyone,everybody,everything" \
  --exact-require-text-boundaries \
  --enable-local-rewrite \
  --rewrite-boundary-only \
  --rewrite-delay-boundary-prefix \
  --rewrite-delay-boundary-min-prefix-chars 2 \
  --rewrite-require-text-boundaries \
  --sample-count 200

python3 "${ROOT_DIR}/slm/data_prep/assemble_latency_curriculum.py" \
  --base-jsonl "${BASE_TRAIN}" \
  --supplement-jsonl "${FINAL_SUPPLEMENT}" \
  --output-jsonl "${FINAL_TRAIN}" \
  --stats-json "${OUTPUT_DIR}/curriculum_stats.json" \
  --focus-multiplier 1 \
  --base-focus-rows 1048 \
  --expected-base-rows 12500 \
  --expected-supplement-rows 1048

sha256sum \
  "${RAW_INPUT}" "${BASE_TRAIN}" "${FINAL_SUPPLEMENT}" "${FINAL_TRAIN}" \
  > "${OUTPUT_DIR}/sha256sums.txt"

echo "[DONE] Ja lm=1 curriculum: ${FINAL_TRAIN}"
