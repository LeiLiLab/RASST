#!/usr/bin/env bash
set -euo pipefail

# InfiniSST no-RAG baseline eval wrapper.
#
# Mirrors eval_main_result.sh but defaults to the no-RAG baseline manifest, so
# the InfiniSST baseline can be validated and reproduced with the same surface
# (--validate-only / --dry-run / --sbatch / --cache-chunks-by-lm / --domain ...).
# The orchestrator selects the no-RAG path from metadata.common_eval_config.rag_mode.
# A user-supplied --manifest after the default still wins (argparse keeps the last).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RASST_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
if [[ "${RASST_ROOT_DEFAULT}" == /mnt/data2/* ]]; then
  RASST_ROOT_DEFAULT="/mnt/taurus/data2/${RASST_ROOT_DEFAULT#/mnt/data2/}"
fi
export RASST_ROOT="${RASST_ROOT:-${RASST_ROOT_DEFAULT}}"

BASELINE_MANIFEST="${RASST_BASELINE_MANIFEST:-${RASST_ROOT}/code/rasst/manifests/main_result_baseline_no_rag.global_cache30_30_20_20.json}"

exec "${PYTHON:-python}" "${RASST_ROOT}/code/rasst/tools/eval_main_result.py" \
  --manifest "${BASELINE_MANIFEST}" \
  "$@"
