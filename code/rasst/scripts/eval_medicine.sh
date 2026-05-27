#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/task_lib.sh
source "${SCRIPT_DIR}/../common/task_lib.sh"
parse_common_args "$@"

target_rel="${RASST_MEDICINE_EVAL_TARGET:-documents/code/simuleval/launchers/2026/05/20260524__medicine_hardraw_rasst_zh_lm1_max80_batch.sh}"
target_abs="$(legacy_path "${target_rel}")"
run_or_dry_run "eval_medicine" "$(dirname "${target_abs}")" "eval_medicine" bash "${target_abs}" "${RASST_WRAPPER_ARGS[@]}"
