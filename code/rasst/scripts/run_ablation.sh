#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/task_lib.sh
source "${SCRIPT_DIR}/../common/task_lib.sh"
parse_common_args "$@"

target_rel="${RASST_ABLATION_TARGET:-documents/code/train/term_train/launchers/2026/05/20260524__retriever_encoder_ablation_devraw_fixeddenom_eval.sh}"
target_abs="$(legacy_path "${target_rel}")"
run_or_dry_run "run_ablation" "$(dirname "${target_abs}")" "run_ablation" bash "${target_abs}" "${RASST_WRAPPER_ARGS[@]}"
