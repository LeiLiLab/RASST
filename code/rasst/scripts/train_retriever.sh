#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/task_lib.sh
source "${SCRIPT_DIR}/../common/task_lib.sh"
parse_common_args "$@"

target_rel="${RASST_TRAIN_RETRIEVER_TARGET:-documents/code/train/term_train/launchers/2026/05/20260525__varctx576_hn1024_gsonly_resume_best_aries8.sh}"
target_abs="$(legacy_path "${target_rel}")"
run_or_dry_run "train_retriever" "$(dirname "${target_abs}")" "train_retriever" bash "${target_abs}" "${RASST_WRAPPER_ARGS[@]}"
