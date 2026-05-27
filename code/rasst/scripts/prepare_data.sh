#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/task_lib.sh
source "${SCRIPT_DIR}/../common/task_lib.sh"
parse_common_args "$@"

target_rel="${RASST_PREPARE_DATA_TARGET:-documents/code/data_pre/training_terms_for_retriever/run_build_gsv2full_gsdedup_varctx_lmlb2p88_3p84_4p80_5p76.sh}"
target_abs="$(legacy_path "${target_rel}")"
run_or_dry_run "prepare_data" "$(dirname "${target_abs}")" "prepare_data" bash "${target_abs}" "${RASST_WRAPPER_ARGS[@]}"
