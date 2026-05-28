#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/task_lib.sh
source "${SCRIPT_DIR}/../common/task_lib.sh"
parse_common_args "$@"

target_rel="${RASST_INDEX_TARGET:-retriever/launchers/run_build_index_v4.sh}"
target_abs="$(rasst_code_path "${target_rel}")"
run_or_dry_run "build_index" "$(dirname "${target_abs}")" "build_index" bash "${target_abs}" "${RASST_WRAPPER_ARGS[@]}"
