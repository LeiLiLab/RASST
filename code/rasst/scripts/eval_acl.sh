#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/task_lib.sh
source "${SCRIPT_DIR}/../common/task_lib.sh"
parse_common_args "$@"

target_rel="${RASST_ACL_EVAL_TARGET:-documents/code/simuleval/launchers/2026/05/20260523__acl_paper_extracted_main_no_tmsft_llmgen_rasst.sh}"
target_abs="$(legacy_path "${target_rel}")"
run_or_dry_run "eval_acl" "$(dirname "${target_abs}")" "eval_acl" bash "${target_abs}" "${RASST_WRAPPER_ARGS[@]}"
