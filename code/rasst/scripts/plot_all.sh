#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/task_lib.sh
source "${SCRIPT_DIR}/../common/task_lib.sh"
parse_common_args "$@"

plot_rel="${RASST_PLOT_DIR:-documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/plot}"
plot_abs="$(legacy_path "${plot_rel}")"
run_or_dry_run "plot_all" "${plot_abs}" "plot_all" bash -lc 'set -euo pipefail; for f in figure_*/plot_*.py; do python "$f"; done'
