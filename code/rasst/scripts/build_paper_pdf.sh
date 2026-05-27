#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/task_lib.sh
source "${SCRIPT_DIR}/../common/task_lib.sh"
parse_common_args "$@"

paper_rel="${RASST_PAPER_LATEX_DIR:-documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex}"
paper_abs="$(legacy_path "${paper_rel}")"
tex_file="${RASST_PAPER_TEX_FILE:-acl_latex.tex}"
run_or_dry_run "build_paper_pdf" "${paper_abs}" "build_paper_pdf" latexmk -pdf -interaction=nonstopmode -halt-on-error "${tex_file}"
