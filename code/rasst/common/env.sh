#!/usr/bin/env bash
set -euo pipefail

export RASST_ROOT="${RASST_ROOT:-/mnt/data2/jiaxuanluo/RASST}"
export RASST_CODE_ROOT="${RASST_CODE_ROOT:-${RASST_ROOT}/code}"
export RASST_LEGACY_CODE_ROOT="${RASST_LEGACY_CODE_ROOT:-${RASST_CODE_ROOT}/legacy}"
export RASST_DATA_ROOT="${RASST_DATA_ROOT:-${RASST_ROOT}/data}"
export RASST_LOG_ROOT="${RASST_LOG_ROOT:-${RASST_ROOT}/logs}"
export RASST_OUTPUT_ROOT="${RASST_OUTPUT_ROOT:-${RASST_ROOT}/outputs}"
export RASST_CHECKPOINT_ROOT="${RASST_CHECKPOINT_ROOT:-${RASST_ROOT}/checkpoints}"
export RASST_FIGURE_ROOT="${RASST_FIGURE_ROOT:-${RASST_ROOT}/figures}"

export EVAL_TMPDIR="${EVAL_TMPDIR:-/tmp/jx_rasst}"
export TMPDIR="${EVAL_TMPDIR}"

mkdir -p \
  "${EVAL_TMPDIR}" \
  "${RASST_DATA_ROOT}" \
  "${RASST_LOG_ROOT}" \
  "${RASST_OUTPUT_ROOT}" \
  "${RASST_CHECKPOINT_ROOT}" \
  "${RASST_FIGURE_ROOT}"
