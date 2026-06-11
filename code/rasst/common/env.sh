#!/usr/bin/env bash
set -euo pipefail

# Derive the repo root from this file's location (code/rasst/common/env.sh)
# so the release scripts work on any checkout; callers may still set RASST_ROOT.
_RASST_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RASST_ROOT="${RASST_ROOT:-$(cd "${_RASST_ENV_DIR}/../../.." && pwd)}"
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
