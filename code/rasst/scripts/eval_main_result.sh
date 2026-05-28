#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RASST_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
if [[ "${RASST_ROOT_DEFAULT}" == /mnt/data2/* ]]; then
  RASST_ROOT_DEFAULT="/mnt/taurus/data2/${RASST_ROOT_DEFAULT#/mnt/data2/}"
fi
export RASST_ROOT="${RASST_ROOT:-${RASST_ROOT_DEFAULT}}"

exec "${PYTHON:-python}" "${RASST_ROOT}/code/rasst/tools/eval_main_result.py" "$@"
