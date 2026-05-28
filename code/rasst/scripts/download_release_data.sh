#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${RASST_ROOT:-/mnt/taurus/data2/jiaxuanluo/RASST}"
export RASST_ROOT="${ROOT_DIR}"
MANIFEST="${RASST_RELEASE_DATA_MANIFEST:-${ROOT_DIR}/code/rasst/manifests/main_result_eval.global_cache30_30_20_20.json}"

execute_args=()
passthrough=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --download|--execute)
      execute_args+=(--execute)
      shift
      ;;
    --dry-run)
      shift
      ;;
    *)
      passthrough+=("$1")
      shift
      ;;
  esac
done

exec python "${ROOT_DIR}/code/rasst/tools/hf_release_data.py" download \
  --manifest "${MANIFEST}" \
  "${execute_args[@]}" \
  "${passthrough[@]}"
