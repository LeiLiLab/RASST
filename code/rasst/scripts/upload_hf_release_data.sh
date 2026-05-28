#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${RASST_ROOT:-/mnt/taurus/data2/jiaxuanluo/RASST}"
export RASST_ROOT="${ROOT_DIR}"
MANIFEST="${RASST_RELEASE_DATA_MANIFEST:-${ROOT_DIR}/code/rasst/manifests/main_result_eval.global_cache30_30_20_20.json}"

action="${1:-upload}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${action}" in
  prepare)
    exec python "${ROOT_DIR}/code/rasst/tools/hf_release_data.py" prepare --manifest "${MANIFEST}" "$@"
    ;;
  upload)
    exec python "${ROOT_DIR}/code/rasst/tools/hf_release_data.py" upload --manifest "${MANIFEST}" "$@"
    ;;
  *)
    echo "usage: $0 {prepare|upload} [args...]" >&2
    exit 2
    ;;
esac
