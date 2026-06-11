#!/usr/bin/env bash
set -euo pipefail

# Prepare / upload / download the public RASST Speech-LLM SFT dataset.
#
# Only JSONL metadata + stats/recipe are published; audio is held out and the
# 'audios' fields are rewritten to GigaSpeech-style relative keys. prepare and
# (dry-run) upload/download are safe by default; pass --execute to actually
# upload/download (also requires RASST_ALLOW_HF_UPLOAD=1 / RASST_ALLOW_DOWNLOAD=1).

ROOT_DIR="${RASST_ROOT:-/mnt/taurus/data2/jiaxuanluo/RASST}"
export RASST_ROOT="${ROOT_DIR}"
MANIFEST="${RASST_SLM_DATASET_MANIFEST:-${ROOT_DIR}/code/rasst/manifests/slm_training_dataset.cap16_denoise_budget_ttag.json}"

action="${1:-prepare}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${action}" in
  prepare)
    exec python "${ROOT_DIR}/code/rasst/tools/hf_release_slm_dataset.py" prepare --manifest "${MANIFEST}" "$@"
    ;;
  upload)
    exec python "${ROOT_DIR}/code/rasst/tools/hf_release_slm_dataset.py" upload --manifest "${MANIFEST}" "$@"
    ;;
  download)
    exec python "${ROOT_DIR}/code/rasst/tools/hf_release_slm_dataset.py" download --manifest "${MANIFEST}" "$@"
    ;;
  *)
    echo "usage: $0 {prepare|upload|download} [args...]" >&2
    exit 2
    ;;
esac
