#!/usr/bin/env bash
set -euo pipefail

# Prepare / upload / download the public RASST wiki-synthetic retriever dataset.
#
# prepare writes the parquet tables, dataset card, and shard plans under
# --stage-root. upload and download are dry runs unless --execute is passed
# (plus RASST_ALLOW_HF_UPLOAD=1 / RASST_ALLOW_DOWNLOAD=1).

ROOT_DIR="${RASST_ROOT:-/mnt/taurus/data2/jiaxuanluo/RASST}"
export RASST_ROOT="${ROOT_DIR}"
MANIFEST="${RASST_WIKI_SYNTH_DATASET_MANIFEST:-${ROOT_DIR}/code/rasst/manifests/retriever_wiki_synth_dataset.json}"
TOOL="${ROOT_DIR}/code/rasst/tools/hf_release_wiki_synth_dataset.py"

action="${1:-}"
case "${action}" in
  prepare|upload|download)
    shift
    exec python "${TOOL}" "${action}" --manifest "${MANIFEST}" "$@"
    ;;
  *)
    echo "usage: $0 {prepare|upload|download} [args...]" >&2
    exit 2
    ;;
esac
