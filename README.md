# RASST

This repository freezes the RASST experiment code and paper artifact for
rebuttal reproducibility.

- `code/legacy/` is a tracked snapshot exported from the InfiniSST freeze commit.
- `code/rasst/` contains the release-facing RASST reproduction closure and wrappers.
- `code/provenance/freeze_20260527/` records the upstream Git anchor, file inventories, and checksums.
- `paper/` contains the tracked paper PDF from the freeze.
- `data/`, `logs/`, `outputs/`, `checkpoints/`, and `figures/` are intentionally ignored runtime roots.

The main reproduction guide is at [REPRODUCE_MAIN_RESULT.md](REPRODUCE_MAIN_RESULT.md).
The tracked global-cache result snapshot is under
`docs/results/main_result_global_cache30_30_20_20/`.

Use the curated wrappers with `--dry-run` first:

```bash
bash code/rasst/scripts/prepare_data.sh --dry-run
bash code/rasst/scripts/train_retriever.sh --dry-run
bash code/rasst/scripts/eval_main_result.sh --dry-run
```

Actual long-running launches require `RASST_ALLOW_LAUNCH=1` and are detached with logs under `logs/curated/`.

Public Hugging Face release asset IDs for the three SLMs and HN1024 retriever
are declared in `code/rasst/manifests/main_result_eval.global_cache30_30_20_20.json`.
Download them into ignored `checkpoints/` paths with:

```bash
bash code/rasst/scripts/download_release_assets.sh --dry-run
RASST_ALLOW_DOWNLOAD=1 bash code/rasst/scripts/download_release_assets.sh --download
```
