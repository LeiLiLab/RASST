# RASST

This repository freezes the RASST experiment code and paper artifact for rebuttal reproducibility.

- `code/legacy/` is a tracked snapshot exported from the InfiniSST freeze commit.
- `code/rasst/` contains thin reproducibility wrappers with RASST-local path defaults.
- `code/provenance/freeze_20260527/` records the upstream Git anchor, file inventories, and checksums.
- `paper/` contains the tracked paper PDF from the freeze.
- `data/`, `logs/`, `outputs/`, `checkpoints/`, and `figures/` are intentionally ignored runtime roots.

Use the curated wrappers with `--dry-run` first:

```bash
bash code/rasst/scripts/prepare_data.sh --dry-run
bash code/rasst/scripts/train_retriever.sh --dry-run
bash code/rasst/scripts/eval_acl.sh --dry-run
```

Actual long-running launches require `RASST_ALLOW_LAUNCH=1` and are detached with logs under `logs/curated/`.
