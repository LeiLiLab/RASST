# Curated RASST Reproduction Code

The release-facing main-result reproduction closure is copied into
`code/rasst/`. Public wrappers print commands in `--dry-run` and only launch
when `RASST_ALLOW_LAUNCH=1`.

Override any default target by setting the matching environment variable, for example:

```bash
RASST_ACL_EVAL_TARGET=eval/launchers/example.sh \
  bash code/rasst/scripts/eval_acl.sh --dry-run
```

`code/legacy/` remains frozen provenance for comparison. Release-facing
commands should not require launching code from `code/legacy/`.

## RASST-Local Code Layout

```text
code/rasst/slm/data_prep/        cap16 denoise-budget termtag data builders
code/rasst/slm/train/            SLM training launchers and Docker wrapper
code/rasst/eval/                 serial SimulEval, batched vLLM, scorer, agent
code/rasst/retriever/            retriever training and MaxSim index/runtime code
code/rasst/analysis/main_result/ main-result table and figure builders
```

## Main Result Eval Manifest

The release-facing final result uses one global cache policy:

```text
lm=1,2 -> max_chunks=keep_chunks=30
lm=3,4 -> max_chunks=keep_chunks=20
```

The tracked release snapshot is:

```text
docs/results/main_result_global_cache30_30_20_20/
```

The release-facing eval manifest is:

```text
code/rasst/manifests/main_result_eval.global_cache30_30_20_20.json
```

The submitted-paper exact RASST manifest is preserved as reference provenance:

```text
code/rasst/manifests/main_result_eval.paper_canonical_rasst24.json
```

A human-readable note for the release-canonical table, submitted-paper exact
table, and corresponding settings is maintained at
`docs/paper_canonical_main_result.md`.

It tracks the submitted-paper RASST cells only: `acl_tagged_raw` and
`medicine_hardraw`, `zh/de/ja`, and `lm=1..4` for 24 eval cells. Each cell keeps
the frozen legacy path plus a RASST-local path/env override for migration.

Validate the frozen manifest and all referenced current-host assets:

```bash
bash code/rasst/scripts/eval_main_result.sh --validate-only
```

Print all concrete eval commands without launching:

```bash
bash code/rasst/scripts/eval_main_result.sh --dry-run
```

Print the final global-cache policy commands without launching:

```bash
bash code/rasst/scripts/eval_main_result.sh --dry-run \
  --cache-chunks-by-lm 1:30/30,2:30/30,3:20/20,4:20/20
```

Filter a subset:

```bash
bash code/rasst/scripts/eval_main_result.sh --dry-run --domain acl_tagged_raw --lang zh --lm 1
```

Launch detached only after inspecting the dry run:

```bash
RASST_ALLOW_LAUNCH=1 bash code/rasst/scripts/eval_main_result.sh
```

Submit the same run through Slurm on Taurus:

```bash
RASST_ALLOW_LAUNCH=1 bash code/rasst/scripts/eval_main_result.sh --sbatch
```

Optional Slurm controls are `RASST_SBATCH_PARTITION`, `RASST_SBATCH_GRES`,
`RASST_SBATCH_CPUS`, `RASST_SBATCH_MEM`, and `RASST_SBATCH_TIME`.

Useful path-policy controls:

```bash
# Default: env override -> RASST-local path -> frozen legacy path.
RASST_USE_LEGACY_PATHS=1 bash code/rasst/scripts/eval_main_result.sh --validate-only

# Public-release check: env override -> RASST-local path only.
RASST_REQUIRE_LOCAL_ASSETS=1 bash code/rasst/scripts/eval_main_result.sh --validate-only
```

Completed runs write `run_meta.json`, cell outputs, `summary_all.tsv`, and
`comparison_report.tsv` under `${RASST_OUTPUT_ROOT:-outputs}/main_result_eval/<UTCSTAMP>/`.
They also write `config_cells.tsv`, `config_differences.tsv`, and
`config_report.md` so submitted-paper exact configuration drift is explicit.
Detached scripts, stdout/stderr, and PID files are written under
`${RASST_LOG_ROOT:-logs}/curated/`.

## SLM Reproduction

The release-facing SLM recipe is unified across `de`, `ja`, and `zh`:
cap16 denoise-budget term tagging with the manifest at
`code/rasst/manifests/slm_training.cap16_denoise_budget_ttag.json`.

Print all data-preparation and SLM-training commands:

```bash
bash code/rasst/scripts/reproduce_slm.sh --lang all --stage all
```

Launch detached only after inspecting the dry run:

```bash
RASST_ALLOW_LAUNCH=1 bash code/rasst/scripts/reproduce_slm.sh --lang all --stage all --launch
```

The old zh `new_v9` path is preserved only as reference provenance in
`docs/reference/zh_new_v9_reference_only.md`.
