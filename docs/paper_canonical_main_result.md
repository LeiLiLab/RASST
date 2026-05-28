# RASST Main Results

This note separates the release-canonical main result from the submitted-paper
exact provenance.

The release-canonical result uses one global cache policy for all languages:

```text
lm=1,2 -> max_chunks=keep_chunks=30
lm=3,4 -> max_chunks=keep_chunks=20
```

The tracked release snapshot is:

```text
/mnt/taurus/data2/jiaxuanluo/RASST/docs/results/main_result_global_cache30_30_20_20/
```

The release-canonical eval manifest is:

```text
/mnt/taurus/data2/jiaxuanluo/RASST/code/rasst/manifests/main_result_eval.global_cache30_30_20_20.json
```

The submitted-paper exact RASST rows remain preserved for provenance. Their
historical manifest is:

```text
/mnt/taurus/data2/jiaxuanluo/RASST/code/rasst/manifests/main_result_eval.paper_canonical_rasst24.json
```

## Release-Canonical Artifacts

Tracked release snapshot:

```text
/mnt/taurus/data2/jiaxuanluo/RASST/docs/results/main_result_global_cache30_30_20_20/main_result.tsv
/mnt/taurus/data2/jiaxuanluo/RASST/docs/results/main_result_global_cache30_30_20_20/rasst24.tsv
/mnt/taurus/data2/jiaxuanluo/RASST/docs/results/main_result_global_cache30_30_20_20/compare_vs_infinisst_and_paper.tsv
/mnt/taurus/data2/jiaxuanluo/RASST/docs/results/main_result_global_cache30_30_20_20/new_main_result_tagged_global_cache30_30_20_20.pdf
/mnt/taurus/data2/jiaxuanluo/RASST/docs/results/main_result_global_cache30_30_20_20/medicine_main_result_global_cache30_30_20_20.pdf
```

Runtime source artifacts:

```text
/mnt/taurus/data2/jiaxuanluo/RASST/outputs/canonical/main_result/paper_global_cache30_30_20_20_main_result.tsv
/mnt/taurus/data2/jiaxuanluo/RASST/outputs/canonical/main_result/paper_global_cache30_30_20_20_rasst24.tsv
/mnt/taurus/data2/jiaxuanluo/RASST/outputs/canonical/main_result/paper_global_cache30_30_20_20_compare.tsv
/mnt/taurus/data2/jiaxuanluo/RASST/figures/main_result_global_cache30_30_20_20/
```

The release-canonical table improves BLEU over InfiniSST in 19 of 24 RASST
cells. It is the result to use for release-facing discussion of the final global
cache policy.

## Submitted-Paper Exact Artifacts

The submitted-paper exact table contains 24 RASST cells:

- Domains: `acl_tagged_raw`, `medicine_hardraw`
- Languages: `zh`, `de`, `ja`
- Latency multipliers: `lm=1,2,3,4`
- Method: `RASST`

It excludes ablations, `acl_paper_extracted`, Offline ST, Offline + GT terms,
and InfiniSST baselines.

Submitted-paper exact TSVs in this workspace:

```text
/mnt/taurus/data2/jiaxuanluo/RASST/outputs/canonical/main_result/paper_exact_main_result_rasst24.tsv
/mnt/taurus/data2/jiaxuanluo/RASST/outputs/canonical/main_result/paper_exact_main_result_rasst24_de.tsv
/mnt/taurus/data2/jiaxuanluo/RASST/outputs/canonical/main_result/paper_exact_main_result_rasst24_check.tsv
```

The release-facing global-policy snapshot is:

```text
/mnt/taurus/data2/jiaxuanluo/RASST/docs/results/main_result_global_cache30_30_20_20/main_result.tsv
```

The submitted-paper exact TSVs remain preserved under `outputs/canonical/` as
historical/reference artifacts. The check TSV verifies that the rounded table
metrics match each recorded `eval_results.tsv` source path.

Each result row keeps:

- `source_path`: original verified `eval_results.tsv`
- `event_id`: launch/provenance event for that cell
- `wandb_run_id`: W&B run when available
- `status`: expected to be `verified`
- `note`: selection rationale or known caveat

## Submitted-Paper Exact Common Settings

Common settings recorded by the manifest:

| Setting | Value |
| --- | --- |
| Retriever | `retriever_hn1024` |
| Retriever source run | `lh1b88kw` |
| RAG top-k | `10` |
| RAG score threshold | `0.78` |
| RAG timeline lookback | `1.92` sec |
| Term-map format | `plain` |
| Empty term-map policy | `omit` |
| System prompt style | `given_chunks` |
| Output tags stripped before scoring | `term_t` |
| Term FCR policy | `term_map_source_ref_negative_sentence` |

Drivers:

| Runner | Driver |
| --- | --- |
| `serial_simuleval` | `/mnt/taurus/data2/jiaxuanluo/RASST/code/rasst/eval/eval_density_unified.sh` |
| `batch_vllm` | `/mnt/taurus/data2/jiaxuanluo/RASST/code/rasst/eval/launchers/20260524__batched_vllm_rag_eval.sh` |

Input/model/glossary assets are resolved by the release wrapper in this order:
explicit environment override, RASST-local path, then frozen legacy path when
`RASST_USE_LEGACY_PATHS=1` permits legacy fallback.

## Main Assets

| Asset | RASST-local path |
| --- | --- |
| `retriever_hn1024` | `/mnt/taurus/data2/jiaxuanluo/RASST/checkpoints/retriever/hn1024.pt` |
| `model_zh_cap16_denoise` | `/mnt/taurus/data2/jiaxuanluo/RASST/checkpoints/slm/zh_cap16_denoise_ttag/speech_llm_zh_cap16_denoise_budget_ttag_r32a32_ep1_taurus4_hf` |
| `model_zh_new_v9` | `/mnt/taurus/data2/jiaxuanluo/RASST/checkpoints/slm/zh_new_v9/v0-20260524-062743-hf` (reference-only provenance) |
| `model_de_cap16_denoise` | `/mnt/taurus/data2/jiaxuanluo/RASST/checkpoints/slm/de_cap16_denoise_ttag/v0-20260525-203735-hf` |
| `model_ja_cap16_denoise` | `/mnt/taurus/data2/jiaxuanluo/RASST/checkpoints/slm/ja_cap16_denoise_ttag/v2-20260525-235251-hf` |
| `glossary_acl_tagged_raw` | `/mnt/taurus/data2/jiaxuanluo/RASST/data/glossaries/acl6060_tagged_gt_raw_min_norm2.json` |
| `glossary_medicine_hardraw` | `/mnt/taurus/data2/jiaxuanluo/RASST/data/glossaries/hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json` |
| ACL inputs | `/mnt/taurus/data2/jiaxuanluo/RASST/data/main_result/inputs/acl_<lang>` |
| Medicine inputs | `/mnt/taurus/data2/jiaxuanluo/RASST/data/main_result/inputs/medicine_<lang>` |

## Cell-Specific Overrides

This table is generated from `code/rasst/manifests/main_result_eval.paper_canonical_rasst24.json` after retargeting the release-facing manifest to the global cache policy.

| Domain | Lang | lm | Runner | Model | max_new_tokens | token_policy | audio_limit | max_model_len | cache max/keep | Provenance event |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `acl_tagged_raw` | `zh` | 1 | `serial_simuleval` | `model_zh_cap16_denoise` | 40 | `fixed` | auto | 12288 | 30/30 | `20260524T1442__simuleval__tagged_acl_same_lm_batch_v1exact_hn1024_tau078_raw_zh_lm1_max256` |
| `acl_tagged_raw` | `zh` | 2 | `serial_simuleval` | `model_zh_cap16_denoise` | 80 | `fixed` | auto | 12288 | 30/30 | `20260524T0522__simuleval__tagged_acl_new_v9_hn1024_tau078_raw_zh_lm23_aries4567` |
| `acl_tagged_raw` | `zh` | 3 | `serial_simuleval` | `model_zh_cap16_denoise` | 120 | `fixed` | auto | 12288 | 20/20 | `20260524T0522__simuleval__tagged_acl_new_v9_hn1024_tau078_raw_zh_lm23_aries4567` |
| `acl_tagged_raw` | `zh` | 4 | `serial_simuleval` | `model_zh_cap16_denoise` | 160 | `fixed` | auto | 12288 | 20/20 | `20260524T0555__simuleval__tagged_acl_new_v9_hn1024_tau078_raw_zh_lm4_aries45` |
| `acl_tagged_raw` | `de` | 1 | `serial_simuleval` | `model_de_cap16_denoise` | 40 | `fixed` | auto | 12288 | 30/30 | `20260526T003437__simuleval__tagged_acl_de_lm1_serial_promptfix_cache30_audioauto_max40lm_taurus` |
| `acl_tagged_raw` | `de` | 2 | `serial_simuleval` | `model_de_cap16_denoise` | 80 | `fixed` | 128 | 12288 | 30/30 | `20260525T215343__simuleval__main_result_rasst_serial_de_ja_acl_then_medicine_cache30_max40lm_taurus` |
| `acl_tagged_raw` | `de` | 3 | `batch_vllm` | `model_de_cap16_denoise` | 40 | `lm_scaled` | 128 | manifest default | 20/20 | `20260525T172158__simuleval__de_acl_lm23_then_ja_medicine_cap16_denoise_aries` |
| `acl_tagged_raw` | `de` | 4 | `serial_simuleval` | `model_de_cap16_denoise` | 160 | `fixed` | auto | 12288 | 20/20 | `20260526T020512__simuleval__tagged_acl_de_lm4_serial_promptfix_audioauto_cache30_max40lm_taurus_direct` |
| `acl_tagged_raw` | `ja` | 1 | `serial_simuleval` | `model_ja_cap16_denoise` | 40 | `fixed` | 128 | 12288 | 30/30 | `20260526T013751__simuleval__tagged_acl_ja_lm1to4_serial_promptfix_cache30_vllmaudio128_max40lm_taurus` |
| `acl_tagged_raw` | `ja` | 2 | `serial_simuleval` | `model_ja_cap16_denoise` | 80 | `fixed` | 128 | 12288 | 30/30 | `20260526T013751__simuleval__tagged_acl_ja_lm1to4_serial_promptfix_cache30_vllmaudio128_max40lm_taurus` |
| `acl_tagged_raw` | `ja` | 3 | `serial_simuleval` | `model_ja_cap16_denoise` | 120 | `fixed` | 128 | 12288 | 20/20 | `20260526T013751__simuleval__tagged_acl_ja_lm1to4_serial_promptfix_cache30_vllmaudio128_max40lm_taurus` |
| `acl_tagged_raw` | `ja` | 4 | `serial_simuleval` | `model_ja_cap16_denoise` | 160 | `fixed` | 128 | 12288 | 20/20 | `20260526T013751__simuleval__tagged_acl_ja_lm1to4_serial_promptfix_cache30_vllmaudio128_max40lm_taurus` |
| `medicine_hardraw` | `zh` | 1 | `serial_simuleval` | `model_zh_cap16_denoise` | 40 | `fixed` | auto | 12288 | 30/30 | `20260524T0242__simuleval__medicine_hardraw_hn1024_tau078_new_v9_batch` |
| `medicine_hardraw` | `zh` | 2 | `serial_simuleval` | `model_zh_cap16_denoise` | 80 | `fixed` | auto | 12288 | 30/30 | `20260524T0242__simuleval__medicine_hardraw_hn1024_tau078_new_v9_batch` |
| `medicine_hardraw` | `zh` | 3 | `serial_simuleval` | `model_zh_cap16_denoise` | 120 | `fixed` | auto | 12288 | 20/20 | `20260524T0242__simuleval__medicine_hardraw_hn1024_tau078_new_v9_batch` |
| `medicine_hardraw` | `zh` | 4 | `serial_simuleval` | `model_zh_cap16_denoise` | 160 | `fixed` | auto | 12288 | 20/20 | `20260524T0242__simuleval__medicine_hardraw_hn1024_tau078_new_v9_batch` |
| `medicine_hardraw` | `de` | 1 | `serial_simuleval` | `model_de_cap16_denoise` | 40 | `fixed` | auto | 12288 | 30/30 | `20260526T035925__simuleval__medicine_de_lm1234_serial_promptfix_cache30_audioauto_max40lm_taurus` |
| `medicine_hardraw` | `de` | 2 | `serial_simuleval` | `model_de_cap16_denoise` | 80 | `fixed` | auto | 12288 | 30/30 | `20260526T035925__simuleval__medicine_de_lm1234_serial_promptfix_cache30_audioauto_max40lm_taurus` |
| `medicine_hardraw` | `de` | 3 | `batch_vllm` | `model_de_cap16_denoise` | 40 | `lm_scaled` | 128 | 12288 | 20/20 | `20260525T170456__simuleval__medicine_de_cap16_denoise_lm34_batch_taurus03` |
| `medicine_hardraw` | `de` | 4 | `serial_simuleval` | `model_de_cap16_denoise` | 160 | `fixed` | auto | 12288 | 20/20 | `20260526T043300__simuleval__medicine_de_lm4_serial_promptfix_cache30_audioauto_max40lm_taurus23` |
| `medicine_hardraw` | `ja` | 1 | `serial_simuleval` | `model_ja_cap16_denoise` | 40 | `fixed` | 128 | 12288 | 30/30 | `20260526T045605__simuleval__medicine_ja_lm1_serial_promptfix_vllmaudio128_aries45` |
| `medicine_hardraw` | `ja` | 2 | `serial_simuleval` | `model_ja_cap16_denoise` | 80 | `fixed` | 128 | 12288 | 30/30 | `20260526T045605__simuleval__medicine_ja_lm2_serial_promptfix_vllmaudio128_aries67` |
| `medicine_hardraw` | `ja` | 3 | `batch_vllm` | `model_ja_cap16_denoise` | 40 | `lm_scaled` | 128 | 12288 | 20/20 | `20260525T1840__simuleval__medicine_ja_cap16_denoise_lm1234_batch_taurus` |
| `medicine_hardraw` | `ja` | 4 | `batch_vllm` | `model_ja_cap16_denoise` | 40 | `lm_scaled` | 128 | 12288 | 20/20 | `20260525T1840__simuleval__medicine_ja_cap16_denoise_lm1234_batch_taurus` |

## Validation And Rerun Entry Point

Validate the manifest and current-host assets:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
bash code/rasst/scripts/eval_main_result.sh --validate-only
```

Print concrete commands without launching:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
bash code/rasst/scripts/eval_main_result.sh --dry-run
```

Filter a single canonical cell:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
bash code/rasst/scripts/eval_main_result.sh --dry-run --domain acl_tagged_raw --lang de --lm 1
```

Completed reruns write `summary_all.tsv`, `comparison_report.tsv`,
`config_cells.tsv`, `config_differences.tsv`, and `config_report.md` under the
chosen run root. Use `comparison_report.tsv` and the source `eval_results.tsv`
artifacts before changing any submitted-paper exact row.

## Update Rule

Do not replace a submitted-paper exact row only because a newer run exists. A
row should be updated only when the new artifact is complete, the metric
tradeoff is accepted, and the TSV keeps the new `source_path`, `event_id`,
config evidence, and validation status.
