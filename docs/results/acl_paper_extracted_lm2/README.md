# ACL Paper-Extracted Realistic-Glossary Evaluation

This directory is the stable entry point for the rebuttal experiment that
replaces tagged-ACL terminology with per-paper terminology extracted from the
paper itself. The five tracked glossaries are under
[`data/glossaries/acl_paper_extracted`](../../../data/glossaries/acl_paper_extracted/).

## Fixed protocol

- Languages: En-Zh, En-Ja, En-De.
- Latency multiplier: `lm=2` only.
- Five ACL6060 talks, each paired with its own extracted glossary.
- Released RASST cap16-denoise term-tagging SLM for each language.
- Retriever: released HN1024 checkpoint, top-k `10`, threshold `0.78`.
- Prompting: `given_chunks`, plain term map, omit empty term maps.
- Cache: `max_cache_chunks=30`, `keep_cache_chunks=30`.
- Decode limit: `max_new_tokens=80`.
- Metrics are aggregated across the five per-paper runs; terminology counts use
  the matching per-paper extracted glossary as the denominator.

## Run status

The first submission (`47028`, `47029`, `47030`) built the glossary indices but
failed before inference because the selected conda environment had a
non-portable `simuleval` shebang. No metrics from that attempt are valid.

The corrected two-GPU run uses the host-qualified `spaCyEnv` and shares the
retriever with vLLM on `cuda:0`:

| Language | Slurm job | Dependency | Current status |
| --- | ---: | --- | --- |
| zh | `47067` | none | complete and verified |
| ja | `47068` | after `47067` | running |
| de | `47069` | after `47068` | queued |

## Verified En-Zh result

The aggregate uses macro averages for BLEU and StreamLAAL and micro counts for
terminology metrics. The tracked source is [`zh/aggregate.tsv`](zh/aggregate.tsv);
the five source rows are under [`zh/per_paper`](zh/per_paper/).

| BLEU | Masked-term BLEU | StreamLAAL | TERM_ACC | Real term adoption | TERM_FCR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 46.1716 | 43.2382 | 1827.82 | 223/235 (94.89%) | 209/215 (97.21%) | 51/172 (29.65%) |

`Masked-term BLEU` is the macro average of the five per-paper rows; the legacy
aggregate script does not yet emit this column in `aggregate.tsv`.

| Paper | BLEU | Masked-term BLEU | StreamLAAL | TERM_ACC |
| --- | ---: | ---: | ---: | ---: |
| `2022.acl-long.110` | 46.7347 | 46.0713 | 1720.69 | 27/32 (84.38%) |
| `2022.acl-long.117` | 45.0885 | 42.5288 | 1852.14 | 69/71 (97.18%) |
| `2022.acl-long.268` | 43.7800 | 39.8689 | 1928.80 | 60/63 (95.24%) |
| `2022.acl-long.367` | 51.6280 | 47.1516 | 1781.53 | 33/34 (97.06%) |
| `2022.acl-long.590` | 43.6268 | 40.5704 | 1855.97 | 34/35 (97.14%) |

The inherited output directory label contains `tau073`, but the recorded
runtime threshold and every per-paper path use `th0.78`; `0.78` is the actual
configuration. Ja and De results will be added only after all five per-paper
artifacts for that language finish successfully.
