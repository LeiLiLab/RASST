# ACL Paper-Extracted Realistic-Glossary Evaluation

This directory is the stable entry point for the rebuttal experiment that uses
terminology extracted from each talk's paper as the RASST retrieval index. The
five tracked index glossaries are under
[`data/glossaries/acl_paper_extracted`](../../../data/glossaries/acl_paper_extracted/).

## Two distinct glossary roles

The inference and evaluation glossaries are intentionally different:

- **Inference index:** RASST retrieves only from the matching paper-derived
  glossary. InfiniSST does not use a glossary or retriever.
- **Evaluation denominator:** both systems are scored against the unchanged
  tagged raw ACL6060 glossary. Its SHA-256 is
  `f9f171c6475c4bb19250f5f93063a5ef034cbdcc1f8a995c593647718cf9a5b6`.
- Consequently, `TERM_TOTAL` remains `890` for En-Zh, `940` for En-Ja, and
  `935` for En-De. Terms missing from the paper-derived index remain errors.

The previously published matched-paper-glossary denominator was incorrect for
this ablation and has been removed.

## Fixed protocol

- Languages: En-Zh, En-Ja, En-De.
- Latency multiplier: `lm=2` only.
- Five ACL6060 talks, each paired with its own paper-derived RASST index.
- Released cap16-denoise term-tagging SLM and HN1024 retriever.
- Retriever top-k `10`, threshold `0.78`.
- Prompting: `given_chunks`, plain term map, omit empty term maps.
- Cache: `max_cache_chunks=30`, `keep_cache_chunks=30`.
- Decode limit: `max_new_tokens=80`.
- InfiniSST is not rerun; its registered `lm=2` `instances.log` is rescored.
- RASST's five per-paper inference logs are combined in ACL dev order and
  rescored once with the same tagged raw glossary as InfiniSST.

## Corrected unified comparison

All three languages are complete and verified. The machine-readable table is
[`comparison.tsv`](comparison.tsv).

| Language | System | BLEU | Masked-term BLEU | StreamLAAL | Tagged-raw TERM_ACC |
| --- | --- | ---: | ---: | ---: | ---: |
| En-Zh | InfiniSST | 45.8268 | 40.7155 | 1765.72 | 669/890 (75.17%) |
| En-Zh | RASST, paper-derived index | 46.3280 | 41.0132 | 1827.50 | 693/890 (77.87%) |
| En-Ja | InfiniSST | 27.7202 | 25.0201 | 2291.16 | 620/940 (65.96%) |
| En-Ja | RASST, paper-derived index | 27.7656 | 25.0567 | 2107.86 | 614/940 (65.32%) |
| En-De | InfiniSST | 30.2516 | 28.9758 | 1759.06 | 632/935 (67.59%) |
| En-De | RASST, paper-derived index | 28.1247 | 26.3229 | 1624.46 | 652/935 (69.73%) |

RASST minus InfiniSST tagged-raw TERM_ACC is `+2.70`, `-0.64`, and `+2.14`
percentage points for En-Zh, En-Ja, and En-De. BLEU changes by `+0.5012`,
`+0.0455`, and `-2.1270`, respectively. This is mixed evidence: the
paper-derived index retains a terminology advantage in Zh and De, is slightly
below InfiniSST in Ja, and does not provide a uniform translation-quality gain.

## Artifacts and validation

- RASST corrected post-eval outputs: [`zh/aggregate.tsv`](zh/aggregate.tsv),
  [`ja/aggregate.tsv`](ja/aggregate.tsv), and
  [`de/aggregate.tsv`](de/aggregate.tsv).
- InfiniSST corrected post-eval outputs: [`infinisst/zh`](infinisst/zh),
  [`infinisst/ja`](infinisst/ja), and [`infinisst/de`](infinisst/de).
- Every TSV is produced in evaluator mode `acl6060` with the tagged raw
  glossary, not mode `extracted_by_paper`.
- InfiniSST BLEU and StreamLAAL reproduce the registered tagged-ACL `lm=2`
  values. Ja/De TERM counts also reproduce their registered rows. For Zh, the
  retained `instances.log` recomputes to `669/890 = 0.7517`, matching its tracked
  artifact TSV; this supersedes the older user-supplied `0.7655` summary value,
  which has no retained numerator/denominator.
- RASST combined logs contain exactly five entries in ACL dev order:
  `268`, `367`, `590`, `110`, `117`; their indices are normalized to `0..4`.
- RASST output `<term>`/`<t>` markup is stripped while preserving inner text
  before scoring.

The first RASST submission (`47028`, `47029`, `47030`) only built indices and
failed before inference due to a non-portable `simuleval` shebang. Corrected
jobs `47067` (Zh), `47068` (Ja), and `47072` (De) completed successfully. The
inherited output directory label contains `tau073`, but the runtime threshold
and every per-paper path record `th0.78`; `0.78` is the actual setting.

## Provenance limitation

The legacy paper-derived glossary files and hashes are retained, but the exact
historical Gemini model identifier was not stored beside the original
extraction. These results must therefore be described as the legacy
paper-derived glossary v1 condition, not as a newly reproduced Gemini 2.5 Flash
condition.
