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

## Author-confirmed default-setting comparison

All three languages use the default `lm=2` operating point. The
author-confirmed rebuttal readout is preserved in
[`author_reported_lm2_update.tsv`](author_reported_lm2_update.tsv), and the
combined machine-readable table is [`comparison.tsv`](comparison.tsv).

| Language | Reported TERM_ACC RASST / InfiniSST (delta) | Pooled correct counts | BLEU RASST / InfiniSST (delta) |
| --- | ---: | ---: | ---: |
| En-Zh | 77.87 / 75.17 (**+2.70 pp**) | 693/890 vs. 669/890 | 46.3280 / 45.8268 (**+0.5012**) |
| En-Ja | 65.32 / 65.96 (**-0.64 pp**) | 614/940 vs. 620/940 | 27.7656 / 27.7202 (**+0.0455**) |
| En-De | 70.91 / 68.21 (**+2.70 pp**) | 652/935 vs. 632/935 | 29.2086 / 30.2743 (**-1.0657**) |

The author-confirmed reported TERM_ACC macro delta is `+1.59 pp`. The
paper-derived index retains a terminology advantage in Zh and De and is near
parity in Ja. BLEU changes by `+0.5012/+0.0455/-1.0657` for Zh/Ja/De, so this
supports terminology robustness under non-oracle preparation rather than a
uniform translation-quality gain.

For En-De, reported TERM_ACC and pooled counts are distinct aggregations:
`652/935 = 69.73%` and `632/935 = 67.59%` at the pooled level, whereas the
author-confirmed reported values are `70.91%` and `68.21%`. They are kept in
separate columns and must not be joined with an equals sign.

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
- The new De headline BLEU and reported TERM_ACC are author-confirmed summary
  values. A matching De masked-BLEU/latency bundle was not supplied, so those
  cells are `N/A` in `comparison.tsv`; the earlier tracked auxiliary values
  are not mixed into the new row.

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
