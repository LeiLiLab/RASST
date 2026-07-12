# ACL Paper-Extracted Realistic-Glossary Evaluation

This directory is the stable entry point for the rebuttal experiment that
replaces the tagged-ACL glossary with terminology extracted from each talk's
paper. The five tracked glossaries are under
[`data/glossaries/acl_paper_extracted`](../../../data/glossaries/acl_paper_extracted/).

## Fixed protocol

- Languages: En-Zh, En-Ja, En-De.
- Latency multiplier: `lm=2` only.
- Five ACL6060 talks, each paired with its own paper-derived glossary.
- RASST uses the released cap16-denoise term-tagging SLM for each language and
  the released HN1024 retriever with top-k `10` and threshold `0.78`.
- Prompting: `given_chunks`, plain term map, omit empty term maps.
- Cache: `max_cache_chunks=30`, `keep_cache_chunks=30`.
- Decode limit: `max_new_tokens=80`.
- RASST metrics are aggregated across five independent per-paper runs.
- InfiniSST is not rerun. Its existing `lm=2` `instances.log` for each language
  is rescored by paper using the same five glossaries.
- `TERM_ACC` in this directory is a matched-glossary diagnostic: its denominator
  contains occurrences covered by the corresponding paper-derived glossary.
  It is not the full raw-gold ACL denominator and must not be described as
  paper-glossary recall or raw-gold coverage.

## Unified comparison

All three languages are complete and verified. The exact machine-readable table
is [`comparison.tsv`](comparison.tsv).

| Language | System | BLEU | Masked-term BLEU | StreamLAAL | TERM_ACC |
| --- | --- | ---: | ---: | ---: | ---: |
| En-Zh | InfiniSST | 45.8268 | 43.7725 | 1765.72 | 188/235 (80.00%) |
| En-Zh | RASST | 46.1716 | 43.2382 | 1827.82 | 223/235 (94.89%) |
| En-Ja | InfiniSST | 27.7202 | 27.1852 | 2291.16 | 126/185 (68.11%) |
| En-Ja | RASST | 27.6944 | 27.4811 | 2104.52 | 159/185 (85.95%) |
| En-De | InfiniSST | 30.2516 | 30.3218 | 1759.06 | 151/236 (63.98%) |
| En-De | RASST | 27.8001 | 27.7735 | 1642.45 | 210/236 (88.98%) |

RASST minus InfiniSST exact-form terminology accuracy is `+14.89`, `+17.84`,
and `+25.00` percentage points for En-Zh, En-Ja, and En-De, respectively. BLEU
changes by `+0.3448`, `-0.0257`, and `-2.4515`. This supports a terminology
handling claim under paper-derived preparation, but not a uniform overall
translation-quality improvement.

## Artifact layout and validation

- RASST aggregates: [`zh/aggregate.tsv`](zh/aggregate.tsv),
  [`ja/aggregate.tsv`](ja/aggregate.tsv), and
  [`de/aggregate.tsv`](de/aggregate.tsv). Each language also has five source
  rows under its `per_paper` directory.
- InfiniSST post-eval outputs: [`infinisst/zh`](infinisst/zh),
  [`infinisst/ja`](infinisst/ja), and [`infinisst/de`](infinisst/de). The JSON
  `posteval_details.json` files record the per-paper counts used in each
  aggregate.
- The InfiniSST post-eval reproduces the existing BLEU and StreamLAAL values to
  full precision. This confirms that only glossary-sensitive metrics changed.
- Both systems use identical denominators within each language: `235` for Zh,
  `185` for Ja, and `236` for De.
- InfiniSST has no retained runtime retrieval JSONL, so runtime-gated
  `REAL_TERM_ADOPT` is `N/A`; `TERM_ACC`, masked-term BLEU, adoption against the
  fixed glossary, false-copy diagnostics, BLEU, and latency remain available.

The first RASST submission (`47028`, `47029`, `47030`) only built indices and
failed before inference due to a non-portable `simuleval` shebang. The corrected
jobs `47067` (Zh), `47068` (Ja), and `47072` (De) all completed successfully.
The inherited output directory label contains `tau073`, but the runtime
threshold and every per-paper path record `th0.78`; `0.78` is the actual setting.

## Provenance limitation

The glossary files and their hashes are retained, but the exact historical
Gemini model identifier was not stored beside the original extraction. The
manifest records the historical tool default without claiming that it is
verified. Therefore these results must be described as the legacy
paper-derived glossary v1 condition, not as a newly reproduced Gemini 2.5 Flash
condition.
