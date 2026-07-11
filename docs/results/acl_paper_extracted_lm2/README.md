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

| Language | Slurm job | Dependency | Status at submission |
| --- | ---: | --- | --- |
| zh | `47067` | none | running |
| ja | `47068` | after `47067` | queued |
| de | `47069` | after `47068` | queued |

Verified results will be added here from `eval_results.tsv` only after all five
per-paper cells for a language finish successfully.
