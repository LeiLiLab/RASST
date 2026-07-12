# Multi-Scale Retriever End-to-End Ablation

This directory is the stable result package for the end-to-end counterpart to
the paper's retrieval-only multi-scale ablation. The experiment fixes the
released En-Zh tagged-ACL `lm=2` protocol and changes only the retriever's
training/inference window policy.

## Fixed protocol

- Dataset: all five tagged-ACL talks, En-Zh, latency multiplier `lm=2`.
- Glossary: fixed raw ACL glossary, 238 entries.
- Retrieval: top-k `10`, threshold `0.78`.
- Prompting: `given_chunks`, plain term map, omit empty term maps.
- Cache: `max_cache_chunks=30`, `keep_cache_chunks=30`.
- Decode limit: `max_new_tokens=80`.
- Speech LLM: released cap16-denoise En-Zh term-tagging model.
- Each final `instances.log` contains five JSONL records, one per talk.

The 1K/10K/100K conditions in the paper figure are retrieval-only glossary
scale conditions. They are not used as end-to-end glossary sizes here.

## Results

Lower StreamLAAL is better. The tracked source rows are under
[`artifacts`](artifacts/), and the compact machine-readable comparison is
[`summary.tsv`](summary.tsv).

| Variant | BLEU | StreamLAAL (ms) | StreamLAAL-CA (ms) | Term accuracy |
| --- | ---: | ---: | ---: | ---: |
| Multi-scale | **47.8780** | **1814.34** | 2566.76 | **801/890 (90.00%)** |
| Largest-infer | 47.8261 | 1868.37 | 2563.45 | 754/890 (84.72%) |
| Largest-train | 45.7960 | 1863.20 | **2435.17** | 658/890 (73.93%) |

Relative to Multi-scale, Largest-infer is nearly tied in BLEU (`-0.052`) but
has `54.03 ms` higher StreamLAAL and `5.28` percentage points lower term
accuracy. Largest-train loses `2.082` BLEU, has `48.86 ms` higher StreamLAAL,
and loses `16.07` percentage points of term accuracy.

The matched end-to-end comparison therefore supports the retrieval-only
conclusion: multi-scale inference combined with MFA-localized training is the
strongest overall configuration.

## Variant provenance

- **Multi-scale:** training run `lh1b88kw`; inference windows
  `2,3,4,5,6,7,8,10,12,16,20,24`.
- **Largest-infer:** the same `lh1b88kw` checkpoint with inference window `24`
  only.
- **Largest-train:** historical fixed-1.92-second dense training run
  `r5l4780c`, transformer pooling, no MaxSim.

Exact checkpoint paths, Slurm jobs, artifact hashes, and runtime paths are in
[`run_manifest.json`](run_manifest.json). The reproducible launcher is
[`20260711__multiscale_e2e_lm2_aries23.sh`](../../../code/rasst/eval/launchers/20260711__multiscale_e2e_lm2_aries23.sh).

The full `instances.log` files remain under the Aries runtime output root and
are not duplicated in git. Their exact paths, sizes, five-record validation,
and SHA-256 hashes are recorded in the manifest.
