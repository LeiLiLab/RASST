# En-Zh Target-Tag Training Ablation

This package measures the effect of wrapping terminology translations in the
Speech LLM training targets with `<t>...</t>`. The no-tag model was retrained
from the same Qwen3-Omni base after removing only those delimiters from the
final En-Zh cap16 denoise-budget train and development JSONLs.

## Result

Across latency multipliers 1-4, target-tag supervision improves terminology
accuracy by `4.11` percentage points and real terminology adoption by `5.36`
points on average. Average BLEU changes by only `+0.08` in favor of the tagged
model. This supports the interpretation that target tags primarily strengthen
terminology adherence rather than general translation quality.

The no-tag model also has `2.53` points lower false-copy rate on average. The
tagged model therefore trades a modest increase in false copying for materially
higher correct terminology use.

All deltas below are `no-tag - tagged`.

| LM | Tagged BLEU | No-tag BLEU | Delta BLEU | Tagged TERM_ACC | No-tag TERM_ACC | Delta TERM_ACC | Delta StreamLAAL (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 43.8652 | 45.3446 | +1.4794 | 0.8865 | 0.8494 | -0.0371 | +98.7 |
| 2 | 48.7921 | 48.9172 | +0.1251 | 0.8899 | 0.8539 | -0.0360 | +35.7 |
| 3 | 50.8150 | 49.9595 | -0.8554 | 0.9079 | 0.8674 | -0.0405 | +13.7 |
| 4 | 50.7421 | 49.6701 | -1.0720 | 0.8989 | 0.8483 | -0.0506 | -48.0 |
| Mean | 48.5536 | 48.4728 | -0.0807 | 0.8958 | 0.8548 | -0.0411 | +25.0 |

`summary.tsv` contains the compact numeric comparison.
`comparison.full_provenance.tsv` preserves the original artifact paths and all
reported metrics.

## Experimental Control

Held fixed:

- En-Zh cap16 denoise-budget examples, row order, audio paths, user prompts,
  term maps, assistant text, and local target-boundary rewrites;
- LoRA rank/alpha `32/32`, global batch size `4`, maximum length `3072`, one
  epoch, optimizer, learning rate, and scheduler;
- ACL6060 tagged raw evaluation, HN1024 retriever, top-k `10`, threshold
  `0.78`, timeline lookback `1.92` seconds, `given_chunks` prompt, and omitted
  empty term maps;
- cache `30/30` for LM1-2 and `20/20` for LM3-4, with
  `max_new_tokens = 40 * LM`.

Changed:

- assistant targets contain no `<t>` or `</t>` delimiters;
- the tagged reference run used four GPUs, while the no-tag run used two GPUs;
  both used TP2, EP2, sequence parallelism, and global batch size 4;
- one packed epoch is 628 optimizer steps for tagged data and 591 for no-tag
  data because removing the delimiters shortens the packed token stream.

The data transformation was validated row by row: all non-message fields and
all non-assistant messages are identical, while each no-tag assistant message
equals its tagged counterpart after deleting the two delimiters. The transform
removed 90,086 tag pairs from 12,500 training rows and 2,490 pairs from 355
development rows.

## Models And Runs

- Tagged model: [gavinlaw/rasst-speech-llm-zh-cap16-denoise-ttag](https://huggingface.co/gavinlaw/rasst-speech-llm-zh-cap16-denoise-ttag)
- Tagged training: W&B `ccgjhu4r`, 628 optimizer steps
- No-tag training: W&B [`60n5gmzs`](https://wandb.ai/luojiaxuan1215-johns-hopkins-university/sst_omni/runs/60n5gmzs), 591 optimizer steps
- No-tag HF export: 15 safetensor shards, approximately 66 GiB; the cluster
  path is recorded in `run_manifest.json`

## Artifact Layout

```text
artifacts/
  tagged/lm{1,2,3,4}/
    eval_results.tsv
    instances.log
    instances.strip_term.log
  notag/lm{1,2,3,4}/
    eval_results.tsv
    instances.log
    instances.strip_term.log
```

Each `instances.log` and `instances.strip_term.log` has five rows, one per
ACL6060 talk. `eval_results.tsv` was recomputed from the corresponding instance
log with the same StreamLAAL and terminology scorer. `sha256sums.txt` covers
every tracked result artifact in this directory.
