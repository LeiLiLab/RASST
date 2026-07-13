# Ja `lm=1` curriculum Speech LLM retraining

## Objective

Test whether the Japanese `lm=1` quality regression is partly caused by sparse
short-chunk training coverage. The intervention is intentionally narrow: keep
the original 12,500-row Ja SFT set unchanged and append one independently
denoised version of every training row whose audio chunks all use latency
multiplier 1.

## Data intervention

- Base data: 12,500 rows, including 1,048 all-`lm=1` rows (8.38%).
- Supplement: the same 1,048 all-`lm=1` source/target rows, rebuilt from the
  raw retriever records with denoise seed 43.
- Final curriculum: 13,548 rows, including 2,096 all-`lm=1` rows (15.47%).
- Denoise budgets, score-dropout policy, assistant `<t>` targets, raw speech,
  references, and dev data remain identical to the original recipe.

The supplement is not a byte-identical duplication: its term-map distractors
are independently sampled. This preserves the supervised target while exposing
the model to another short-chunk retrieval context.

## Frozen training recipe

- Base model: `Qwen3-Omni-30B-A3B-Instruct-v2` MCore export.
- LoRA: rank 32, alpha 32.
- Topology: 4 A6000 GPUs, TP=2, EP=2.
- Batch: micro batch 1, global batch 4.
- Sequence length: 3072.
- Epochs: 1.
- Validation set: the original 355-row Ja cap16 denoise-budget `<t>` dev set.
- HF conversion: direct single-copy export beside the MCore adapter checkpoint;
  staged and cache duplicates are disabled to avoid tripling the 66 GB export.
- W&B mode: offline; standalone logs and verified checkpoint/eval artifacts are
  authoritative for this run.

The A6000 recipe is retained for direct comparability and data locality: the
verified MCore/HF bases and speech inputs are already available under
`/mnt/gemini`, whereas moving them to B200 storage would require copying roughly
190 GB plus the audio corpus. The launcher records the selected host explicitly
and places checkpoints and exports on a local `/mnt/<host>/data*` disk.

For the Aries run, the 60 GB MCore base is staged once to local `/mnt/data3`.
Direct loading from shared `/mnt/gemini` left all four GPUs at 0% with a
per-rank ETA up to 50 minutes; local staging removes that repeated-read
bottleneck. The larger HF export remains on local `/mnt/data6`.

## Completed training and export

- Host: Aries, GPUs 0--3, TP=2 / EP=2.
- Run: `/mnt/aries/data6/jiaxuanluo/RASST_release_runs/ja_lm1_curriculum_20260713/checkpoints/keep1.0_r32/v3-20260713-114017`.
- Training-data SHA-256: `fbab3f4c3ffb080d6c711d44c522ae051b2691d2c6b2bae6de087108a48430b0`.
- Completed steps: `573/573`; final dev `eval_lm loss=0.29417345`.
- Final training log SHA-256: `78b79c290f8e73cacf7e9630f29c7491ea4355acec2310db839d2204e645f45f`.
- Structured logging SHA-256: `7a3b68ac87fc7b210ceb840455e001863e09739c9308bc0565f2d559e48f0d99`.
- HF export: `/mnt/aries/data6/jiaxuanluo/RASST_release_runs/ja_lm1_curriculum_20260713/checkpoints/keep1.0_r32/v3-20260713-114017-hf`.
- Export validation: 15 `safetensors` shards, 28,010 indexed weights;
  `config.json` SHA-256 `e63c4cc3acc787d38ae818a416a53fcff684d60e8db497b30328e1f85bbd6401`,
  index SHA-256 `03ffe9a54efe9007ceb788c2b036f5c589094918c4582add4e92bc17e1cba838`.

## ACL Japanese `lm=1` evaluation

The frozen ACL cell uses the same raw-gold glossary, retriever, top-k 10,
threshold 0.78, cache policy, prompt, and decoding settings as the original
RASST point. The canonical post-evaluation scorer uses mWER segmentation and
strips the explicit term tags. The raw SimulEval BLEU printed during inference
is not used.

- BLEU: `23.892923864689116`.
- Masked BLEU: `20.424150161457373`.
- TERM_ACC: `0.7777` (`731/940`).
- StreamLAAL: `1522.4505785846513 ms`.
- StreamLAAL_CA: `2286.170651611981 ms`.
- `eval_results.tsv` SHA-256: `73ccbeb5eae456b139286dace647d7673ef6ebd25057a85a647f97e288a932c4`.
- `instances.log` SHA-256: `a754df2c09327bc822fc4d8e6c6afbc207e3961c39e75d1783ddc5c61b3abdb8`.
- `instances.strip_term.log` SHA-256: `59d9ddaed5a55b0c2caa62a03fc446d72b0190b0f2c14a821db51c4f3627b0aa`.

The requested xCOMET readout is sentence-level: one source/reference sentence
per segment after mWER resegmentation (`sentences_per_segment=1`). A
block-aware score is not part of this experiment's final result.

- InfiniSST: `63.2560745885` xCOMET points.
- Original RASST: `59.1526793158` (`-4.1033952727` vs. InfiniSST).
- `lm=1` curriculum RASST: `61.1241974971` (`+1.9715181813` vs.
  original RASST; `-2.1318770914` vs. InfiniSST).
- Validation: 3 systems, 1 strict pair, 1,404 segments, status `ok`.
- Full segments SHA-256: `6ea75fd94d72dba025b8fb16c7c16ec93011018e2642b836d3c3c30505a130d0`.
- Validation report SHA-256: `0e3a6f3e79e312324835592d05a04ac5bb01dadb1fd2b3875f1844d12750f30a`.

## Artifact destinations

- Intended training-data dataset: `gavinlaw/rasst-speech-llm-sft-cap16-denoise-ttag`,
  revision/tag `ja-lm1-curriculum-seed43`.
- Intended model repo: `gavinlaw/rasst-speech-llm-ja-cap16-denoise-ttag`,
  revision/tag `lm1-curriculum-seed43`.
- Upload status: **pending** because no authorized Hugging Face write credential
  is available. The Aries paths above are temporary staging, not canonical
  reusable artifacts.
