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

The A6000 recipe is retained for direct comparability and data locality: the
verified MCore/HF bases and speech inputs are already available under
`/mnt/gemini`, whereas moving them to B200 storage would require copying roughly
190 GB plus the audio corpus. The launcher records the selected host explicitly
and places checkpoints and exports on a local `/mnt/<host>/data*` disk.

## Required evaluation

Evaluate the new model on the frozen ACL Japanese `lm=1` cell and report:

- BLEU, TERM_ACC, and StreamLAAL from the standalone evaluation artifacts;
- block-aware xCOMET using five consecutive reference sentences per segment;
- the corresponding original RASST and InfiniSST values under the same scoring
  protocol.

The actual host-qualified run path, generated training-data hash, and model
artifact location are recorded after the run. Their Hugging Face destinations
and revisions must be recorded after upload.
