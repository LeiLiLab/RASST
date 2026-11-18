# RASST Wiki-Synthetic Retriever Dataset

The RASST speech-to-term retriever (`gavinlaw/rasst-retriever-hn1024`) is
trained on GigaSpeech-derived chunks plus wiki-synthetic terminology chunks.
This note covers the wiki-synthetic part, which is published in full, audio
included:

HF dataset repo: [`gavinlaw/rasst-retriever-wiki-synth-en`](https://huggingface.co/datasets/gavinlaw/rasst-retriever-wiki-synth-en),
revision `713bacbc9bdcd7ea8d8df43625c00fb622f08399` (public, CC BY-NC 4.0, 222.4 GB:
193 audio shards plus three parquet tables).

The GigaSpeech-derived rows are not published because GigaSpeech audio cannot be
redistributed.

## Pipeline

```text
Wikidata truthy dump -> ranked term list (1M terms, P31 inverse-type-frequency ranking)
        |
        v
Gemini-2.5-Flash     -> 3 sentences per term that contain it verbatim (2,998,703 sentences)
        |
        v
CosyVoice 3          -> speech for every sentence, GigaSpeech speaker prompts
        |
        v
MFA + chunk cutting  -> 2,893,705 chunks of 2.88 / 3.84 / 4.80 / 5.76 s with the term span
```

The original builders and launchers are under
`code/legacy/documents/code/data_pre/wiki_synth/` (see `PIPELINE.md` there).
The speaker prompt pool is public at
[`gavinlaw/gigaspeech_speaker_prompts_v2`](https://huggingface.co/datasets/gavinlaw/gigaspeech_speaker_prompts_v2).

## What is published

| Path | Content |
| --- | --- |
| `terms/wikidata_terms_top1m.parquet` | `term`, `term_key`, `short_description`, `p31_rank`, `primary_p31_qid`, `primary_p31_label` |
| `utterances/wiki_synth_utterances.parquet` | `utter_id`, `term`, `term_key`, `short_description`, `p31_rank`, `variant_idx`, `utterance`, `has_train_chunk`, `in_hn1024_train` |
| `retriever_train/index.parquet` | one row per training chunk with its shard name |
| `retriever_train/audio/train-NNNNN.tar` | WebDataset shards, `<key>.flac` + `<key>.json` |
| `dataset_manifest.json` | counts, packaging, provenance, source commit |

Each training row carries `key`, `utter_id`, `term`, `term_key`, `p31_rank`,
`utterance`, `variant_idx`, `chunk_text`, `chunk_idx`, `context_duration_sec`,
and the MFA term span in seconds (`mfa_term_start_in_chunk`,
`mfa_term_end_in_chunk`, `mfa_term_duration`, `mfa_term_full_duration`), plus
`in_hn1024_train`.

`in_hn1024_train` marks the rows the released retriever actually trained on. The
trainer (W&B run `lh1b88kw`, `--wiki_rank 1000000`) skips rows whose `p31_rank` is
at least 1,000,000, which kept 2,567,645 of the 2,893,705 wiki rows. `p31_rank`
comes from `wiki_synth_terms_p31_ranked.jsonl`, the ranking the training-data
builder attached; rank -1 means the term has no rank and the row is kept.

`utter_id` (`wiki_synth_NNNNNNN`) is the row number of the sentence in the
`utterances` table, so training rows join to their full sentence directly. The
sample key is `<utter_id>_ctx<context>_chunk<chunk_idx>`.

Audio is 16 kHz mono. The FLAC files are a lossless re-encode of the PCM16 WAV
chunks the trainer read, so decoding returns the exact training samples.

## Source of truth

| Item | Value |
| --- | --- |
| HF repo | `gavinlaw/rasst-retriever-wiki-synth-en` |
| Revision | `713bacbc9bdcd7ea8d8df43625c00fb622f08399` |
| Export code | `code/rasst/tools/hf_release_wiki_synth_dataset.py` at commit `bfdba4c` |
| Export run | `/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/hf_wiki_synth/20261118T205708Z` (`launch.json`, `logs/`, `stage/_work/uploaded.jsonl`) |
| Verification | all 193 remote shard sha256 and sizes match the build records; 300 sampled chunks of `train-00000` decode bit-exact to the source WAV |
| Source data | `/mnt/gemini/...` paths listed in `code/rasst/manifests/retriever_wiki_synth_dataset.json` |

## Export, upload, download

The export is described by
`code/rasst/manifests/retriever_wiki_synth_dataset.json` and implemented in
`code/rasst/tools/hf_release_wiki_synth_dataset.py`.

```bash
cd RASST
STAGE=/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/hf_wiki_synth/<UTC>/stage

# Write the parquet tables, card, and per-shard build plans.
bash code/rasst/scripts/upload_hf_wiki_synth_dataset.sh prepare --stage-root "${STAGE}"

# Build tar shards batch by batch, commit them, check remote sizes, delete local tars.
# Re-running resumes from the shards already on the Hub.
RASST_ALLOW_HF_UPLOAD=1 bash code/rasst/scripts/upload_hf_wiki_synth_dataset.sh \
  upload --stage-root "${STAGE}" --execute
```

Download the tables only, or everything (about 220 GB), into the ignored path
`data/retriever_training/wiki_synth`:

```bash
RASST_ALLOW_DOWNLOAD=1 bash code/rasst/scripts/upload_hf_wiki_synth_dataset.sh download --execute --text-only
RASST_ALLOW_DOWNLOAD=1 bash code/rasst/scripts/upload_hf_wiki_synth_dataset.sh download --execute
```

## License

CC BY-NC 4.0. Terms come from Wikidata (CC0). Sentences were generated with
Gemini 2.5 Flash and may contain factual errors about the entities they mention.
Speech was synthesized with Fun-CosyVoice3-0.5B-2512 from GigaSpeech speaker
prompts, and GigaSpeech audio is licensed for non-commercial research and
education.
