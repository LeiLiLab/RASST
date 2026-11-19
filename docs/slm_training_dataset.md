# RASST Speech-LLM SFT Dataset And Recipe

This note describes the release-facing Speech-LLM (SLM) supervised fine-tuning
(SFT) dataset for RASST and how it is produced and published. The dataset
covers `de`, `ja`, and `zh` with the cap16 denoise-budget term-tagging recipe.

JSONL metadata, stats/recipe, and the exact GigaSpeech boundaries of every audio
clip are published. **Audio is not redistributed**; it can be rebuilt from a
GigaSpeech copy with `code/rasst/tools/reconstruct_slm_sft_audio.py`.

## Pipeline

```text
data-prep launcher  ->  per-language SFT JSONL (train/dev) + wrap stats
        |
        v
train launcher      ->  LoRA SFT on Qwen3-Omni-30B-A3B-Instruct
        |
        v
HF export           ->  released SLM checkpoints (gavinlaw/rasst-speech-llm-<lang>-cap16-denoise-ttag)
        |
        v
dataset export      ->  public SFT JSONL dataset (this note)
```

The reproduction recipe (launchers, provenance manifests, exported checkpoints)
is in
`code/rasst/manifests/slm_training.cap16_denoise_budget_ttag.json`.
The dataset export is described by
`code/rasst/manifests/slm_training_dataset.cap16_denoise_budget_ttag.json`.

## What is published

For each language (`de`, `ja`, `zh`), under a per-language subfolder:

- `train_s_<lang>_..._ttag_exactboundary.jsonl`: 12,500 SFT training rows.
- `dev_s_<lang>_..._ttag_exactboundary_first355.jsonl`: 355 dev rows. Their
  utterances also appear in the train file, so the dev split is not held out.
- `audio_clip_map.parquet`: one row per clip key with its GigaSpeech opus file,
  start sample, and length at 16 kHz.
- `aligned_utterances.tsv`: the utterance spans and target-side trajectories the
  clips were cut from.
- `*_wrap_stats.json`, `validation_summary.json`, and (ja/zh)
  `runtime_termmap_budget_schedule.json`: data-prep stats.

Plus, at the dataset root:

- `dataset_manifest.json`: recipe, repo id, and per-language row, clip, and
  utterance counts.
- `audio_sources.json`: GigaSpeech provenance and the per-language inventory of
  referenced GigaSpeech audio IDs (no internal paths).
- `README.md`: dataset card.

HF dataset repo: `gavinlaw/rasst-speech-llm-sft-cap16-denoise-ttag` (one repo,
per-language subfolders). Revision `a38219fd64faa40125fad5021edb880d27caa965`
adds the clip maps and utterance boundaries; the JSONL files are unchanged from
the first release. Export run:
`/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/hf_slm_audio_map/20261119T033612Z`.

## Each JSONL row

Rows follow the Qwen-Omni chat SFT format:

- `messages`: `system` / `user` / `assistant` turns. The `user` turn holds an
  `<audio>` placeholder plus the per-chunk `term_map`. The `assistant` turn is
  the translation with terminology wrapped as `<t>...</t>`.
- `audios`: ordered list of audio clip references, one per `<audio>` placeholder.
- `gt_terms_by_chunk`, `denoise_budget_*`, `assistant_term_target_tagging`:
  term-selection and tagging metadata for the cap16 denoise-budget recipe.

## Audio clips (GigaSpeech)

The audio clips are cut from [GigaSpeech](https://github.com/SpeechColab/GigaSpeech)
and are **not** included in the release. During export, each absolute `audios`
path is rewritten to a relative key:

```text
audio/<lang>/<gigaspeech_audio_id>/<utterance_index>/<clip_index>.wav
```

`<gigaspeech_audio_id>_<utterance_index>` is the utterance id of the aligned TSV
the clips were cut from (`train_xl_case_robust_asr-filtered_<lang>_metricx-qe3.0_align.tsv`).
Each utterance is a span `<opus>:<start_sample>:<num_samples>` of a GigaSpeech
recording at 16 kHz, split into `trajectory_len` equal base chunks (one per
target trajectory entry; 0.96 s, 15,360 samples, for every utterance here). Clip `i` covers base chunks
`[i*m, min((i+1)*m, trajectory_len))`, where `m` is the clip's
`chunk_multiplier`. This is the rule in
`code/legacy/documents/code/tools/rebuild_siqi_audios.py`, which wrote the
training clips.

The export reads `m` from the first clip of each row on disk and checks every
clip length against the boundaries before writing:

- `<lang>/audio_clip_map.parquet`: `key`, `utter_id`, `gigaspeech_audio_id`,
  `gigaspeech_path` (path inside the GigaSpeech release), `clip_index`,
  `chunk_multiplier`, `start_sample`, `num_samples`, `start_sec`,
  `duration_sec`, `pad_samples` (zeros past the end of the recording).
- `<lang>/aligned_utterances.tsv`: `id`, `audio` (same format as the original
  TSV, with the GigaSpeech-relative path), `n_frames`, `trajectory_len`,
  `chunk_multiplier`, `num_clips`, `splits`, `tgt_text`, `trajectory`. English
  transcripts are part of the GigaSpeech database and are left out.

In `de` and `ja`, the JSONL field `merge_multiplier` is 1 on rows whose clips were
cut with larger multipliers (11,812 of 12,855 `de` rows, 11,772 of 12,855 `ja`
rows; `zh` has none). The models were trained on the clips, so
`chunk_multiplier` in the clip map is the value that describes the audio.

To rebuild the clips, get the full-length opus files from the official
GigaSpeech download and run:

```bash
python code/rasst/tools/reconstruct_slm_sft_audio.py \
  --dataset-dir data/slm_training/cap16_denoise_budget_ttag \
  --gigaspeech-root /path/to/GigaSpeech
```

Clips land at `<dataset-dir>/<key>`, which is where the JSONL `audios` keys
point. Starts and lengths match the training clips exactly. Sample values can
differ slightly across Opus decoder builds: rebuilding 450 sampled clips on
Taurus gave identical lengths for all 450, bit-identical audio for 215, and at
least 35.9 dB SNR (median 57 dB) for the rest. Respect the GigaSpeech terms for
any audio you rebuild.

The exporter also strips any other internal absolute mount/home path from the
JSONL and stats, and fails fast if any internal path would survive.

## Export and download

Stage the dataset locally (JSONL + stats only, audio rewritten):

```bash
cd RASST
bash code/rasst/scripts/upload_hf_slm_dataset.sh prepare
```

Validate the export logic quickly on a few rows per file:

```bash
cd RASST
bash code/rasst/scripts/upload_hf_slm_dataset.sh prepare --max-rows 50 --force
```

Upload (dry-run by default; `--execute` plus `RASST_ALLOW_HF_UPLOAD=1` performs it):

```bash
cd RASST
bash code/rasst/scripts/upload_hf_slm_dataset.sh upload                  # dry-run
RASST_ALLOW_HF_UPLOAD=1 bash code/rasst/scripts/upload_hf_slm_dataset.sh upload --execute
```

Download into the ignored local path `data/slm_training/cap16_denoise_budget_ttag`:

```bash
cd RASST
RASST_ALLOW_DOWNLOAD=1 bash code/rasst/scripts/upload_hf_slm_dataset.sh download --execute
```
