# RASST Speech-LLM SFT Dataset And Recipe

This note describes the release-facing Speech-LLM (SLM) supervised fine-tuning
(SFT) dataset for RASST and how it is produced and published. The dataset
covers `de`, `ja`, and `zh` with the cap16 denoise-budget term-tagging recipe.

Only JSONL metadata and stats/recipe are published. **Audio is not
redistributed.**

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
`/mnt/taurus/data2/jiaxuanluo/RASST/code/rasst/manifests/slm_training.cap16_denoise_budget_ttag.json`.
The dataset export is described by
`/mnt/taurus/data2/jiaxuanluo/RASST/code/rasst/manifests/slm_training_dataset.cap16_denoise_budget_ttag.json`.

## What is published

For each language (`de`, `ja`, `zh`), under a per-language subfolder:

- `train_s_<lang>_..._ttag_exactboundary.jsonl` — 12,500 SFT training rows.
- `dev_s_<lang>_..._ttag_exactboundary_first355.jsonl` — 355 held-out dev rows.
- `*_wrap_stats.json`, `validation_summary.json`, and (ja/zh)
  `runtime_termmap_budget_schedule.json` — data-prep stats.

Plus, at the dataset root:

- `dataset_manifest.json` — recipe, repo id, and per-language row/audio counts.
- `audio_sources.json` — GigaSpeech provenance and the per-language inventory of
  referenced GigaSpeech audio IDs (no internal paths).
- `README.md` — dataset card.

HF dataset repo: `gavinlaw/rasst-speech-llm-sft-cap16-denoise-ttag` (one repo,
per-language subfolders).

## Each JSONL row

Rows follow the Qwen-Omni chat SFT format:

- `messages`: `system` / `user` / `assistant` turns. The `user` turn holds an
  `<audio>` placeholder plus the per-chunk `term_map`. The `assistant` turn is
  the translation with terminology wrapped as `<t>...</t>`.
- `audios`: ordered list of audio clip references, one per `<audio>` placeholder.
- `gt_terms_by_chunk`, `denoise_budget_*`, `assistant_term_target_tagging`:
  term-selection and tagging metadata for the cap16 denoise-budget recipe.

## Audio policy (GigaSpeech)

The audio clips are derived from
[GigaSpeech](https://github.com/SpeechColab/GigaSpeech) and are **not** included
in the release. During export, each absolute `audios` path is rewritten to a
relative key:

```text
audio/<lang>/<gigaspeech_audio_id>/<window_seconds>/<index>.wav
```

`audio_sources.json` lists the referenced GigaSpeech audio IDs per language. To
train or reproduce, obtain the audio from GigaSpeech, segment it to match the
keys, and re-point the `audios` entries to your local clips. Respect the
GigaSpeech license for any audio you reconstruct.

The exporter also strips any other internal absolute mount/home path from the
JSONL and stats, and fails fast if any internal path would survive.

## Export and download

Stage the dataset locally (JSONL + stats only, audio rewritten):

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
bash code/rasst/scripts/upload_hf_slm_dataset.sh prepare
```

Validate the export logic quickly on a few rows per file:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
bash code/rasst/scripts/upload_hf_slm_dataset.sh prepare --max-rows 50 --force
```

Upload (dry-run by default; `--execute` plus `RASST_ALLOW_HF_UPLOAD=1` performs it):

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
bash code/rasst/scripts/upload_hf_slm_dataset.sh upload                  # dry-run
RASST_ALLOW_HF_UPLOAD=1 bash code/rasst/scripts/upload_hf_slm_dataset.sh upload --execute
```

Download into the ignored local path `data/slm_training/cap16_denoise_budget_ttag`:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
RASST_ALLOW_DOWNLOAD=1 bash code/rasst/scripts/upload_hf_slm_dataset.sh download --execute
```
