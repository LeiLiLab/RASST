# Reproduce The RASST Main Result

This is the release-facing path for reproducing the final RASST main result.
It is intentionally manifest-driven and dry-run first.

The active reproduction code lives under `code/rasst/`: SLM data preparation
and training under `slm/`, eval under `eval/`, retriever code under
`retriever/`, and table/figure generation under `analysis/main_result/`.
`code/legacy/` is kept only as frozen provenance.

## 1. Reproduce SLMs

The canonical SLM recipe is cap16 denoise-budget term tagging for all three
target languages:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
bash code/rasst/scripts/reproduce_slm.sh --lang all --stage all
```

The command above prints the exact data-preparation and training commands. It
does not launch long jobs. To launch detached jobs after reviewing the commands:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
RASST_ALLOW_LAUNCH=1 bash code/rasst/scripts/reproduce_slm.sh --lang all --stage all --launch
```

The manifest behind this wrapper is:

```text
/mnt/taurus/data2/jiaxuanluo/RASST/code/rasst/manifests/slm_training.cap16_denoise_budget_ttag.json
```

The older zh `new_v9` path is not the release-canonical recipe. It is preserved
only as reference provenance in `docs/reference/zh_new_v9_reference_only.md`.

## 2. Reproduce Evaluation Commands

The final global cache policy is:

```text
lm=1,2 -> max_chunks=keep_chunks=30
lm=3,4 -> max_chunks=keep_chunks=20
```

Print eval commands without launching:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
bash code/rasst/scripts/eval_main_result.sh --dry-run \
  --cache-chunks-by-lm 1:30/30,2:30/30,3:20/20,4:20/20
```

The default eval manifest is:

```text
/mnt/taurus/data2/jiaxuanluo/RASST/code/rasst/manifests/main_result_eval.global_cache30_30_20_20.json
```

## 3. Public Model Assets

The final eval manifest names public Hugging Face repositories for the release
SLMs and the HN1024 retriever checkpoint:

```text
gavinlaw/rasst-speech-llm-de-cap16-denoise-ttag
gavinlaw/rasst-speech-llm-ja-cap16-denoise-ttag
gavinlaw/rasst-speech-llm-zh-cap16-denoise-ttag
gavinlaw/rasst-retriever-hn1024
```

The eval data are released as a Hugging Face Datasets repository:

```text
gavinlaw/rasst-main-result-data
```

Download these model assets into ignored local `checkpoints/` paths:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
bash code/rasst/scripts/download_release_assets.sh --dry-run
RASST_ALLOW_DOWNLOAD=1 bash code/rasst/scripts/download_release_assets.sh --download
```

Download ACL6060 tagged and medicine eval inputs, glossaries, and referenced
audio into the ignored local `data/` path:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
bash code/rasst/scripts/download_release_data.sh --dry-run
RASST_ALLOW_DOWNLOAD=1 bash code/rasst/scripts/download_release_data.sh --download
```

Maintainers can refresh the public HF uploads from the source artifacts named in
the manifest:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
bash code/rasst/scripts/upload_hf_release_assets.sh --dry-run
RASST_ALLOW_HF_UPLOAD=1 bash code/rasst/scripts/upload_hf_release_assets.sh --upload
bash code/rasst/scripts/upload_hf_release_data.sh prepare --force
RASST_ALLOW_HF_UPLOAD=1 bash code/rasst/scripts/upload_hf_release_data.sh upload --execute
```

The eval launcher can also download missing HF-declared model assets on demand:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
RASST_AUTO_DOWNLOAD_ASSETS=1 bash code/rasst/scripts/eval_main_result.sh --validate-only
```

The model, retriever, glossary, text input, and referenced-audio assets are
therefore reproducible from public Hugging Face repositories.

Validate the submitted-paper exact manifest and source artifacts:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
bash code/rasst/scripts/eval_main_result.sh --validate-only
```

Launch through Slurm only after inspecting the dry-run output:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
RASST_ALLOW_LAUNCH=1 bash code/rasst/scripts/eval_main_result.sh --sbatch \
  --cache-chunks-by-lm 1:30/30,2:30/30,3:20/20,4:20/20
```

## 4. Compare To The Tracked Snapshot

The tracked release snapshot is:

```text
/mnt/taurus/data2/jiaxuanluo/RASST/docs/results/main_result_global_cache30_30_20_20/
```

The most important files are:

```text
main_result.tsv
rasst24.tsv
compare_vs_infinisst_and_paper.tsv
new_main_result_tagged_global_cache30_30_20_20.pdf
medicine_main_result_global_cache30_30_20_20.pdf
```

Runtime outputs should stay under ignored roots such as `outputs/`, `figures/`,
or `/mnt/taurus/data2/jiaxuanluo/RASST_release_runs`.
