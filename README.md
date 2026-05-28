# RASST

This repository contains the release code, models, data links, and reproduction workflow for **RASST**, a retrieval-augmented streaming speech translation system for domain terminology.

The tracked paper PDF is [paper/acl_latex.pdf](paper/acl_latex.pdf).

## Main Results

RASST uses one global cache policy for all final main-result cells:

```text
lm=1,2 -> max_chunks=keep_chunks=30
lm=3,4 -> max_chunks=keep_chunks=20
```

On the final global-cache snapshot, RASST improves terminology accuracy over InfiniSST in all 24 evaluated cells, with positive BLEU deltas in 19/24 cells.

| Track | Avg. BLEU delta vs. InfiniSST | Avg. TERM_ACC delta vs. InfiniSST |
| --- | ---: | ---: |
| ACL6060 tagged | +1.911 | +0.170 |
| Medicine hard/raw | +1.564 | +0.358 |
| Overall | +1.737 | +0.264 |

![ACL6060 tagged main result](docs/results/main_result_global_cache30_30_20_20/new_main_result_tagged_global_cache30_30_20_20.png)

![Medicine main result](docs/results/main_result_global_cache30_30_20_20/medicine_main_result_global_cache30_30_20_20.png)

The tracked result tables and figure sources are in [docs/results/main_result_global_cache30_30_20_20](docs/results/main_result_global_cache30_30_20_20/).

## Release Assets

| Asset | Link |
| --- | --- |
| Eval data: ACL6060 tagged, medicine, glossaries, audio | [gavinlaw/rasst-main-result-data](https://huggingface.co/datasets/gavinlaw/rasst-main-result-data) |
| Retriever checkpoint | [gavinlaw/rasst-retriever-hn1024](https://huggingface.co/gavinlaw/rasst-retriever-hn1024) |
| SLM en-de | [gavinlaw/rasst-speech-llm-de-cap16-denoise-ttag](https://huggingface.co/gavinlaw/rasst-speech-llm-de-cap16-denoise-ttag) |
| SLM en-ja | [gavinlaw/rasst-speech-llm-ja-cap16-denoise-ttag](https://huggingface.co/gavinlaw/rasst-speech-llm-ja-cap16-denoise-ttag) |
| SLM en-zh | [gavinlaw/rasst-speech-llm-zh-cap16-denoise-ttag](https://huggingface.co/gavinlaw/rasst-speech-llm-zh-cap16-denoise-ttag) |

Download all public release assets into ignored local paths:

```bash
git clone https://github.com/luojiaxuan/RASST.git
cd RASST

RASST_ALLOW_DOWNLOAD=1 bash code/rasst/scripts/download_release_data.sh --download
RASST_ALLOW_DOWNLOAD=1 bash code/rasst/scripts/download_release_assets.sh --download
```

This populates:

```text
data/         # eval inputs, glossaries, and referenced audio
checkpoints/  # SLM and retriever checkpoints
```

## Installation

The release scripts are written for Linux with CUDA GPUs. The reference cluster is Taurus, but the public assets can be downloaded on any machine that can run the required GPU stack.

```bash
conda create -n rasst -y python=3.10
conda activate rasst
pip install uv

uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

uv pip install transformers==4.47.0 accelerate peft deepspeed sentence-transformers \
  huggingface_hub numpy pandas scipy scikit-learn tqdm pyyaml soundfile librosa \
  sacrebleu evaluate jiwer simuleval matplotlib tensorboardX wandb faiss-cpu

# Required for batched vLLM inference.
uv pip install vllm
```

Some training launchers use the original Megatron/Swift Docker path. For exact SLM retraining, inspect the generated command first and run on a Slurm/Docker-capable GPU node.

## Evaluation And Inference

Validate that the manifest, downloaded data, checkpoints, and frozen result artifacts resolve:

```bash
bash code/rasst/scripts/eval_main_result.sh --validate-only --strict-metrics
```

Print all main-result eval commands without launching:

```bash
bash code/rasst/scripts/eval_main_result.sh --dry-run \
  --cache-chunks-by-lm 1:30/30,2:30/30,3:20/20,4:20/20
```

Print one cell only:

```bash
bash code/rasst/scripts/eval_main_result.sh --dry-run \
  --domain acl_tagged_raw --lang de --lm 3 \
  --cache-chunks-by-lm 1:30/30,2:30/30,3:20/20,4:20/20
```

Launch the full eval through Slurm after checking the dry run:

```bash
RASST_ALLOW_LAUNCH=1 bash code/rasst/scripts/eval_main_result.sh --sbatch \
  --cache-chunks-by-lm 1:30/30,2:30/30,3:20/20,4:20/20
```

By default, runtime outputs are written under ignored paths such as `outputs/`, `logs/`, `figures/`, and `checkpoints/`.

## Training

The release-facing SLM recipe is cap16 denoise-budget term tagging for `de`, `ja`, and `zh`. The wrapper is dry-run by default:

```bash
bash code/rasst/scripts/reproduce_slm.sh --lang all --stage all
```

Prepare data only:

```bash
bash code/rasst/scripts/reproduce_slm.sh --lang all --stage prepare
```

Print training commands only:

```bash
bash code/rasst/scripts/reproduce_slm.sh --lang all --stage train
```

Launch detached SLM data-prep/training jobs only after reviewing the printed commands:

```bash
RASST_ALLOW_LAUNCH=1 bash code/rasst/scripts/reproduce_slm.sh \
  --lang all --stage all --launch
```

Retriever training and MaxSim index construction are exposed separately:

```bash
bash code/rasst/scripts/train_retriever.sh --dry-run
bash code/rasst/scripts/build_index.sh --dry-run
```

Launch them only after checking paths and resources:

```bash
RASST_ALLOW_LAUNCH=1 bash code/rasst/scripts/train_retriever.sh
RASST_ALLOW_LAUNCH=1 bash code/rasst/scripts/build_index.sh
```

## Code Layout

The active release code lives under `code/rasst/`:

```text
code/rasst/slm/                  SLM data preparation and training launchers
code/rasst/retriever/            retriever training and MaxSim index/runtime code
code/rasst/eval/                 serial SimulEval, batched vLLM eval, scorer, agent
code/rasst/analysis/main_result/ main-result table and figure builders
code/rasst/manifests/            release manifests
code/rasst/scripts/              public launch/download wrappers
```

`code/legacy/` is kept as frozen provenance from the original InfiniSST-derived workspace. New users should start with the commands above rather than launching from `code/legacy/`.

## Contact

Please raise GitHub issues for questions about reproducing the release results.
