# InfiniSST No-RAG Baseline

This note describes the reproducible InfiniSST baseline used as the comparison
point for RASST in the main result. The baseline runs the same 24 evaluation
cells, eval inputs, eval glossaries, and global cache policy as RASST, with
retrieval disabled.

## What "no RAG" means here

The eval agent
(`/mnt/taurus/data2/jiaxuanluo/RASST/code/rasst/eval/agents/infinisst_omni_vllm_maxsim_rag.py`)
already supports a native no-RAG mode: when no retriever and no oracle term-map
are configured, it builds a plain system prompt and never injects a `term_map`.
The baseline path makes the rest of the pipeline match:

- The serial driver
  (`/mnt/taurus/data2/jiaxuanluo/RASST/code/rasst/eval/eval_density_unified.sh`)
  honors `BASELINE_NO_RAG=1` (alias `RAG_MODE_OVERRIDE=none`). In that mode it
  skips the MaxSim index build, skips retriever/model-path validation, and runs
  SimulEval with no `--rag-enabled` and no term-map arguments. Cache policy,
  latency-multiplier math, GPU layout, and the offline StreamLAAL scoring step
  are unchanged.
- The orchestrator
  (`/mnt/taurus/data2/jiaxuanluo/RASST/code/rasst/tools/eval_main_result.py`)
  reads `metadata.common_eval_config.rag_mode`. When it is `none`, it drops the
  retriever from required assets and sets `BASELINE_NO_RAG=1` for every cell.

## Manifest and wrapper

```text
/mnt/taurus/data2/jiaxuanluo/RASST/code/rasst/manifests/main_result_baseline_no_rag.global_cache30_30_20_20.json
/mnt/taurus/data2/jiaxuanluo/RASST/code/rasst/scripts/eval_baseline.sh
```

The manifest is derived from
`main_result_eval.global_cache30_30_20_20.json`: it keeps the same
`(domain, lang, lm)` cells, eval inputs, and eval glossaries, but every cell
points at the single shared baseline model and ships no frozen canonical table
(baseline numbers are produced by running the eval).

## Baseline model

| Asset | Value |
| --- | --- |
| Manifest key | `model_infinisst_baseline` |
| Hugging Face | [gavinlaw/rasst-infinisst-baseline](https://huggingface.co/gavinlaw/rasst-infinisst-baseline) |
| RASST-local path | `/mnt/taurus/data2/jiaxuanluo/RASST/checkpoints/slm/infinisst_baseline/omni30b_sampling_r16_v3-20260121-021342-hf` |
| Frozen legacy path | `/mnt/gemini/data/jiaxuanluo/Omni-30B-sampling-0107/keep1.0_r16/v3-20260121-021342-hf` |

This is the InfiniSST rank-16 Omni sampling checkpoint (no term tagging). It is a
standard Hugging Face checkpoint directory with 15 safetensors shards, so it
resolves and validates with the same logic as the RASST release models.

## Common settings

| Setting | Value |
| --- | --- |
| RAG mode | `none` (retrieval disabled) |
| System prompt style | `translate_task` (historical InfiniSST wording) |
| Output tags stripped before scoring | `term_t` |
| Cache policy | `lm=1,2 -> 30/30`, `lm=3,4 -> 20/20` |
| max_new_tokens | fixed `40 * lm` |
| max_model_len | `12288` |
| Runner | `serial_simuleval` |

Term accuracy is still scored against the same eval glossaries as RASST, so the
baseline TERM_ACC is directly comparable.

## Reproduce

Validate the manifest, model, inputs, and glossaries on the current host:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
bash code/rasst/scripts/eval_baseline.sh --validate-only
```

Print one baseline cell without launching (a no-RAG SimulEval command):

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
bash code/rasst/scripts/eval_baseline.sh --dry-run \
  --domain acl_tagged_raw --lang zh --lm 1 \
  --cache-chunks-by-lm 1:30/30,2:30/30,3:20/20,4:20/20
```

Launch the full baseline through Slurm after checking the dry run:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
RASST_ALLOW_LAUNCH=1 bash code/rasst/scripts/eval_baseline.sh --sbatch \
  --cache-chunks-by-lm 1:30/30,2:30/30,3:20/20,4:20/20
```

Completed runs write each cell's `eval_results.tsv`, `instances.log`, and
`instances.strip_term.log`, plus `summary_all.tsv` and `config_report.md` under
the chosen run root. Compare baseline BLEU/TERM_ACC against the RASST canonical
table in
`/mnt/taurus/data2/jiaxuanluo/RASST/docs/results/main_result_global_cache30_30_20_20/`.

## Upload the baseline model (maintainer)

The baseline model is also registered in the release-assets manifest, so the
generic uploader/downloader handle it:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST

# Dry-run first.
python code/rasst/tools/hf_release_assets.py upload --asset model_infinisst_baseline

# Execute the upload (creates the public repo if needed).
RASST_ALLOW_HF_UPLOAD=1 python code/rasst/tools/hf_release_assets.py \
  upload --asset model_infinisst_baseline --execute
```
