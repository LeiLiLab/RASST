# Rebuttal：target-term-masked BLEU（global-cache snapshot）

本目录保存的是论文主结果 `global-cache 30/30/20/20` 的 masked-BLEU
快照，不是新的解码结果：`lm=1,2` 使用 `30/30`，`lm=3,4` 使用
`20/20`。评分先用 mWER 按原 reference 边界重切每个 talk-level
prediction，再从 hypothesis 和 reference 中同时删除 raw gold glossary
的 target translations，最后计算 corpus BLEU。这个指标用于检查：RASST
相对 InfiniSST 的 BLEU 增益是否只来自术语本身。

## 文件与来源

| 文件 | 内容 | 状态 |
| --- | --- | --- |
| `masked_terms_quality_global_cache30_30_20_20_snapshot.tsv` | 48 个方法行（2 methods × 2 tracks × 3 languages × 4 latency modes） | 48/48 source rows 为 `ok` |
| `masked_terms_quality_compare_vs_infinisst_global_cache30_30_20_20_snapshot.tsv` | 24 个严格配对的 RASST–InfiniSST 差值 | 24/24 pairs 为 `ok` |
| `code/rasst/analysis/rebuttal/target_term_masked_bleu.py` | 不依赖 dirty offline scorer 的单条结果评分器 | 所有输入路径与 mwerSegmenter executable 均由 CLI 显式传入 |
| `code/rasst/analysis/rebuttal/validate_masked_bleu_snapshot.py` | 对 summary 中所有行做全量复算 | 48/48 精确一致 |

Taurus 原始 summary/compare SHA-256 分别为
`3265c5b3c808c44f5227ebe5626b5a7ff194da6535267a17eb88f7e65604af14`
和 `ac06dec613a35a7c1cc1d04a56dbe7534834178ef229d89cb3357b6f2a727c6b`。
导入 Git 时把空的末列 `note` 写成 TSV 等价的 quoted empty string
`""`，避免 trailing whitespace；summary 还把 18 行 legacy
`/home/jiaxuanluo/...` 或 `/mnt/data1/...` provenance 改成
host-qualified `/mnt/taurus/home/jiaxuanluo/...` 或
`/mnt/taurus/data1/...`。导入后的 summary/compare SHA-256
分别为 `a0a3994462e024e6b7696872a970f3b1f31cfb05f294b1c41cfad955e616c08a`
和 `888af0375a8b9272c906bb011f258ea056d2fbf32da1ee76c0e8cfd83238da71`。
所有 metric 和 count 字段保持不变。

2026-07-10 在 Taurus 的 pinned `infinisst` 环境中，用本目录的新 scorer
重新读取了全部 48 个 instance logs。每个 log 含 5 个 talk-level
predictions，总计检查 240 个 talk-system predictions 和 45,720 个重切后的
segments。以下四项逐行比较均为 **48/48 exact、0 discrepancies**：
`MASKED_TERMS_BLEU`（四位小数）、hypothesis removed count、reference
removed count、term type count。验证 JSON 的 SHA-256 为
`712c510857bb5d59d605a3ebd5dd650e3855ec3de978250bb1be2aee88d98618`。

## Rebuttal 推荐统计口径

ESO/Medicine 的 zh/ja references 不进入推荐的 BLEU aggregate；这些 8 个
comparison rows 仍保留在快照中供审计。推荐口径只汇总 ACL6060 tagged
的全部 12 cells，以及有 human reference 的 ESO/Medicine de 4 cells。

| 口径 | Cells | Regular BLEU wins | Masked BLEU wins | Avg. regular BLEU delta | Avg. masked BLEU delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACL6060 tagged（de/ja/zh） | 12 | 10/12 | 10/12 | +1.9110 | +0.9919 |
| ESO/Medicine（de only） | 4 | 2/4 | 2/4 | +0.2351 | -0.2047 |
| **推荐 rebuttal aggregate** | **16** | **12/16** | **12/16** | **+1.4920** | **+0.6927** |

因此，在排除 ESO/Medicine zh/ja 后，术语被 mask 掉时平均增益仍为
`+0.6927 BLEU`，且 12/16 cells 为正；同时它小于 regular BLEU 的
`+1.4920`，说明术语正确性贡献了部分 BLEU 增益，但不是全部。需要说明，
masked BLEU 会同时改变句长和术语周围的 n-gram，所以它是 diagnostic
而不是严格的因果分解。

## 显式复现命令

下面是 ACL6060 En-Zh InfiniSST `lm=1` 的单行复现示例：

```bash
/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python \
  /mnt/taurus/data2/jiaxuanluo/RASST/code/rasst/analysis/rebuttal/target_term_masked_bleu.py \
  --instances-log /mnt/taurus/data2/jiaxuanluo/RASST/docs/results/main_result_global_cache30_30_20_20/artifacts/acl_tagged_raw_infinisst_zh/lm1/instances.log \
  --reference /mnt/taurus/data2/jiaxuanluo/RASST_release_runs/hf_datasets/rasst-main-result-data/main_result/inputs/acl_zh/ref.txt \
  --audio-yaml /mnt/taurus/data2/jiaxuanluo/RASST_release_runs/hf_datasets/rasst-main-result-data/main_result/inputs/acl_zh/audio.yaml \
  --glossary /mnt/taurus/data2/jiaxuanluo/RASST_release_runs/hf_datasets/rasst-main-result-data/glossaries/acl6060_tagged_gt_raw_min_norm2.json \
  --target-language zh \
  --sacrebleu-tokenizer zh \
  --latency-unit char \
  --mwer-segmenter /mnt/taurus/home/jiaxuanluo/mwerSegmenter/mwerSegmenter
```

48 行全量验证使用：

```bash
/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python \
  /mnt/taurus/data2/jiaxuanluo/RASST/code/rasst/analysis/rebuttal/validate_masked_bleu_snapshot.py \
  --summary-tsv /mnt/taurus/data2/jiaxuanluo/RASST/docs/results/rebuttal_2026/masked_terms_quality_global_cache30_30_20_20_snapshot.tsv \
  --release-data-root /mnt/taurus/data2/jiaxuanluo/RASST_release_runs/hf_datasets/rasst-main-result-data \
  --mwer-segmenter /mnt/taurus/home/jiaxuanluo/mwerSegmenter/mwerSegmenter \
  --output-json /mnt/taurus/data2/jiaxuanluo/RASST_release_runs/rebuttal_2026_masked_bleu_validation/global_cache30_30_20_20_validation.json
```

## Source of Truth

代码和轻量结果表以本 Git 仓库为准。评分输入来自
[gavinlaw/rasst-main-result-data](https://huggingface.co/datasets/gavinlaw/rasst-main-result-data)；
本快照未记录固定 HF revision。48 个 talk-level hypothesis logs 仍是 Taurus
local provenance，路径逐行记录在 summary TSV 中，尚未作为 reusable data
artifact 上传。全量复算 JSON 位于
`/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/rebuttal_2026_masked_bleu_validation/global_cache30_30_20_20_validation.json`，
它是临时验证产物，不替代 Git 中的 summary/compare source of truth。
