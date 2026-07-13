# ACL `lm=1/2` terminology-type outcome analysis

状态：**自动统计完成。taxonomy 仅由 English glossary surface form 决定；`lm=1`
不包含新的语义人工标签。** 本分析只使用 ACL 60/60 五个 talks，明确排除
Medicine/ESO。

## 设置与分类

对 En-De/Zh/Ja 共 3,266 个 gold-term occurrences，分别比较 RASST 与
InfiniSST 在 `lm=1` 和 `lm=2` 的 exact-form outcome：

- `gain`：RASST correct、InfiniSST wrong；
- `loss`：RASST wrong、InfiniSST correct；
- `both_wrong`：两者均未命中固定 target string；
- `both_correct`：两者均命中。

可复算 surface taxonomy：

| Type | Rule | Examples |
| --- | --- | --- |
| Acronym / symbolic name | 含全大写 token、内部大写或数字 | `KinyaBERT`, `RGF`, `BiLSTM-CRF` |
| Multiword expression | 至少两个 alphanumeric tokens，且不属于上一类 | `morphological analyzer`, `named entity recognition` |
| Single-word term | 其余 glossary entries | `transformer`, `input`, `context` |

## `lm=1` pooled 结果与对应比例

| Type | N | RASST / InfiniSST | Delta | Gain rate | Loss rate | Both-wrong rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Acronym / symbolic | 343 | 84.55 / 48.69 | **+35.86 pp** | **39.36%** (135/343) | 3.50% (12/343) | 11.95% (41/343) |
| Multiword | 192 | 79.69 / 50.52 | **+29.17 pp** | **31.77%** (61/192) | 2.60% (5/192) | **17.71%** (34/192) |
| Single-word | 2,731 | 85.35 / 70.71 | **+14.65 pp** | 18.67% (510/2,731) | **4.03%** (110/2,731) | 10.62% (290/2,731) |

按所有 outcome 的组成看：

- 706 个 gains 中，acronym/symbolic、multiword、single-word 分别占
  **19.12% / 8.64% / 72.24%**；single-word 占比高主要因为它本身占全部
  occurrences 的 83.62%。按类内发生率比较，acronym/symbolic 的 gain propensity
  最高（39.36%）。
- 127 个 losses 中三类分别占 **9.45% / 3.94% / 86.61%**。类内 loss rate
  分别为 3.50% / 2.60% / 4.03%，所以 single-word terms 也略容易发生反向 loss。
- `both_wrong` 最突出的是 multiword expressions（17.71%）。这表明最低延迟下，
  需要组合多个成分并保持固定译法的术语最依赖尚未到达的上下文。

三语趋势一致：

| Type | En-De delta | En-Zh delta | En-Ja delta |
| --- | ---: | ---: | ---: |
| Acronym / symbolic | +34.51 | +34.48 | +38.60 |
| Multiword | +41.38 | +17.39 | +30.77 |
| Single-word | +16.25 | +13.77 | +14.21 |

## `lm=1` 与 `lm=2` 的稳定性

| Type | `lm=1` delta | `lm=2` delta | `lm=1` RASST exact | `lm=2` RASST exact | Both-wrong: `lm=1 → lm=2` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acronym / symbolic | +35.86 | +34.99 | 84.55 | 88.05 | 11.95 → 9.33% |
| Multiword | +29.17 | +29.69 | 79.69 | 84.90 | **17.71 → 10.42%** |
| Single-word | +14.65 | +12.23 | 85.35 | 86.56 | 10.62 → 9.45% |

两档 latency 给出相同排序：acronym/symbolic 的相对收益最大，multiword 次之，
single-word 最小。`lm=1` 的 RASST absolute accuracy 更低，但 baseline 降得更多，
所以相对增益没有消失。最清楚的低延迟困难是 multiword 的 both-wrong rate 增加
7.29 points，而不是 reverse loss 数量增加：两档都是 127/3,266 losses。

## 哪些术语受益，哪些仍困难

`lm=1` 中最稳定的正向转移包括：

- rare acronyms/names：`KinyaBERT` 21 gains / 0 losses、`RGF` 20/1、
  `FLR` 13/0、`REALM` 11/0；
- multiword expressions：`morphological analyzer` 7/0、
  `pretrained language` 7/0、`named entity recognition` 5/0、
  `question generation` 5/0。

困难项呈现三种特征：

1. **Multiword compositionality under minimal context.** `Semantic Parsing`
   有 6 个 both-wrong occurrences，`text classification` 有 3 个；这与该类整体
   17.71% both-wrong rate 一致。
2. **Short/generic/polysemous single words.** `utterance`、`answer`、`speech`、
   `features` 分别有 14/14/12/12 个 both-wrong occurrences；`input` 出现
   6 gains / 8 losses，说明固定 glossary sense 可能与局部上下文竞争。
3. **Acoustically confusable rare names.** `LinCE` 与 `BETO` 各有 6 个 both-wrong，
   `BLEU` 有 3 个。正确 term map 能带来很大平均收益，但在 speech evidence
   不充分或未召回时仍然脆弱。

## 关于最低 latency 的 translation-quality 风险

Canonical main-result BLEU 在 `lm=1` 为：En-De `26.4180 vs 27.1215`
（RASST `-0.7035`）、En-Zh `43.8652 vs 40.6663`（`+3.1989`）、En-Ja
`22.6615 vs 24.4002`（`-1.7387`）。到 `lm=2`，三语 BLEU delta 均为正。
因此“最低 latency 可能带来普通翻译质量 trade-off”在 De/Ja 有 corpus-level
迹象，但不能直接归因给 term map：`lm=1` 的 reverse exact losses 并未增加，主要
变化是 both-wrong 和普通生成困难增加。正式回答 reviewer 的 terminology-type
问题时建议使用前面的 type table；这段 BLEU 观察可留作内部解释，不作为因果结论。

## 可复算产物与 Source of Truth

- 实现：
  [`analyze_term_type_outcomes.py`](../../../code/rasst/analysis/rebuttal/analyze_term_type_outcomes.py)
- `lm=1` summary / term transitions / input manifest：
  [`term_type_outcomes_acl_lm1.tsv`](term_type_outcomes_acl_lm1.tsv)、
  [`term_type_terms_acl_lm1.tsv`](term_type_terms_acl_lm1.tsv)、
  [`term_type_analysis_acl_lm1_manifest.json`](term_type_analysis_acl_lm1_manifest.json)、
  [`term_type_analysis_acl_lm1_source_manifest.json`](term_type_analysis_acl_lm1_source_manifest.json)
- `lm=2` 对应产物：
  [`term_type_outcomes_acl_lm2.tsv`](term_type_outcomes_acl_lm2.tsv)、
  [`term_type_terms_acl_lm2.tsv`](term_type_terms_acl_lm2.tsv)、
  [`term_type_analysis_acl_lm2_manifest.json`](term_type_analysis_acl_lm2_manifest.json)
- Canonical main-result BLEU：
  [`../main_result_global_cache30_30_20_20/compare_vs_infinisst_and_paper.tsv`](../main_result_global_cache30_30_20_20/compare_vs_infinisst_and_paper.tsv)

完整 `lm=1` per-occurrence staging 位于 Taurus
`/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/rebuttal_term_type_lm1_20260712/occurrences`
及本机 ignored staging `outputs/rebuttal_2026/term_failure_acl_lm1`。预定 Hugging Face
dataset 仍为 `gavinlaw/rasst-rebuttal-term-failure-analysis-acl`；上传状态为
**pending**，本地与 Taurus staging 均不是 canonical artifact。
