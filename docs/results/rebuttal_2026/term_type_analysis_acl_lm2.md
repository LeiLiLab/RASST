# ACL `lm=2` terminology-type outcome analysis

> `lm=1/2` 的当前合并分析见
> [`term_type_analysis_acl_lm1_lm2.md`](term_type_analysis_acl_lm1_lm2.md)。本文件保留
> `lm=2` 的审计细节与历史 provenance。

状态：**自动统计完成；term taxonomy 是可复算的 English surface-form taxonomy；
loss 的语义标签沿用 De/Zh/Ja exact-miss author-diagnostic draft。**

## 问题与范围

本分析直接回答 reviewer 的问题：哪些类型的 terminology 从 RASST 受益，哪些类型
仍然困难。固定设置为 ACL 60/60 五个 talks、En-De/Zh/Ja、`lm=2`，共 3,266 个
gold-term occurrences。每个 occurrence 比较 RASST 与 InfiniSST 的 exact-form
outcome：

- `gain`：RASST exact-correct，InfiniSST exact-wrong；
- `loss`：RASST exact-wrong，InfiniSST exact-correct；
- `both_correct` / `both_wrong`：两者相同。

为避免再引入主观的逐 term 标签，taxonomy 只使用 English glossary surface form：

| Type | Reproducible rule | Typical examples |
| --- | --- | --- |
| Acronym or symbolic name | 含全大写 token、内部大写或数字 | `KinyaBERT`, `RGF`, `BiLSTM-CRF`, `BETO`, `FLR` |
| Multiword expression | 至少两个 alphanumeric tokens，且不属于上一类 | `morphological analyzer`, `question generation`, `named entity recognition`, `masked language model` |
| Single-word term | 其余条目 | `transformer`, `token`, `question`, `graph` |

## 主要结果

三语 pooled 结果：

| Type | Occurrences | RASST exact | InfiniSST exact | Delta |
| --- | ---: | ---: | ---: | ---: |
| Acronym / symbolic name | 343 | 88.05% (302/343) | 53.06% (182/343) | **+34.99 pp** |
| Multiword expression | 192 | 84.90% (163/192) | 55.21% (106/192) | **+29.69 pp** |
| Single-word term | 2,731 | 86.56% (2,364/2,731) | 74.33% (2,030/2,731) | **+12.23 pp** |

这不是由单一语言驱动。De/Zh/Ja 的 delta 分别为：

| Type | En-De | En-Zh | En-Ja |
| --- | ---: | ---: | ---: |
| Acronym / symbolic name | +24.78 | +39.66 | +40.35 |
| Multiword expression | +32.76 | +21.74 | +35.38 |
| Single-word term | +12.50 | +11.03 | +13.26 |

因此最明确的受益特征是：

1. **Rare acronyms and symbolic names.** InfiniSST 容易发生 phonetic corruption、
   大小写/字符漂移或泛化；retrieved glossary entry 能稳定 exact form。净 gain 最大
   的 term 包括 `KinyaBERT`（+25 occurrences）、`RGF`（+21）、
   `BiLSTM-CRF`（+9）、`BETO`（+7）和 `FLR`（+6）。
2. **Multiword technical expressions.** Baseline 容易只翻译部分成分或改写成 generic
   phrase；RASST 能提供完整 target phrase。典型净 gain 包括
   `pretrained language`（+9）、`question generation`（+9）、
   `morphological analyzer`（+8）、`named entity recognition`（+5）和
   `masked language model`（+2）。
3. **Single-word technical terms also benefit, but less selectively.** 这类 term
   占数据绝大多数，RASST 仍提升 +12.23 pp；但它们更容易与普通词义、自然
   paraphrase 和局部 omission 混在一起。

## 反向 losses 与 difficult terms

Raw exact comparison 中共有 127 个 `loss` occurrences，其中 109 个（85.8%）是
single-word terms。逐条 exact-miss audit 进一步显示：

| Raw exact losses | Count |
| --- | ---: |
| Valid paraphrase / morphology / orthography / boundary | 71 |
| Genuine omissions | 34 |
| Genuine wrong translations | 22 |

因此超过一半的 apparent losses 是 exact metric 或 streaming alignment false
negatives，而不是 RASST 真正降低翻译正确率。剩余 56 个 genuine losses 中，45 个
（80.4%）仍是 single-word terms。

真正困难的 term 呈现三类特征：

- **Short, generic, or polysemous glossary entries**，例如 `question`, `graph`,
  `input`, `system`, `context`。它们容易被自然 paraphrase、被省略，或被相邻
  glossary entry 拉向不适合当前上下文的固定译法。
- **Acoustically confusable rare names/acronyms**。当 speech evidence 或 retrieval
  失败时，仍会出现 `ICLR→ACL`、`LinCE→LINTEX`、`BLEU→Blue`、
  `BETO→BERT`。
- **Overlapping or context-mismatched entries**。例如 En-Ja
  `text classification` 被 `context→文脈` 干扰成 `文脈分類`；这类错误不是
  “术语越多越好”，而是要求更好的 contextual disambiguation。

结论不是“所有 acronym 都容易”或“所有 single-word 都困难”：acronyms 和
multiword expressions 的平均收益最大，但其中声学相似的 rare names 在没有正确
evidence 时仍然脆弱；single-word terms 总体也受益，但构成了大多数反向 loss 和
metric ambiguity。

## 可复算产物

- 实现：
  [`analyze_term_type_outcomes.py`](../../../code/rasst/analysis/rebuttal/analyze_term_type_outcomes.py)
- 三语/pooled summary：
  [`term_type_outcomes_acl_lm2.tsv`](term_type_outcomes_acl_lm2.tsv)
- loss audit：
  [`term_type_loss_audit_acl_lm2.tsv`](term_type_loss_audit_acl_lm2.tsv)
- term-level transitions：
  [`term_type_terms_acl_lm2.tsv`](term_type_terms_acl_lm2.tsv)
- 输入 hashes 与 taxonomy：
  [`term_type_analysis_acl_lm2_manifest.json`](term_type_analysis_acl_lm2_manifest.json)

完整 per-occurrence inputs 仍位于 ignored staging
`outputs/rebuttal_2026/term_failure_acl_lm2/{de,zh,ja}/occurrences.tsv`，预定
Hugging Face dataset 为 `gavinlaw/rasst-rebuttal-term-failure-analysis-acl`；
上传状态仍为 **pending**。
