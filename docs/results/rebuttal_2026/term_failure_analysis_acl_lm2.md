# ACL `lm=2` 术语失败链与 De morphology audit

状态：**自动统计已完成；De/Zh raw false-copy flags 已逐条审计；De 语义标签为
Codex 辅助的非专家 draft，提交前仍需作者签字复核。**

## 固定设置与定义

- 数据：ACL 60/60 的 5 个 talks、468 个对齐句；只看预先固定的中间 operating
  point `lm=2`，比较 paper-exact RASST 与 InfiniSST。
- Gold occurrence 与现有 `term_adoption.json` 完全一致：英文 source 出现 glossary
  source term，且人工 reference 出现固定 target string。
- `retrieved_on_time`：带有该 exact target hint 的第一条 LLM prompt，其 retrieval
  evidence 时间段与 gold 句重叠，且 prompt 在 source sentence end 前发出。
- `retrieved_late`：evidence 仍与该 gold 句重叠，但第一条含 hint 的 prompt 晚于
  source sentence end。
- `never_retrieved`：整段 runtime trace 中没有满足上述 evidence-overlap 条件的
  exact target hint。
- 这个时间定义使用 sentence end 作为宽松 deadline；它不声称知道词级 gold
  emission time。因此结果适合做 failure diagnostic，不是词级 latency metric。
- 当前 canonical run 的 runtime glossary 就是 raw-gold glossary，所以
  `glossary_missing=0` 是实验设计决定的，不是系统能力。Paper-derived realistic
  glossary 跑完后，这一格才会非零。

可复算实现：
[`analyze_term_failure_chain.py`](../../../code/rasst/analysis/rebuttal/analyze_term_failure_chain.py)。

## 1. Retrieval timing 与 exact correctness

| Language | Gold occ. | On time | `P(exact | on time)` | Late | `P(exact | late)` | Never | `P(exact | never)` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| De | 971 | 727 | 86.11% (626/727) | 163 | 82.82% (135/163) | 81 | 59.26% (48/81) |
| Zh | 1,173 | 868 | 92.74% (805/868) | 196 | 89.80% (176/196) | 109 | 71.56% (78/109) |

结论很清楚：late retrieval 有约 3 个百分点的条件正确率损失，但真正的断崖是
never retrieved（De 比 on-time 低 26.85 points；Zh 低 21.18 points）。与此同时，
retrieved on time 也不是充分条件：De 有 101、Zh 有 63 个 on-time exact misses。

在不做语义复核时，所有 exact misses 的第一层分解是：

| Language | Exact misses | Retriever miss | Retrieved late but not exact | On time but not exact |
| --- | ---: | ---: | ---: | ---: |
| De | 162 | 33 | 28 | 101 |
| Zh | 114 | 31 | 20 | 63 |

完整计数见 [`term_failure_chain_acl_lm2.tsv`](term_failure_chain_acl_lm2.tsv)。

## 2. German exact-match audit

我们对全部 162 个 De exact misses 生成候选，而不是只挑成功例：

1. case-folded/compound substring；
2. 去空格、连字符后的 compact match；
3. target 与 hypothesis n-gram 的保守 fuzzy inflection candidate（ratio `>=0.78`）；
4. 对全部 misses 再检查语义、跨句 commitment 和明显 omission/wrong translation。

自动规则给出 51 个候选；逐条 draft audit 后，全部 162 个 misses 的标签为：

| Draft label | Count | 是否加入 morphology-aware diagnostic |
| --- | ---: | --- |
| Valid compound / orthography | 36 | Yes |
| Valid morphology | 14 | Yes |
| Valid paraphrase / glossary synonym | 69 | No |
| Valid alignment-boundary commitment | 13 | No |
| Wrong translation | 19 | No |
| Omitted term | 11 | No |

因此：

- exact-form TERM_ACC：`809/971 = 83.32%`；
- 保守 morphology/compound-aware diagnostic：`859/971 = 88.47%`，即
  **+5.15 points**；
- 更宽的语义 draft：`941/971 = 96.91%`。这一数字高度依赖主观 paraphrase 与
  boundary 判断，**不能作为 rebuttal headline，也不能在作者复核前称为人工或
  专业德语评测**。

在宽语义 draft 上采用互斥的层级归因（先排除 metric false negatives，再看
retrieval，再看 on-time generation），162 个 exact misses 分成：132 metric false
negatives、12 genuine errors with never retrieval、4 genuine errors with late retrieval、
6 on-time-but-unused、8 on-time wrong translations。这个分解只适合内部诊断；正式
回复建议优先报 exact 与保守 morphology-aware 两个数字，并披露 reviewer 资质。

逐项标签及理由见
[`german_morphology_manual_audit.tsv`](german_morphology_manual_audit.tsv)。当前审计者
是 Codex 辅助、非专业德语 annotator；作者需要逐行确认或修改后才能将它称为
`author diagnostic`。

## 3. 为什么 De `lm=2` 的 BLEU 上升而 xCOMET 下降

这个 cell 的 verified 指标是：regular BLEU `+1.3242`、masked BLEU `+0.7094`，
但 paired sentence xCOMET `-0.4907` points。Zh 同一 cell 为 BLEU `+2.9653`、
masked BLEU `+1.1740`、xCOMET `+0.1442`。

按“RASST 相对 InfiniSST 的 exact gold-term 数量”把 468 句互斥分组：

| Language | Group | Sentences | Mean xCOMET delta | Delta sum (raw score) |
| --- | --- | ---: | ---: | ---: |
| De | Net term gain | 135 | +1.86 points | +2.511 |
| De | Exact tie | 308 | -0.64 points | -1.970 |
| De | Net term loss | 25 | -11.35 points | -2.838 |
| Zh | Net term gain | 145 | +2.38 points | +3.448 |
| Zh | Exact tie | 305 | -0.08 points | -0.241 |
| Zh | Net term loss | 18 | -14.07 points | -2.532 |

三个组的 delta sum 在每种语言内恰好加回总差值。De 与 Zh 都表现为“term gain
句正、term loss 句强负”；差异在于 Zh 的 gain 更大且 tie 近乎为零，而 De 的
tie 句额外损失约 `-1.970` raw-score sum。这些 tie 负例大量是输出跨 sentence
boundary、句尾截断或普通生成错误，而不是直接的术语注入。

完整分组见
[`xcomet_failure_groups_acl_lm2.tsv`](xcomet_failure_groups_acl_lm2.tsv)。注意 exact
term gain/loss 本身仍受 morphology 与 paraphrase 影响，所以这是描述性分组，
不是因果分解。

## 4. Term-noise 假设：存在，但不是 raw FCR 显示的规模

最初的 raw `term_map_false_copy` 很像支持 term-noise 假设：De 的 44 个 flagged
句平均 xCOMET `-3.52` points，Zh 的 31 个 flagged 句为 `-5.12`。逐条查看后发现，
这个 exact diagnostic 严重混入两种 false positive：

- source morphology/semantic support，例如 `words→Wort`、
  `finetuned→Feinabstimmung`；
- streaming/mWER boundary，例如上一句的 `System`、`Token`、`FLR` 延迟到当前
  sentence hypothesis。

82 个 raw flagged terms 的审计结果为：

| Language | Raw flags | Source morphology/semantics | Boundary | Harmful unsupported adoption | Benign unsupported adoption |
| --- | ---: | ---: | ---: | ---: | ---: |
| De | 49 | 34 | 12 | 2 | 1 |
| Zh | 33 | 19 | 9 | 2 | 3 |

尤其关键的是，flagged boundary 句的平均 xCOMET 是 De `-20.95`、Zh `-23.70`
points；它们才是 raw flag 与低 xCOMET 的主要混杂来源。De 两个 harmful
unsupported-adoption 句均下降，平均 `-11.35` points；但样本只有 2 句，且其中
`analyzer→analysis` 在 InfiniSST 中也发生，观察日志不能把它单独因果归因给
retrieval。`arXiv→Oracle` 则更像 retrieval-specific error，因为 InfiniSST 没有
生成 Oracle。Zh 两个 harmful-adoption cases 一正一负，平均反而 `+2.74`，说明
xCOMET 也会漏掉明确的实体替换错误。

结论应改成：**错误 retrieval 确实会造成严重个例，但当前 De xCOMET 下降不能
简单归因于 term noise；更强的证据指向 term-loss 句、普通生成变化，以及
sentence-level xCOMET 对 delayed commitment/boundary alignment 的敏感性。**

逐项 audit 见
[`retrieved_false_copy_draft_audit.tsv`](retrieved_false_copy_draft_audit.tsv)，汇总见
[`retrieval_noise_audit_acl_lm2.tsv`](retrieval_noise_audit_acl_lm2.tsv)。

## 5. 可放进正文的真实 cases

1. **Harmful retrieved-hint adoption / acoustic confusion (De, sentence 248).** Source 是
   `arXiv or PubMed`，retriever 给出 `oracle→Oracle`，RASST 生成
   `aus dem Oracle oder PubMed`；xCOMET 将 `dem Oracle` 标为 major error，paired
   delta `-15.86` points。
2. **Multiword morphology false negative (De, sentence 162).** Gold target
   `morphologische Analyse`，on-time hint 已到；RASST 生成语法正确的
   `mittels morphologischer Analyse`，exact metric 判错，但 xCOMET delta
   `+14.52` points。
3. **Acronym/homophone plus retriever miss (De, sentence 332).** `BLEU` 被生成成
   `Blue`；该 occurrence 从未正确 retrieved，xCOMET delta `-6.40` points。
4. **Late acronym retrieval (De, sentence 91).** `LinCE` 的正确 hint 首次出现于
   source sentence end 后约 `0.58 s`，模型已生成 `LINTEX`。这是明确的 late
   failure；但 xCOMET delta 仍为 `+18.69`，显示 contextual metric 不能替代术语
   audit。
5. **Delayed commitment / alignment artifact (De, sentence 303).** `after each
   token` 的 `jedem Token` 被生成到下一 aligned hypothesis；当前句 exact miss，
   但 talk-level 内容没有丢失。
6. **Harmful retrieved-hint adoption (Zh, sentence 22).** Source 是 `Spanish newspapers`，
   retrieved `newswire→新闻专线` 被采用为 `西班牙语新闻专线`；xCOMET 标成 major
   error，delta `-6.82` points。
7. **xCOMET blind spot (Zh, sentence 434).** Source 是随机抽取 `passages and
   answers`，错误 hint 使 RASST 变成 `回答和问题`；尽管实体类别错了，xCOMET
   反而从 `86.18` 升到 `98.48`。

这些例子覆盖 multiword morphology、homophone/acronym、never retrieval、late
retrieval、retrieved-and-misused 和 delayed commitment。不要把第 5 类误写成
translation omission，也不要把 raw false-copy flag 直接叫作 true term noise。

## 6. 建议给 reviewers 的简短表述

给 oktu 的主回答应使用 term-type analysis，而不是 retrieval timing：

> Across 3,266 ACL gold-term occurrences at `lm=2`, RASST improves exact-form
> accuracy over InfiniSST by 34.99 points for acronyms/symbolic names, 29.69
> points for multiword expressions, and 12.23 points for single-word terms.
> Benefits are largest for rare names/acronyms that the baseline corrupts
> phonetically and for multiword expressions that it partially omits or
> paraphrases. Conversely, 109/127 reverse exact losses are single-word terms;
> an author audit identifies 71 as paraphrase, form, or boundary false
> negatives, leaving 56 genuine losses, mainly omissions or wrong translations
> of short/context-dependent terms.

完整 taxonomy、三语一致性与 examples 见
[`term_type_analysis_acl_lm2.md`](term_type_analysis_acl_lm2.md)。

下面的 timing chain 更适合回答 gbii 关于 retrieval failure 的问题：

> At the fixed intermediate operating point (`lm=2`), exact correctness is
> 86.1% when the correct hint arrives within the source sentence, 82.8% when it
> first arrives after the sentence boundary, and 59.3% when it is never
> retrieved for En-De (92.7/89.8/71.6% for En-Zh). For German, exact-form
> TERM_ACC is 83.32%; a conservative audit that additionally accepts validated
> inflection, compounding, case, and tokenization variants raises it to 88.47%
> (+5.15 points). We continue to report exact-form accuracy as the primary
> reproducible metric and present the morphology-aware number only as an
> author diagnostic.

> The En-De `lm=2` cell improves BLEU by 1.32 and target-term-masked BLEU by
> 0.71, but decreases xCOMET by 0.49 points. Manual inspection does identify
> harmful unsupported-hint adoptions (e.g., `arXiv→Oracle`), yet it also shows that a raw
> exact-match false-copy statistic substantially confounds German morphology
> and streaming sentence-boundary commitment. We therefore do not attribute
> the full contextual-metric decrease to terminology noise; term-loss and
> boundary/ordinary generation errors are also important.

在作者逐行确认德语 audit 前，第一段的 `validated` / `author diagnostic` 仍是提交
保护项。

## Provenance 与 artifact 状态

- xCOMET segments SHA-256：
  `43b5581bc79e8c389383e6fb84b684f4f7207334c114a0cc0b8b19d47d2a459b`。
- De runtime / term-adoption SHA-256：
  `3d78b8adc2783af8f97a1a608eeab5989639b3426bef9ae7171b4b402921a71f` /
  `15b58e98ddd2e99f373483b1e00b18c992b41c1538f68dfe6fdb4e911f0b9f1b`。
- Zh runtime / term-adoption SHA-256：
  `cf3bb08c58f9949e08dd24ff3a1b95ed1570c8743ea1b1bf4bcea54b347d4d7f` /
  `bce65b241c1d1cdc78825e70a439727a1090e630b358e73659afb334e7f90f3d`。
- Audio YAML SHA-256：
  `60c3c999cd6c4fcd80565269367f92fa65e3171a8598d0859fb87ac22be9dfaa`。
- 完整 per-occurrence/per-sentence 输出已从临时目录复制到本机 persistent ignored
  staging
  `/Users/luojiaxuan/Documents/RASST/outputs/rebuttal_2026/term_failure_acl_lm2/{de,zh}`；
  预定 Hugging Face dataset 为
  `gavinlaw/rasst-rebuttal-term-failure-analysis-acl`。上传前该 staging 不是
  canonical artifact。当前本机没有 Hugging Face CLI/写入凭据，上传状态为
  **pending**。
