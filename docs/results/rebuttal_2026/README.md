# RASST rebuttal 2026：实验与 Source of Truth

本目录是 rebuttal 补实验的 Git-tracked 索引。代码、轻量结果表、实验决策和
进度以本仓库 `main` 分支为准；公开评测数据和模型
仍以 README 中列出的 Hugging Face 仓库为准。大模型权重、逐句评分和生成式
glossary 原始响应不进入 Git。

## 当前结论

- **xCOMET-XXL 已完成，并修正为 new InfiniSST × paper-exact RASST。** ACL
  12 cells 的 cell-macro 差值为 `+0.1158` xCOMET points（乘以 100），8/12
  为正；ESO En-De 为 `-1.3848`，0/4 为正。16-cell macro 为 `-0.2593`，
  8/16 为正。旧 `-3.1716` 同时使用旧 InfiniSST 与 release-cache RASST；中间
  `-2.1012` 使用旧 InfiniSST 与 paper-exact RASST，均不再是推荐口径。当前 lm4
  差值仅 `-0.2287`，主要负差来自 lm1 `-2.5040` 与 lm3 `-1.8294`。
- **Masked BLEU 已完成并全量复算。** 排除 ESO En-Zh/En-Ja 后，RASST 相对
  InfiniSST 的 target-term-masked BLEU 在 12/16 cells 为正，平均差值
  `+0.6927`；ACL 12 cells 的平均差值为 `+0.9919`，ESO En-De 4 cells 的
  平均差值为 `-0.2047`。详细协议、逐行快照和验证哈希见
  [`masked_bleu_global_cache_snapshot.md`](masked_bleu_global_cache_snapshot.md)。
- **术语占比已完成。** Raw-gold aligned target terms 占 ACL reference tokens 的
  `10.66%–18.06%`，占 ESO En-De reference tokens 的 `2.20%`。ACL 中包含 aligned
  target term 的句子比例为 `83.76%–86.54%`，所以不能笼统表述为“术语只出现在
  很少句子”；更准确的说法是它们广泛分布、但只占少数 token。逐语言计数和
  输入哈希见 [`term_prevalence.tsv`](term_prevalence.tsv)。
- **ESO 结果口径已确定。** 原始 ESO German reference 是人工翻译；本项目新增的
  Chinese/Japanese reference 是 GPT-5.4 生成。Revision 保留 ESO En-De，移除依赖
  synthetic sentence reference 的 ESO En-Zh/En-Ja BLEU、xCOMET 等
  translation-quality claims。若保留 term-only readout，必须与 reference-based
  metrics 分表，并标为 GPT-5.4-assisted、manually checked glossary，不能称为
  fully human-authored 或 domain-expert annotation。Hard glossary 实际为 215
  rows、212 unique terms，不是论文中的 217。
- **ESO hard-term pipeline provenance 已补齐。** Stage 1/2/5 使用 GPT-5.4：
  Stage 1 以 10 sentences/batch 生成 Zh/Ja sentence translations 与 candidate
  terms，Stage 2 统一 exact-span/abbreviation decisions，Stage 5 以
  3 sentences/batch 对照 no-RAG baseline 确定 hard terms；Stage 3/4 做非 LLM
  consistency/source exact-match 处理，Stage 6 manual check glossary，Stage 7
  生成 final output。三个 exact prompts、hash 与允许的论文口径见
  [`eso_hard_term_pipeline/`](eso_hard_term_pipeline/)。
- **ACL terminology qualification 已由官方论文核实。** 60/60 terminology lists
  non-exhaustive；source spans 自动标记；英文技术词经 domain expert 复核；target
  translation 由目标语言母语 professional post-editor 处理并经第二位 annotator
  复核。不能据此声称 target translations 也由 domain experts 完成。来源为
  [Salesky et al. (2023), Secs. 3.4--3.7 and App. A.5](https://aclanthology.org/2023.iwslt-1.2/)。
- **ACL `lm=2` failure analysis 已完成。** De/Zh 的
  `P(exact | retrieved on time/late/never)` 分别为
  `86.11/82.82/59.26%` 与 `92.74/89.80/71.56%`。De exact-form TERM_ACC 为
  `83.32%`；接受逐条核验的屈折、复合、大小写和 tokenization variants 后，保守
  morphology-aware diagnostic 为 `88.47%`（`+5.15 points`）。82 个 raw
  false-copy flags 的审计表明，其中多数是 source morphology/semantics 或 streaming
  boundary，不应直接称为 term noise。完整结论、真实 cases 与提交限制见
  [`term_failure_analysis_acl_lm2.md`](term_failure_analysis_acl_lm2.md)。德语语义
  标签当前仍是 Codex 辅助的非专家 draft，作者 sign-off 前不能称为人工专业评测。
- **ACL `lm=1/2` term-type outcome analysis 已完成。** 对三语 3,266 个 gold
  occurrences 使用可复算 English surface taxonomy。最短延迟 `lm=1` 下，
  acronym/symbolic、multiword、single-word 的 RASST/InfiniSST exact delta 分别为
  `+35.86/+29.17/+14.65 pp`；`lm=2` 为 `+34.99/+29.69/+12.23 pp`，排序稳定。
  `lm=1` 的 gain/loss/both-wrong 类内比例分别为 acronym
  `39.36/3.50/11.95%`、multiword `31.77/2.60/17.71%`、single-word
  `18.67/4.03/10.62%`。两档均有 127 个 reverse exact losses；最低 latency
  主要增加的是 both-wrong，尤其 multiword，而不是 term-map reverse loss。
  完整定义、三语结果、examples、BLEU trade-off 和限制见
  [`term_type_analysis_acl_lm1_lm2.md`](term_type_analysis_acl_lm1_lm2.md)。
  对四个 En-Ja 强负 xCOMET cases 的 MFA-aligned term-map 复核进一步发现：只有
  ACL 367:16 可直接归因于 `sentence/document→文章` 的 target collision；另外三例
  主要是 streaming commitment / mWER resegmentation。逐 chunk 证据见
  [`ja_xcomet_mfa_term_map_cases.md`](ja_xcomet_mfa_term_map_cases.md)。5-sentence
  block 回归把后三例 delta 从约 `-0.80` 修正为 `+0.034/-0.002/+0.151`，见
  [`xcomet_acl_block5_case_audit.tsv`](xcomet_acl_block5_case_audit.tsv)。
- **Retrieval degradation sensitivity 已完成。** 在 ACL 三语 `lm=2` 固定
  hint count 与 compute，将 sentence-relevant correct hints 按 `25% / 50%`
  概率替换为同域 distractors。En-Zh/De TERM_ACC 相对各自 `0%` 分别下降
  `1.91/4.27` 与 `3.31/7.59` points，但 BLEU 反而分别上升
  `0.259/0.453` 与 `0.208/0.715`；六个 degraded cells 的 xCOMET 均低于同语言
  `0%`。用于 rebuttal 的合并展示中，En-Ja TERM_ACC 为
  `84.57 → 82.77 → 78.40%`，xCOMET 为 `70.379 → 68.417 → 67.137`；三语六个
  degraded cells 的 TERM_ACC 与 xCOMET 均低于各自 `0%`。完整展示表、底层来源和
  限制见
  [`retrieval_degradation_ablation.md`](retrieval_degradation_ablation.md)。
- **Legacy paper-derived glossary v1 的 `lm=2` 对比已完成并修正评测口径。** RASST
  推理使用五篇论文各自的 Gemini-extracted glossary 作为 index；RASST 与 InfiniSST
  的 TERM_ACC 均使用未改变的 tagged raw ACL glossary 作为评测 denominator。因此
  En-Zh/Ja/De 的分母保持 `890/940/935`。作者确认的 reported TERM_ACC delta 为
  `+2.70/-0.64/+2.70` points（macro `+1.59`），BLEU delta 为
  `+0.5012/+0.0455/-1.0657`。De reported TERM_ACC 与 pooled counts 属于不同聚合
  口径，已在表中分列。此前使用 paper-derived glossary 同时作为 denominator 的
  matched-glossary 数字口径错误，已从 tracked artifacts 删除。完整协议、三语结果
  与 author-confirmed snapshot 见
  [`../acl_paper_extracted_lm2/README.md`](../acl_paper_extracted_lm2/README.md)。

## xCOMET

状态：**已完成并独立复算验证**。

- 推荐评分矩阵：ACL En-Zh/De/Ja 与 ESO En-De，RASST/InfiniSST，4 个 latency
  settings，共 32 system rows / 16 cells / 22,728 sentence segments。ACL 使用已
  验证的 release run；ESO En-De 交叉组合 independently validated 的 new InfiniSST
  与 submitted-paper exact 四档 cache `30/30` RASST system means。
- Metric：`Unbabel/XCOMET-XXL`，revision
  `873bac1b1c461e410c4a6e379f6790d3d1c7c214`。
- Encoder tokenizer/config：`facebook/xlm-roberta-xxl`，revision
  `03e0fb540c3c9afd4bdda0072e7cb82d2eafd060`。
- 原 32-system release-cache 输入清单：
  [`xcomet_input_manifest.taurus.tsv`](xcomet_input_manifest.taurus.tsv)。paper-exact
  ESO 四行的原始路径与 hash 见
  [`xcomet_paper_exact_eso_de_input_provenance.tsv`](xcomet_paper_exact_eso_de_input_provenance.tsv)，
  corrected portable manifest 见
  [`xcomet_paper_exact_eso_de_manifest.portable.tsv`](xcomet_paper_exact_eso_de_manifest.portable.tsv)。
- ACL 12 cells 的 cell-macro xCOMET（乘以 100）为 RASST `78.7042`、
  InfiniSST `78.5884`，平均差值 `+0.1158`，8/12 cells 为正；其中 En-Zh
  `+0.6356`、En-De `+0.0050`、En-Ja `-0.2933`。
- New InfiniSST × paper-exact RASST 的 ESO En-De 4 cells 为 RASST `75.9398`、
  InfiniSST `77.3245`，平均差值 `-1.3848`，0/4 cells 为正。16 cells 合计为
  RASST `78.0131`、InfiniSST `78.2724`、差值 `-0.2593`，8/16 为正。
- 结果是 mixed；不能用 xCOMET 声称 RASST 的 overall translation quality 普遍或
  显著提升。paper-exact ESO 复算、cache 差异、运行环境和全部哈希见
  [`xcomet_paper_exact_eso_de_report.md`](xcomet_paper_exact_eso_de_report.md)。
  原 [`xcomet_xxl_report.md`](xcomet_xxl_report.md) 保留为 release-cache 诊断，
  其中 ESO `-3.1716` 不再作为 submitted-paper exact 结果；旧 baseline 与
  paper-exact RASST 的 `-2.1012` 也仅保留为中间修正记录。
- Git-tracked paper-exact ESO 轻量产物：
  [`xcomet_new_infinisst_vs_paper_exact_rasst.tsv`](xcomet_new_infinisst_vs_paper_exact_rasst.tsv)、
  [`xcomet_new_infinisst_lm123_summary.tsv`](xcomet_new_infinisst_lm123_summary.tsv)、
  [`xcomet_new_infinisst_lm123_validation.json`](xcomet_new_infinisst_lm123_validation.json)、
  [`xcomet_new_infinisst_lm4_summary.tsv`](xcomet_new_infinisst_lm4_summary.tsv)、
  [`xcomet_new_infinisst_lm4_validation.json`](xcomet_new_infinisst_lm4_validation.json)、
  [`xcomet_paper_exact_combined_summary.tsv`](xcomet_paper_exact_combined_summary.tsv)、
  [`xcomet_paper_exact_combined_paired.tsv`](xcomet_paper_exact_combined_paired.tsv)、
  [`xcomet_paper_exact_eso_de_summary.tsv`](xcomet_paper_exact_eso_de_summary.tsv)、
  [`xcomet_paper_exact_eso_de_paired.tsv`](xcomet_paper_exact_eso_de_paired.tsv) 和
  [`xcomet_paper_exact_eso_de_validation.json`](xcomet_paper_exact_eso_de_validation.json)。
  Paper-exact RASST 来源运行的 validator 核对了 8 systems / 4 pairs / 11,496
  segments；new InfiniSST 的 lm1--3 与 lm4 来源运行分别核对 6 systems / 3 pairs /
  8,622 segments 和 2 systems / 1 pair / 2,874 segments。交叉表只组合已验证
  system means，不声称一次新的 combined sentence-level win/tie/loss 检验。

逐句 JSONL 与输入 bundle 位于 Hyper00 staging
`/data02/jaxan/RASST_rebuttal_20260710`；paper-exact ESO 逐句文件为
`results/xcomet_paper_exact_eso_de_20260712/segments.jsonl`。它们的预定
New InfiniSST 逐句文件位于
`results/xcomet_eso_de_infinisst_rerun_lm123_20260712/segments.jsonl` 与
`results/xcomet_eso_de_infinisst_rerun_lm4_20260712/segments.jsonl`。它们的预定
Hugging Face 目标为
`gavinlaw/rasst-main-result-data` 下的 versioned rebuttal artifact。目前没有可用的
Hugging Face 写入凭据，上传状态为 **pending**，本地 staging 不能视为 canonical。

## Paper-derived realistic glossary

状态：**legacy v1 realistic-index + tagged-raw-evaluation default `lm=2` 已完成；
fresh Gemini 2.5 extraction 仍 blocked，推理未启动**。

Legacy v1 的 RASST 三语运行和 InfiniSST 三语 post-eval 已全部完成。InfiniSST
没有重新推理；两套系统统一使用 tagged raw glossary 计算 TERM_ACC，只有 RASST
推理时使用 paper-derived index。作者确认的三语 headline 结果为：

| Language | Reported TERM_ACC RASST / InfiniSST (delta) | Pooled correct counts | BLEU RASST / InfiniSST (delta) |
| --- | ---: | ---: | ---: |
| En-Zh | 77.87 / 75.17 (**+2.70 pp**) | 693/890 vs. 669/890 | 46.3280 / 45.8268 (**+0.5012**) |
| En-Ja | 65.32 / 65.96 (**-0.64 pp**) | 614/940 vs. 620/940 | 27.7656 / 27.7202 (**+0.0455**) |
| En-De | 70.91 / 68.21 (**+2.70 pp**) | 652/935 vs. 632/935 | 29.2086 / 30.2743 (**-1.0657**) |

De reported TERM_ACC 与 pooled ratio（69.73% vs. 67.59%）不是同一聚合口径，
因此必须分列。统一对比与 author-confirmed snapshot 见
[`../acl_paper_extracted_lm2/comparison.tsv`](../acl_paper_extracted_lm2/comparison.tsv)。

### Paper-derived + NLP/AI/CS 10k condition

状态：**glossary、coverage 与 15/15 retrieval indices 已验证；三语 `lm=2`
end-to-end inference 等待 Aries GPU 资源，尚无可引用指标。**

- 每个 talk 的 runtime index 是“该 paper 的 extracted terms + 完整 NLP/AI/CS
  10k glossary”。normalized source 重合时保留 paper-specific translation，最终每篇
  glossary 为 `10,036–10,058` 项。ACL raw-gold glossary 不参与 index 构建，仍是
  TERM_ACC 的固定 denominator。
- 相对 238 个 raw-gold unique entries，paper-only source overlap 为
  `57/238 (23.95%)`；加入 10k 后为 `68/238 (28.57%)`。联合 index 的 exact
  source+target pair overlap 为 En-Zh `53/238 (22.27%)`、En-Ja
  `47/238 (19.75%)`、En-De `43/238 (18.07%)`。可复算表见
  [`paper_plus_nlp_ai_cs_10k_coverage.tsv`](paper_plus_nlp_ai_cs_10k_coverage.tsv)，
  paper-only 对照见
  [`realistic_glossary_coverage.tsv`](realistic_glossary_coverage.tsv)。
- 运行代码已在 `main` commit `0d14e4c`。Aries staging root 为
  `/mnt/aries/data6/jiaxuanluo/RASST_release_runs/rebuttal_acl_paper_plus_nlp_ai_cs_10k_lm2_20260713T031147Z`；
  Slurm job 为 `47211`。15 个 index 与 15 个 sidecar manifest 已在 Taurus 构建，
  复制到 Aries 后逐文件 SHA-256 一致。
- 在 `outputs/aggregate/{zh,ja,de}/lm2/eval_results.tsv` 全部生成并通过 manifest
  验证前，不得把 legacy paper-only 数字写成这个联合 index condition 的结果。

- Legacy v1 保留了 glossary 文件与 SHA-256，但原始 extraction 旁没有记录 exact
  Gemini model identifier。因此该结果不能冒充下面计划中的 fresh
  `gemini-2.5-flash` reproduction；raw-gold denominator 已满足，但 exact extractor
  provenance 的保护标记仍不能删除。

- 新 extractor 固定 `gemini-2.5-flash`，只读取 5 篇对应 ACL paper PDF，不读取
  transcript、reference、gold tags 或 gold evaluation glossary；保存 prompt、PDF、
  raw response 和 glossary hashes。
- Taurus legacy script 中的 Gemini API key 已被 Google 标记为 leaked 并返回 403。
  没有使用其他用户的 key，也没有把 key 复制到新文件或日志。
- 新 pipeline 为每篇 paper / 每种 target language 生成独立 runtime glossary，
  inference 只使用该 glossary；最终 exact-form TERM_ACC 始终使用现有 raw-gold
  glossary（SHA-256
  `f9f171c6475c4bb19250f5f93063a5ef034cbdcc1f8a995c593647718cf9a5b6`）作为固定
  denominator。评分器通过显式 `--mwer-segmenter` 参数运行，不依赖环境变量。
- 旧表中已有 12 个 `acl_paper_extracted` RASST cells，但它们被标记为
  `user_supplied_reusable`，没有可追溯的 exact `instances.log`，因此不作为这次
  fresh rebuttal evidence。

Fresh glossary、raw responses 和 manifest 的预定 Hugging Face 目标为
`gavinlaw/rasst-main-result-data` 下的 versioned rebuttal artifact；在新 key 可用并
完成生成前，上传状态为 **blocked**。

## Retrieval degradation sensitivity

状态：**已完成并独立验证 xCOMET**。

- 固定 ACL 三语 `lm=2`，将 sentence-window-relevant correct hints 以
  `0% / 25% / 50%` 的概率替换为同域 distractors。
- 检索执行、top-k、prompt hint count、rank/score metadata 和 Speech LLM 配置不变；
  runtime log 保存 before/after references 和逐次 audit。
- 9 个 cells 共保留 5 talks / 1795 retrieval events per language；En-Zh/De
  各档均为 2637 hints，En-Ja 均为 2622 hints。`0%` references 与 paper-canonical
  runtime 在三语全部事件上逐项一致。
- Rebuttal presentation table 使用 En-Zh/De 的 primary controlled runs 与 En-Ja
  fresh-mask run；六个 degraded cells 相对同语言 `0%` 的 xCOMET delta 为 En-Zh
  `-0.599/-1.020`、En-De `-0.752/-0.471`、En-Ja `-1.962/-3.242` points。
  En-Zh/De primary bundle 与 En-Ja fresh-mask bundle 的独立 validators 均为 `ok`。
  可直接引用的合并表见
  [`retrieval_degradation_rebuttal_table.tsv`](retrieval_degradation_rebuttal_table.tsv)。
- 原始预注册九格表和 En-Ja seed sensitivity 均继续保留用于 provenance；逐行结果见
  [`retrieval_degradation_acl_lm2.tsv`](retrieval_degradation_acl_lm2.tsv)、
  [`retrieval_degradation_ja_seed_sensitivity.tsv`](retrieval_degradation_ja_seed_sensitivity.tsv)
  和 [`retrieval_degradation_ja_rerun_xcomet_summary.tsv`](retrieval_degradation_ja_rerun_xcomet_summary.tsv)。
- 预注册定义、完整结果、compute placement、验证哈希和局限见
  [`retrieval_degradation_ablation.md`](retrieval_degradation_ablation.md)。

完整 runtime logs、instances 和 4212-row xCOMET JSONL 位于 Aries/Hyper00 staging，
预定 Hugging Face 目标为
`gavinlaw/rasst-rebuttal-retrieval-degradation-acl`。上传状态为 **blocked**：
Taurus/Aries 没有 token，Hyper00 现有凭据属于另一账号，未越权使用；在作者提供
授权凭据并上传前，staging 仍不是 canonical artifact。

## Failure analysis 与 German morphology audit

状态：**自动计数完成；draft audit 完成；作者复核 pending**。

- 固定 ACL `lm=2`，从 runtime retrieval timestamps、sentence-aligned
  `term_adoption.json` 和 paper-exact xCOMET segments 复算 971 个 De 与 1,173 个
  Zh gold occurrences。
- 自动化输出包含 on-time/late/never conditional accuracy、exact-miss failure chain、
  paired sentence xCOMET term-gain/tie/loss 分组，以及每个 occurrence 的 provenance。
- De 162 个 exact misses 和 De/Zh 共 82 个 raw false-copy flags 已逐条给出 draft
  label 与理由。保守 morphology-aware 数字只加入 36 个 compound/orthography 和
  14 个 morphology variants；69 个 paraphrases 与 13 个 boundary cases 不进入该
  数字。
- `term_map_false_copy` 是 candidate diagnostic，不是 true-noise ground truth。审计后
  只确认 De/Zh 各 2 个 harmful unsupported-hint adoptions；这仍是观察性标签，
  不是 retrieval 的因果效应。Boundary flags 的 xCOMET 反而极低，说明 sentence
  alignment 是重要混杂。
- Git-tracked 轻量证据：
  [`term_failure_chain_acl_lm2.tsv`](term_failure_chain_acl_lm2.tsv)、
  [`xcomet_failure_groups_acl_lm2.tsv`](xcomet_failure_groups_acl_lm2.tsv)、
  [`retrieval_noise_audit_acl_lm2.tsv`](retrieval_noise_audit_acl_lm2.tsv)、
  [`german_morphology_manual_audit.tsv`](german_morphology_manual_audit.tsv) 和
  [`retrieved_false_copy_draft_audit.tsv`](retrieved_false_copy_draft_audit.tsv)。

完整 per-occurrence/per-sentence 输出已复制到本机 persistent ignored staging
`/Users/luojiaxuan/Documents/RASST/outputs/rebuttal_2026/term_failure_acl_lm2/{de,zh}`，
预定 Hugging Face 目标为
`gavinlaw/rasst-rebuttal-term-failure-analysis-acl`，上传状态为 **pending**。
当前本机没有 Hugging Face CLI/写入凭据；在上传完成前，该本机 staging 仍不是
canonical artifact。

## Rebuttal 文本

按 Reviewer Mzub、oktu、gbii 顺序整理的精简 OpenReview 单评论稿位于
[`../../rebuttal_2026_openreview_responses.md`](../../rebuttal_2026_openreview_responses.md)。
三段正文分别为 `3713 / 4684 / 4751` characters，均低于 5000-character 上限；该稿
只使用已验证的 main/rebuttal-experiments 结果，并明确排除 LLM-as-a-judge 与宽语义
audit。文末的内部取舍和 evidence SoT 不应提交到 OpenReview。

较长的英文历史工作稿位于 [`../../rebuttal_2026_draft.md`](../../rebuttal_2026_draft.md)，
不应直接提交；以 concise response 文件为准。其中
所有 `PENDING` 都是提交保护标记：只有生成、复核并写入本索引的数字才可替换。
Masked BLEU 是 diagnostic，不是因果分解；TERM_ACC 必须称为 exact-form metric。
German morphology-aware 结果只能称为 non-expert author diagnostic，并且必须在
作者逐行复核 draft audit 后才能提交；更宽的 paraphrase-aware `96.91%` 仅供内部
定位 metric false negatives，不应作为 rebuttal headline。
