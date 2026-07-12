# RASST rebuttal 2026：实验与 Source of Truth

本目录是 rebuttal 补实验的 Git-tracked 索引。代码、轻量结果表、实验决策和
进度以本仓库 `luojiaxuan/rebuttal-experiments` 分支为准；公开评测数据和模型
仍以 README 中列出的 Hugging Face 仓库为准。大模型权重、逐句评分和生成式
glossary 原始响应不进入 Git。

## 当前结论

- **xCOMET-XXL 已完成并修正为 paper-exact ESO 输出。** ACL 12 cells 的
  cell-macro 差值为 `+0.1158` xCOMET points（乘以 100），8/12 为正；
  submitted-paper exact ESO En-De 为 `-2.1012`，0/4 为正。16-cell macro 为
  `-0.4385`，8/16 为正。此前 `-3.1716` 的 ESO 数字来自 release-canonical
  `30/30, 30/30, 20/20, 20/20` cache 输出，不是 paper-exact 四档 `30/30` 输出。
  该结果仍不支持“overall translation quality 普遍提升”，但 lm4 的旧大幅负差
  `-4.3289` 修正为 `-0.2571`。
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
  synthetic reference 的 ESO En-Zh/En-Ja BLEU/xCOMET、exact-form TERM_ACC 和
  reference-aligned latency。Hard glossary 实际为 215 rows、212 unique terms，
  不是论文中的 217。
- **ACL `lm=2` failure analysis 已完成。** 本分析只使用 5 个 ACL talks，明确排除
  Medicine/ESO。De/Zh/Ja 的
  `P(exact | retrieved on time/late/never)` 分别为
  `86.11/82.82/59.26%`、`92.74/89.80/71.56%` 与
  `89.21/83.59/60.78%`。De exact-form TERM_ACC 为
  `83.32%`；接受逐条核验的屈折、复合、大小写和 tokenization variants 后，保守
  morphology-aware diagnostic 为 `88.47%`（`+5.15 points`）。135 个 raw
  false-copy flags 的审计表明，其中多数是 source morphology/semantics 或
  streaming boundary，不应直接称为 term noise。完整结论、真实 cases 与提交限制见
  [`term_failure_analysis_acl_lm2.md`](term_failure_analysis_acl_lm2.md)。三语语义
  标签当前仍是 Codex 辅助的非专家 draft，作者 sign-off 前不能称为人工专业评测。
- **Retrieval degradation sensitivity 已完成。** 在 ACL 三语 `lm=2` 固定
  hint count 与 compute，将 sentence-relevant correct hints 按 `25% / 50%`
  概率替换为同域 distractors。En-Zh/De TERM_ACC 相对各自 `0%` 分别下降
  `1.91/4.27` 与 `3.31/7.59` points，但 BLEU 反而分别上升
  `0.259/0.453` 与 `0.208/0.715`；六个 degraded cells 的 xCOMET 均低于同语言
  `0%`。En-Ja 出现明显 autoregressive path sensitivity，不能把单次 run 解读为
  单调 dose response。完整表和限制见
  [`retrieval_degradation_ablation.md`](retrieval_degradation_ablation.md)。

## xCOMET

状态：**已完成并独立复算验证**。

- 评分矩阵：ACL En-Zh/De/Ja 与 ESO En-De，RASST/InfiniSST，4 个 latency settings，
  共 32 system rows / 16 strict pairs / 22,728 sentence segments。ACL 使用已验证的
  release run；ESO En-De 使用 submitted-paper exact 四档 cache `30/30` 输出。
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
- Paper-exact ESO En-De 4 cells 为 RASST `75.9398`、InfiniSST `78.0410`，
  平均差值 `-2.1012`，0/4 cells 为正。16 cells 合计为 RASST `78.0131`、
  InfiniSST `78.4515`、差值 `-0.4385`，8/16 为正。
- 结果是 mixed；不能用 xCOMET 声称 RASST 的 overall translation quality 普遍或
  显著提升。paper-exact ESO 复算、cache 差异、运行环境和全部哈希见
  [`xcomet_paper_exact_eso_de_report.md`](xcomet_paper_exact_eso_de_report.md)。
  原 [`xcomet_xxl_report.md`](xcomet_xxl_report.md) 保留为 release-cache 诊断，
  其中 ESO `-3.1716` 不再作为 submitted-paper exact 结果。
- Git-tracked paper-exact ESO 轻量产物：
  [`xcomet_paper_exact_combined_summary.tsv`](xcomet_paper_exact_combined_summary.tsv)、
  [`xcomet_paper_exact_combined_paired.tsv`](xcomet_paper_exact_combined_paired.tsv)、
  [`xcomet_paper_exact_eso_de_summary.tsv`](xcomet_paper_exact_eso_de_summary.tsv)、
  [`xcomet_paper_exact_eso_de_paired.tsv`](xcomet_paper_exact_eso_de_paired.tsv) 和
  [`xcomet_paper_exact_eso_de_validation.json`](xcomet_paper_exact_eso_de_validation.json)。
  独立 validator 从 11,496 条逐句结果反算并核对了 8 systems / 4 pairs；原 ACL
  24 systems / 12 pairs 的 validator 结果保持不变。

逐句 JSONL 与输入 bundle 位于 Hyper00 staging
`/data02/jaxan/RASST_rebuttal_20260710`；paper-exact ESO 逐句文件为
`results/xcomet_paper_exact_eso_de_20260712/segments.jsonl`。它们的预定
Hugging Face 目标为
`gavinlaw/rasst-main-result-data` 下的 versioned rebuttal artifact。目前没有可用的
Hugging Face 写入凭据，上传状态为 **pending**，本地 staging 不能视为 canonical。

## Paper-derived realistic glossary

状态：**fresh extraction blocked，推理未启动**。

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
- xCOMET-XXL 覆盖 9 systems / 4212 sentence segments；独立 validator 从逐句
  JSONL 反算全部 system 均值，状态为 `ok`。六个 degraded cells 相对同语言
  `0%` 的 xCOMET delta 为 En-Zh `-0.599/-1.020`、En-De
  `-0.752/-0.471`、En-Ja `-5.971/-1.071` points。
- 预注册定义、完整结果、compute placement、验证哈希和局限见
  [`retrieval_degradation_ablation.md`](retrieval_degradation_ablation.md)。

完整 runtime logs、instances 和 4212-row xCOMET JSONL 位于 Aries/Hyper00 staging，
预定 Hugging Face 目标为
`gavinlaw/rasst-rebuttal-retrieval-degradation-acl`。上传状态为 **blocked**：
Taurus/Aries 没有 token，Hyper00 现有凭据属于另一账号，未越权使用；在作者提供
授权凭据并上传前，staging 仍不是 canonical artifact。

## Failure analysis 与三语 exact-match audit

状态：**自动计数完成；draft audit 完成；作者复核 pending**。

- 固定 ACL `lm=2`，从 runtime retrieval timestamps、sentence-aligned
  `term_adoption.json` 和 paper-exact xCOMET segments 复算 971 个 De、1,173 个
  Zh 与 1,122 个 Ja gold occurrences。分析器用 `--require-acl-talks` 校验 scope，
  Medicine/ESO 没有进入分析。
- 自动化输出包含 on-time/late/never conditional accuracy、exact-miss failure chain、
  paired sentence xCOMET term-gain/tie/loss 分组，以及每个 occurrence 的 provenance。
- De/Zh/Ja 的 162/114/161 个 exact misses 和三语共 135 个 raw false-copy flags 已
  逐条给出 draft label 与理由。De 保守 morphology-aware 数字只加入 36 个
  compound/orthography 和
  14 个 morphology variants；69 个 paraphrases 与 13 个 boundary cases 不进入该
  数字。
- `term_map_false_copy` 是 candidate diagnostic，不是 true-noise ground truth。审计后
  De/Zh/Ja 分别只保留 2/2/4 个 harmful unsupported-hint adoption candidates；这仍是
  观察性标签，不是 retrieval 的因果效应。Ja 4 个 harmful cases 平均 xCOMET
  `-9.50` points，但 exact-tie 句也平均下降 `-3.96`，所以 term noise 只解释部分
  质量下降。Boundary flags 的 xCOMET 很低，说明 sentence alignment 是重要混杂。
- Git-tracked 轻量证据：
  [`term_failure_chain_acl_lm2.tsv`](term_failure_chain_acl_lm2.tsv)、
  [`xcomet_failure_groups_acl_lm2.tsv`](xcomet_failure_groups_acl_lm2.tsv)、
  [`retrieval_noise_audit_acl_lm2.tsv`](retrieval_noise_audit_acl_lm2.tsv)、
  [`german_morphology_manual_audit.tsv`](german_morphology_manual_audit.tsv)、
  [`zh_exact_miss_draft_audit.tsv`](zh_exact_miss_draft_audit.tsv)、
  [`ja_exact_miss_draft_audit.tsv`](ja_exact_miss_draft_audit.tsv) 和
  [`retrieved_false_copy_draft_audit.tsv`](retrieved_false_copy_draft_audit.tsv)。

完整 per-occurrence/per-sentence 输出已复制到本机 persistent ignored staging
`/Users/luojiaxuan/Documents/RASST/outputs/rebuttal_2026/term_failure_acl_lm2/{de,zh,ja}`，
预定 Hugging Face 目标为
`gavinlaw/rasst-rebuttal-term-failure-analysis-acl`，上传状态为 **pending**。
当前本机没有 Hugging Face CLI/写入凭据；在上传完成前，该本机 staging 仍不是
canonical artifact。

## Rebuttal 文本

英文工作稿位于 [`../../rebuttal_2026_draft.md`](../../rebuttal_2026_draft.md)。其中
所有 `PENDING` 都是提交保护标记：只有生成、复核并写入本索引的数字才可替换。
Masked BLEU 是 diagnostic，不是因果分解；TERM_ACC 必须称为 exact-form metric。
German morphology-aware 结果只能称为 non-expert author diagnostic，并且必须在
作者逐行复核 draft audit 后才能提交；De/Zh/Ja 更宽的语义 draft
`96.91/96.85/95.28%` 仅供内部定位 metric false negatives，不应作为 rebuttal
headline。
