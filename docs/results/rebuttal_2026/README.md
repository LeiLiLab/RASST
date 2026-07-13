# RASST rebuttal 2026：实验与 Source of Truth

本目录是 rebuttal 补实验的 Git-tracked 索引。代码、轻量结果表、实验决策和
进度以本仓库 `main` 分支为准；公开评测数据和模型
仍以 README 中列出的 Hugging Face 仓库为准。大模型权重、逐句评分和生成式
glossary 原始响应不进入 Git。

## 当前结论

- **xCOMET streaming pairing 已修复。** 旧 ACL sentence-level `+0.1158` 依赖
  逐句 mWER resegmentation；MFA/raw-chunk audit 证实它会生成错配的
  `src/mt/ref` 个例，因此不再推荐作为 rebuttal claim。固定 5-sentence blocks 的
  ACL 重跑为 24 systems / 12 cells / 2,304 blocks，cell-macro 差值 `-0.8699`
  xCOMET points（乘以 100）。它修复了四个 Ja 极端负例中的三个，但没有逆转
  aggregate。协议、结果和提交建议见
  [`xcomet_acl_block5_report.md`](xcomet_acl_block5_report.md)。
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
  `0%`。En-Ja seed `20260711` 的 same-mask full rerun 精确复现了 `25%` collapse；
  fresh mask seed `20260712` 无循环，TERM_ACC 为 `84.57 → 82.77 → 78.40%`，
  xCOMET 为 `70.379 → 68.417 → 67.137`。两种 mask 的 degraded conditions 均低于
  `0%`，但幅度和排序对 mask 敏感，不能把单 seed 解读为普遍单调 dose response。
  完整表和限制见
  [`retrieval_degradation_ablation.md`](retrieval_degradation_ablation.md)。
- **LLM-as-a-judge cost pilot 已完成。** 100 个 paired system outputs 上，当前可用的
  `gemini-3.1-pro-preview` 与 `gemini-2.5-flash` 使用 WMT25 Appendix A prompt 的
  score Pearson 为 `0.9241`。完整 22,728-request Batch（已经包含 InfiniSST
  baseline）分别预计 `$109.06`（bootstrap `$101.95–$116.08`）和 `$48.19`
  （`$43.72–$52.93`）。`gemini-2.5-pro` 对当前账号返回“不再向 new users 开放”，
  因此 Pro 3.1 只能称为 current proxy。完整协议与 provenance 见
  [`llm_judge_pilot_100.md`](llm_judge_pilot_100.md)。

## xCOMET

状态：**已完成；旧 sentence-level ACL 口径已被 block-aware 诊断取代。**

- 当前推荐诊断是 ACL En-Zh/De/Ja 的 fixed 5-sentence blocks：24 system rows /
  12 cells / 2,304 blocks。旧 ACL 与 ESO En-De sentence-level artifacts 保留作
  provenance，但不能再和 block scores 合并成一个 16-cell macro。
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
- 旧 ACL sentence-level 12-cell macro 为 `+0.1158`，但 case audit 证实逐句
  mWER 会把 delayed output 配到错误 source/reference，因此该值只保留为历史诊断，
  不应进入 rebuttal。固定 5-sentence block 重跑为 RASST `57.8310`、InfiniSST
  `58.7009`，差值 `-0.8699`；完整结果见
  [`xcomet_acl_block5_report.md`](xcomet_acl_block5_report.md)、
  [`xcomet_acl_block5_summary.tsv`](xcomet_acl_block5_summary.tsv)、
  [`xcomet_acl_block5_paired.tsv`](xcomet_acl_block5_paired.tsv) 和
  [`xcomet_acl_block5_validation.json`](xcomet_acl_block5_validation.json)。
- New InfiniSST × paper-exact RASST 的 ESO En-De 4 cells 为 RASST `75.9398`、
  InfiniSST `77.3245`，平均差值 `-1.3848`，0/4 cells 为正。这是 sentence-level
  历史结果，不与 ACL block-aware 分数求 macro。
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

## LLM-as-a-judge（WMT25 prompt）

状态（2026-07-12 01:27 PDT）：**100-request pilot、corrected-ESO full preparation 和
pre-submit validation 已完成；13/16 shards 已完成下载，Medicine En-De lm2--4 已提交
但因 Gemini project 预付余额耗尽而仍为 pending。完整结果尚不可发布。**

- 使用 WMT25 Appendix A 的 exact Task 1 prompt；judge request 只含 source 与
  hypothesis。评分 inference 是 reference-free，但逐句 unit 来自依赖 human
  reference 的 `mwerSegmenter` alignment，不能写成全流程 reference-independent。
- 全量矩阵为 ACL 5 talks × 4 LM × En-Zh/De/Ja，以及 ESO/Medicine 5 talks ×
  4 LM × En-De：共 32 system rows / 16 strict pairs / 22,728 requests，按 pair cell
  拆为 16 shards。该数量已经包含 RASST 和 InfiniSST，不需要再乘以 2。
- ACL 只从 SHA-256
  `43b5581bc79e8c389383e6fb84b684f4f7207334c114a0cc0b8b19d47d2a459b`
  的 artifact 选择 11,232 个 `acl_tagged_raw` rows。ESO/Medicine 的 InfiniSST
  lm1--3 与 lm4 分别来自 SHA-256 `244a8fb4...fcca3` 和 `50708b41...15df9` 的
  independently validated new-baseline artifacts；RASST 只从 paper-exact SHA-256
  `40454563...1e997` 选择四档。三个文件均按 method/lm 过滤，不能整文件直读。
- WMT25 的 `gemini-2.5-pro` 在当前账号实测返回 404。100-request paired pilot 使用
  `generation_config={}` 和 model-default thinking；当前建议可用的
  `gemini-2.5-flash`，全量 Batch 投影约 `$48.19`（bootstrap 95% `$43.72–$52.93`）。
  `gemini-3.1-pro-preview` 投影约 `$109.06`，且不是 WMT25 model reproduction。
- API key 只允许从 Taurus mode `0600` 私有文件读取，key 值不进入 Git、命令、
  manifest 或日志。Staging root、pinned venv、
  exact prompt、输入路径/哈希、费用风险和 HF pending 状态见
  [`llm_judge_wmt25_protocol.md`](llm_judge_wmt25_protocol.md)；pilot 详细数字见
  [`llm_judge_pilot_100.md`](llm_judge_pilot_100.md)。
- 正式 bundle 位于 Taurus `full_flash_api_default_corrected_eso/`；run config
  SHA-256 为
  `5d01d8c2790be654272dc848aba265239a4c34838c91490b850148a76232628a`，manifest
  SHA-256 为
  `67d01bde0d93f748312ccdc2308d9056bfcef490c543a3f27f761769543ddad3`；独立
  pre-submit validation 报告 SHA-256 为 `844e1d85...ed38e`。旧 baseline bundle 已
  标记 superseded，未提交任何 job。
- 已下载的 14,106 responses 中有 1 个非整数-only 格式违例；没有截取解释末行分数。
  相同配置的 format-only retry 因余额 429 未产生评分。原始 response、retry ledger、
  16 shard states 与 monitor log 均保留，待余额恢复后继续。

## Paper-derived realistic glossary

状态：**legacy paper-derived glossary v1 的三语 `lm=2` default-setting 结果已由
作者确认；fresh model-ID-pinned re-extraction 仍未完成。**

- 每个 RASST run 只使用对应 ACL paper 导出的 glossary 作为 inference index；
  index 构建不读取 transcript、reference、gold tags 或 gold evaluation glossary。
- RASST 与 InfiniSST 最终都用完整 tagged-raw ACL glossary（SHA-256
  `f9f171c6475c4bb19250f5f93063a5ef034cbdcc1f8a995c593647718cf9a5b6`）评分；
  未被 paper-derived index 覆盖的 gold terms 仍计为错误。
- 作者确认的 rebuttal readout：

| Language | Reported TERM_ACC RASST / InfiniSST (delta) | Pooled correct counts | BLEU RASST / InfiniSST (delta) |
| --- | ---: | ---: | ---: |
| En-Zh | 77.87 / 75.17 (**+2.70 pp**) | 693/890 vs. 669/890 | 46.3280 / 45.8268 (**+0.5012**) |
| En-Ja | 65.32 / 65.96 (**-0.64 pp**) | 614/940 vs. 620/940 | 27.7656 / 27.7202 (**+0.0455**) |
| En-De | 70.91 / 68.21 (**+2.70 pp**) | 652/935 vs. 632/935 | 29.2086 / 30.2743 (**-1.0657**) |

De 的 reported TERM_ACC 与 pooled count ratio（69.73% vs. 67.59%，delta
+2.14 pp）不是同一聚合口径，因此必须分列，不能写成
`652/935 = 70.91%`。正式 rebuttal 使用作者确认的 reported TERM_ACC；pooled counts
仅保留作 provenance。三语 reported TERM_ACC macro delta 为 `+1.59 pp`；BLEU
为 mixed，因此该实验只支持 non-oracle terminology robustness，不支持 uniform
overall-quality improvement。

Legacy glossary 的 exact historical Gemini model identifier 未保存在 extraction
旁，正式文本只能称 `paper-derived glossary v1`，不能称 fresh Gemini 2.5 Flash。
对应的 Git source of truth 已同步到 `main@60c995e`，包括 reported/pool 两套
TERM_ACC 字段和 author-confirmed snapshot。
完整数据 artifact 的预定 Hugging Face 目标仍为
`gavinlaw/rasst-main-result-data` 下的 versioned rebuttal artifact；在上传前状态为
**pending**。

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
- En-Ja same-mask full rerun 逐位复现 BLEU、TERM_ACC 与 StreamLAAL，并在同一 talk
  再现 max-token loop；fresh corruption seed `20260712` 则无循环，并在 `0/25/50%`
  上得到 TERM_ACC `84.57/82.77/78.40%` 与 xCOMET `70.379/68.417/67.137`。
  两次额外 xCOMET validator 均为 `ok`；逐行结果见
  [`retrieval_degradation_ja_seed_sensitivity.tsv`](retrieval_degradation_ja_seed_sensitivity.tsv)
  和 [`retrieval_degradation_ja_rerun_xcomet_summary.tsv`](retrieval_degradation_ja_rerun_xcomet_summary.tsv)。
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

按 Reviewer Mzub、oktu、gbii 顺序整理的精简 OpenReview 单评论稿位于
[`../../rebuttal_2026_openreview_responses.md`](../../rebuttal_2026_openreview_responses.md)。
三段正文分别为 `3711 / 4414 / 3972` characters，均低于 5000-character 上限；该稿
只使用已验证的 main/rebuttal-experiments 结果，并明确排除 LLM-as-a-judge 与宽语义
audit。文末的内部取舍和 evidence SoT 不应提交到 OpenReview。

较长的英文历史工作稿位于 [`../../rebuttal_2026_draft.md`](../../rebuttal_2026_draft.md)，
不应直接提交；以 concise response 文件为准。其中
所有 `PENDING` 都是提交保护标记：只有生成、复核并写入本索引的数字才可替换。
Masked BLEU 是 diagnostic，不是因果分解；TERM_ACC 必须称为 exact-form metric。
German morphology-aware 结果只能称为 non-expert author diagnostic，并且必须在
作者逐行复核 draft audit 后才能提交；De/Zh/Ja 更宽的语义 draft
`96.91/96.85/95.28%` 仅供内部定位 metric false negatives，不应作为 rebuttal
headline。
