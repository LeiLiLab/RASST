# RASST rebuttal 2026：实验与 Source of Truth

本目录是 rebuttal 补实验的 Git-tracked 索引。代码、轻量结果表、实验决策和
进度以本仓库 `luojiaxuan/rebuttal-experiments` 分支为准；公开评测数据和模型
仍以 README 中列出的 Hugging Face 仓库为准。大模型权重、逐句评分和生成式
glossary 原始响应不进入 Git。

## 当前结论

- **xCOMET-XXL 已完成。** ACL 12 cells 的 cell-macro 差值为 `+0.1158`
  xCOMET points（乘以 100），8/12 为正；ESO En-De 为 `-3.1716`，0/4 为正。
  该结果不支持“overall translation quality 普遍提升”，应将论文主张收窄为
  terminology handling 改善且 contextual quality effects mixed。
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

## xCOMET

状态：**已完成并独立复算验证**。

- 评分矩阵：ACL En-Zh/De/Ja 与 ESO En-De，RASST/InfiniSST，4 个 latency settings，
  共 32 system rows / 16 strict pairs / 22,728 sentence segments。
- Metric：`Unbabel/XCOMET-XXL`，revision
  `873bac1b1c461e410c4a6e379f6790d3d1c7c214`。
- Encoder tokenizer/config：`facebook/xlm-roberta-xxl`，revision
  `03e0fb540c3c9afd4bdda0072e7cb82d2eafd060`。
- Taurus 原始输入清单：[`xcomet_input_manifest.taurus.tsv`](xcomet_input_manifest.taurus.tsv)。
- 可移植输入 bundle：40 个 content-addressed payloads、26,100,339 bytes；portable
  manifest SHA-256 为
  `dbf07fd4f8fc6460f2edd0cf1c167ffe61740da6cab63cab9307d3775ad84e89`。
- ACL 12 cells 的 cell-macro xCOMET（乘以 100）为 RASST `78.7042`、
  InfiniSST `78.5884`，平均差值 `+0.1158`，8/12 cells 为正；其中 En-Zh
  `+0.6356`、En-De `+0.0050`、En-Ja `-0.2933`。
- ESO En-De 4 cells 为 RASST `74.8694`、InfiniSST `78.0410`，平均差值
  `-3.1716`，0/4 cells 为正。16 cells 合计差值为 `-0.7060`，8/16 为正。
- 结果是 mixed；不能用 xCOMET 声称 RASST 的 overall translation quality 普遍或
  显著提升。详细结果、失败恢复链条、环境和全部哈希见
  [`xcomet_xxl_report.md`](xcomet_xxl_report.md)。
- Git-tracked 轻量产物：[`xcomet_xxl_summary.tsv`](xcomet_xxl_summary.tsv)、
  [`xcomet_xxl_paired.tsv`](xcomet_xxl_paired.tsv) 和
  [`xcomet_xxl_validation.json`](xcomet_xxl_validation.json)。独立 validator 从
  22,728 条逐句结果反算并核对了全部 system 与 paired statistics。

逐句 JSONL 与 DDP prediction 文件位于 Hyper00 staging
`/data02/jaxan/RASST_rebuttal_20260710`。它们的预定 Hugging Face 目标为
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

## Rebuttal 文本

英文工作稿位于 [`../../rebuttal_2026_draft.md`](../../rebuttal_2026_draft.md)。其中
所有 `PENDING` 都是提交保护标记：只有生成、复核并写入本索引的数字才可替换。
Masked BLEU 是 diagnostic，不是因果分解；TERM_ACC 必须称为 exact-form metric，
不能声称已经解决 German morphology 或合法 paraphrase。
