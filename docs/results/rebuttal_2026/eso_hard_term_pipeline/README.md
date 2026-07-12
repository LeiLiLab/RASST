# ESO hard-term glossary pipeline provenance

本目录记录 rebuttal 中 ESO/Medicine hard-term glossary 的可审计构建协议。Prompt、
步骤说明和轻量配置属于 Git source of truth；逐句模型响应、最终 glossary 与评测输入
仍属于数据 artifact，应沿用 `gavinlaw/rasst-main-result-data` 的 versioned artifact
管理，不应只留在本机目录。

## 七阶段协议

作者确认：pipeline 中只有 Stage 1、2、5 调用 LLM，三者均使用 **GPT-5.4**。
Prompt 本身不保存 model ID，因此 model provenance 来自作者确认；实际运行清单在
revision 中还应同时记录 API model identifier、日期与输入/output hashes。

| Stage | 类型 | 作用 |
| --- | --- | --- |
| 1 | GPT-5.4；10 sentences/batch | 使用 batch 上下文生成完整 En-Zh/En-Ja 句译文与候选术语；原始 ESO German 译文原样保留。 |
| 2 | GPT-5.4 | 强制 term translation 与句译文 exact-match；统一缩写处理；调整或删除噪声、重复、不可可靠翻译的 term。 |
| 3 | 非 LLM | 恢复被误删的部分 terms，并在 sample level 统一各 sentence 的 decision。 |
| 4 | 非 LLM | 对 source term 做 exact-match，删除 source sentence 中不存在的 terms。 |
| 5 | GPT-5.4；3 sentences/batch | 对照 no-RAG baseline 判断术语是否已被正确翻译，据此确定 hard terms；较小 batch 用于降低上下文幻觉。 |
| 6 | 人工 | Manual check glossary。 |
| 7 | 非 LLM | 生成 final output。 |

Stage 1 的 10-sentence batch 和 Stage 5 的 3-sentence batch 是运行配置，不在 prompt
模板内部硬编码。Stage 3、4、6、7 不调用 LLM；当前记录不将未提供的实现细节泛称为
全自动或 domain-expert procedure。

## Prompt source of truth

| Prompt | Git path | 原始本机 SHA-256 |
| --- | --- | --- |
| Stage 1 | [`stage1_translate.txt`](stage1_translate.txt) | `75b7d126056a2af7c4f9f524aef75737df31c1c27593cca14b0882d900729481` |
| Stage 2 | [`stage2_term_match.txt`](stage2_term_match.txt) | `b1ef45cffb4f8b56ddd11b307bc5ede6be95abda60e7ddbdf42deed5d3a68bfe` |
| Stage 5 | [`stage5_hard_term_judge.txt`](stage5_hard_term_judge.txt) | `e345773ab96f440db087f54b9468e97b37597ee3bc1bf257d665c5903a230be1` |

原始 staging 路径为 `/Users/luojiaxuan/Downloads/prompts/`。Git 版本提交后才是 prompt
的 canonical copy；Downloads 目录不再承担 source-of-truth 角色。

## Rebuttal 中允许的 provenance 口径

- ESO En-De sentence references 是原始 ESO dataset 的 manual translations；本项目的
  En-Zh/En-Ja sentence references 在 Stage 1 由 GPT-5.4 生成。
- ESO hard-term inventory 应称为 **GPT-5.4-assisted and manually checked**，不能称为
  fully human-authored、professional-translator verified 或 domain-expert annotated。
- Main paper 的 ESO En-Zh/En-Ja 应删除 BLEU、xCOMET 等依赖 synthetic sentence
  references 的 translation-quality claims；如保留 term-only readout，必须与 sentence
  reference metrics 分表，并明确 glossary provenance。
- Stage 5 用于构建 hard-term 子集，不作为 rebuttal 的通用 translation-quality judge
  结果。不要把尚未报告的 LLM-as-a-judge pilot 混入正式回复。

## ACL 60/60 对照 provenance

ACL 与 ESO 的标注来源不能混写。ACL 60/60 官方论文说明：60/60 initiative 创建了
non-exhaustive terminology lists；source spans 自动标记；英文转写由 aiXplain
professionally post-edit 并经过三层 review，之后由 domain expert 核验、修正技术词；
译文由 Translated professionally post-edit，第一位 annotator 是目标语言母语者，
第二位复核输出与一致性，post-editors 负责纠正并标注 source-tagged terms 的 target
spans。论文没有说 target translators 本身是 domain experts。

官方来源：[Salesky et al. (2023), Secs. 3.4--3.7 and App. A.5](https://aclanthology.org/2023.iwslt-1.2/)。
