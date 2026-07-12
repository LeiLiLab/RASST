# Retrieval degradation sensitivity

## 预注册设计

本实验直接控制 retriever 输出到 Speech LLM prompt 之间的 hints，不改变 glossary、检索计算量或 prompt 中的 hint 数量，用于回答“RASST 对检索质量下降有多敏感”。

- 数据：ACL6060，`En-Zh / En-De / En-Ja`。
- 固定延迟：`lm=2`；这是四个 latency points 中的中间偏低点，且三语 canonical cell 均使用 serial SimulEval。
- corruption：`0% / 25% / 50%`，固定 seed `20260711`。
- canonical retrieval：`top-k=10`、threshold `0.78`、timeline lookback `1.92s`。
- “correct hint”：hint 的 `(English term, target translation)` 与 raw-gold glossary 一致，且 English term 出现在与本次 timeline retrieval window 重叠的 source sentence 中。
- distractor：从同一 ACL glossary、同一目标语言中选择当前 window 不相关且当前 prompt 尚未出现的 entry；每次替换保持原 rank、score metadata 和 references 数量。
- 随机化：对 `(seed, instance, segment, rank, term)` 做 SHA-256 deterministic draw。结果必须报告实际替换率，而不能只报告设定比例。

运行时日志同时保存 `original_references`、最终 `references` 和逐次 replacement audit。`code/rasst/tools/score_retrieval_degradation.py` 只聚合模型实际看到的 `llm_input`，报告 micro precision/recall、实际替换率，并验证 hint count 不变。

## 指标与判定

每个语言和 corruption level 报告 achieved retrieval precision/recall、achieved correct-hint replacement rate、`TERM_ACC`、`REAL_TERM_ADOPT`、BLEU、XCOMET-XXL 和 StreamLAAL。

`0%` cell 既是对照，也是实现等价性检查：除新增 audit 外，其 references 和 canonical RASST 必须一致。若 `0%` 的 BLEU、TERM_ACC 或 StreamLAAL 无法复现 canonical cell，暂停其余 corruption runs 并先排查配置漂移。

主要分析看每种语言从 achieved retrieval quality 到 `TERM_ACC / REAL_TERM_ADOPT / BLEU / XCOMET-XXL` 的单调变化；不把三种语言强行平均成唯一结论。LAAL 理论上应基本不变，明显变化表示生成长度或 runtime 行为发生了次生变化，需要单独解释。

## Source of Truth

- 实现分支：`luojiaxuan/rebuttal-experiments`
- 首个实现 commit：`d269d52`
- 显式 local GPU 修复：`257ff8b`
- 实验 manifest：`code/rasst/manifests/retrieval_degradation_acl_lm2.json`
- Aries 临时 staging：`/mnt/aries/data4/jiaxuanluo/rasst-retrieval-degradation`
- 计划中的 Hugging Face dataset：`gavinlaw/rasst-rebuttal-retrieval-degradation-acl`（尚未上传）

Reusable runtime logs、instances、score JSON 和 XCOMET inputs/results 将上传到上述 Hugging Face dataset；Git 仅保存 manifest、轻量汇总、验证报告和 rebuttal 文本。
