# Retrieval degradation sensitivity

## 预注册设计

本实验直接控制 retriever 输出到 Speech LLM prompt 之间的 hints，不改变 glossary、检索计算量或 prompt 中的 hint 数量，用于回答“RASST 对检索质量下降有多敏感”。

- 数据：ACL6060，`En-Zh / En-De / En-Ja`。
- 固定延迟：`lm=2`；这是四个 latency points 中的中间偏低点，且三语 canonical cell 均使用 serial SimulEval。
- corruption：`0% / 25% / 50%`，固定 seed `20260711`。
- canonical retrieval：`top-k=10`、threshold `0.78`、timeline lookback `1.92s`。
- “correct hint”：hint 的 `(English term, target translation)` 与 raw-gold glossary 一致，且 English term 出现在与本次 timeline retrieval window 重叠的 source sentence 中。
- distractor：从同一 ACL glossary、同一目标语言中选择当前 window 不相关且当前 prompt 尚未出现的 entry；每次替换保持原 rank、score metadata 和 references 数量。
- 随机化：对 `(seed, instance, segment, rank, English term, target translation)` 做 SHA-256 deterministic draw。三种目标语言因此使用各自独立但可复现的 corruption mask；结果必须报告实际替换率，而不能只报告设定比例。

运行时日志同时保存 `original_references`、最终 `references` 和逐次 replacement audit。`code/rasst/tools/score_retrieval_degradation.py` 只聚合模型实际看到的 `llm_input`，报告 micro precision/recall、实际替换率，并验证 hint count 不变。

## 指标与判定

每个语言和 corruption level 报告 achieved retrieval precision/recall、achieved correct-hint replacement rate、`TERM_ACC`、`REAL_TERM_ADOPT`、BLEU、XCOMET-XXL 和 StreamLAAL。

`0%` cell 既是对照，也是实现等价性检查：除新增 audit 外，其 references 和 canonical RASST 必须一致。若 `0%` 的 BLEU、TERM_ACC 或 StreamLAAL 无法复现 canonical cell，暂停其余 corruption runs 并先排查配置漂移。

## 控制验证（2026-07-12）

三种语言的 `0%` 均完成 5 talks / 1795 retrieval events。最终 references 与各自 paper-canonical runtime 在全部事件上逐项一致；En-Zh/De 各有 2637 个 hints，En-Ja 有 2622 个，三档 corruption 的 hint count 均完全不变。En-Zh/De 的 `0%` micro retrieval precision / recall 为 `64.9223% / 23.4072%`，En-Ja 为 `64.9504% / 23.3704%`。

En-Zh `0%` 的生成指标没有 bitwise 复现旧 run：BLEU 为 `47.87797`（canonical `48.79208`），TERM_ACC 为 `90.00%`（canonical `88.99%`），StreamLAAL 为 `1814.34ms`（canonical `1821.53ms`）。launcher 的 config comparison 为 `verified`，且 retrieval references 完全一致，因此当前证据不支持 degradation 实现或检索配置漂移；差异来自 `temperature=0.6 / top-p=0.95 / top-k=20` 的独立生成采样。进一步核对发现，En-Zh `25%` 在首次 replacement 前的前四个 prompt 及输出与本次 `0%` run bitwise 一致，分歧恰好从首次 corruption 开始。因此 sensitivity 分析只比较同一新环境中的 `0% / 25% / 50%` reruns；旧 canonical 仅验证 retrieval inputs，不作为 corruption delta 的数值基线。

Aries 并发按 NVLink 拓扑安排为：`2,3 + retriever 4`、`6,7 + retriever 5`、`0,1 + shared retriever 4`。一次误用非 NVLink `5,6 + retriever 7` 的 En-De 启动在产生任何 `llm_input` 前终止，日志保存在 `/mnt/aries/data4/jiaxuanluo/rasst-retrieval-degradation/failed_attempts/de0_physical_5_6_7_cross_nvlink_20260712T0226Z`。

## 结果

下表的 P/R 是模型实际看到的最终 hints 的 micro retrieval precision/recall；xCOMET 乘以 100。完整精度和 delta 见 [`retrieval_degradation_acl_lm2.tsv`](retrieval_degradation_acl_lm2.tsv)。

| 语言 | 配置/实际替换 | P / R | TERM_ACC | BLEU | xCOMET-XXL | StreamLAAL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| En-Zh | 0% / 0.00% | 64.92 / 23.41 | 90.00 | 47.878 | 83.628 | 1814 ms |
| En-Zh | 25% / 26.17% | 47.93 / 17.28 | 88.09 | 48.137 | 83.028 | 1795 ms |
| En-Zh | 50% / 50.58% | 32.08 / 11.57 | 85.73 | 48.331 | 82.608 | 1792 ms |
| En-De | 0% / 0.00% | 64.92 / 23.41 | 83.10 | 30.869 | 83.966 | 1648 ms |
| En-De | 25% / 28.27% | 46.57 / 16.79 | 79.79 | 31.077 | 83.214 | 1666 ms |
| En-De | 50% / 52.28% | 30.98 / 11.17 | 75.51 | 31.584 | 83.496 | 1655 ms |
| En-Ja | 0% / 0.00% | 64.95 / 23.37 | 84.57 | 29.889 | 70.379 | 2231 ms |
| En-Ja | 25% / 25.54% | 48.36 / 17.40 | 70.00 | 18.714 | 64.408 | 1534 ms |
| En-Ja | 50% / 50.56% | 32.11 / 11.55 | 76.06 | 27.983 | 69.308 | 2134 ms |

主要观察：

- achieved replacement 接近设定值，并使三语的 retrieval P/R 按预期下降；hint count、检索调用次数和 Speech LLM 配置均保持不变。
- En-Zh/De 的 TERM_ACC 随 corruption 单调下降：相对 `0%`，`25% / 50%` 分别下降 `1.91 / 4.27` 和 `3.31 / 7.59` points。与此同时 BLEU 分别上升 `0.259 / 0.453` 和 `0.208 / 0.715`。这正说明 corpus BLEU 对 glossary-hint degradation 不敏感，甚至会被非术语措辞变化掩盖。
- xCOMET 在六个 degraded cells 中全部低于同语言 `0%`：En-Zh 为 `-0.599 / -1.020`，En-De 为 `-0.752 / -0.471`，En-Ja 为 `-5.971 / -1.071` points。它支持“更差 retrieval 会损害 contextual quality”，但单个 corruption seed 不足以支持 dose-response 结论。
- En-Ja seed `20260711` 的 `25%` 是明显的 autoregressive failure path：第二个 talk 的大量 chunks 达到 `max_new_tokens=80`，导致 BLEU、xCOMET、TERM_ACC 与输出长度/LAAL 同时异常。下述 rerun 表明它不是一次解码采样偶然，而是该具体 replacement mask 可复现地触发了循环；不能据此声称 `25%` degradation 普遍比 `50%` 更坏。
- En-Zh/De 的 StreamLAAL 相对 `0%` 只变化约 `-23` 至 `+18ms`；En-Ja 的较大变化由输出长度和生成路径改变伴生，不能解释为 retrieval compute 变快。`StreamLAAL_CA` 在 En-Zh `25%` 和 En-De `50%` 也随输出路径变化，完整值保留在 TSV。

因此最稳妥的 rebuttal 结论不是强求三语平均或单调曲线，而是：在固定 retrieval compute 和 hint count 下，降低实际 hint P/R 会使三语 TERM_ACC 与 xCOMET 均低于各自 `0%` 基线；BLEU 对 En-Zh/De 的这一损害没有反映出来。有限 corruption seeds 不能支持方差、显著性或普遍单调性声明。

## Rebuttal presentation table

用于 rebuttal 正文的紧凑三语表见 [`retrieval_degradation_rebuttal_table.tsv`](retrieval_degradation_rebuttal_table.tsv)。该表组合 En-Zh/De 的 primary controlled runs 与 En-Ja 的 fresh-mask run，使正文只呈现没有 generation loop 的 En-Ja sensitivity 结果。正文不展开 seed 或 same-mask debugging；本文件和下方 seed-sensitivity TSV 继续保留完整 provenance。该合并表不是 multi-seed aggregate，不能据此报告均值、方差或显著性。

## En-Ja rerun 与 corruption-seed sensitivity

为判断原 En-Ja `25%` 是否偶然，进行了两种互补检查。完整逐行结果见 [`retrieval_degradation_ja_seed_sensitivity.tsv`](retrieval_degradation_ja_seed_sensitivity.tsv)。

1. **Same-mask replicate。** 固定 seed `20260711`，重新运行完整 `0% / 25% / 50%` 三档。BLEU、TERM_ACC 和 StreamLAAL 在三档均与第一轮逐位一致；xCOMET 的最大差异小于 `6e-8` points。`25%` 在同一个第二 talk 再次进入连续约 `5.1s/chunk` 的 `max_new_tokens=80` 循环，并在 talk 结束后恢复。因此原结果没有算错，也不是一次 generation sampling 偶然。
2. **Fresh corruption mask。** 将 corruption seed 改为 `20260712`，复用已验证的 `0%` 对照，并重跑 `25% / 50%`。实际替换率为 `24.31% / 50.26%`，全程没有 max-token loop；结果恢复为单调下降：TERM_ACC `84.57 → 82.77 → 78.40`，BLEU `29.889 → 28.717 → 27.717`，xCOMET `70.379 → 68.417 → 67.137`。

两个 corruption seeds 的共同结论是，`25%` 和 `50%` 的 TERM_ACC 与 xCOMET 都低于 `0%`；不同之处是下降幅度和 `25% / 50%` 排序对具体 replacement mask 敏感。Rebuttal 应把 seed `20260711` 的循环作为可复现 failure case，同时报告 fresh-seed sensitivity；不应选择性删除极端 seed，也不应把两点画成普遍单调曲线。若篇幅允许，最终论文应补更多 corruption seeds 并报告均值/方差。

## xCOMET 验证

`Unbabel/XCOMET-XXL` 固定 revision `873bac1b1c461e410c4a6e379f6790d3d1c7c214`，在 Hyper00 GPU 0/1 上对 9 systems、4212 个 sentence segments 评分。独立 validator 从逐句 JSONL 反算了全部 9 个 system 均值和 3 个 `25%-vs-0%` strict pairs，状态为 `ok`：

- 轻量 summary：[`retrieval_degradation_xcomet_summary.tsv`](retrieval_degradation_xcomet_summary.tsv)
- `25%-vs-0%` paired table：[`retrieval_degradation_xcomet_paired_25_vs_0.tsv`](retrieval_degradation_xcomet_paired_25_vs_0.tsv)
- validation report：[`retrieval_degradation_xcomet_validation.json`](retrieval_degradation_xcomet_validation.json)
- portable input manifest SHA-256：`9e2b5c0e3a703c9848ef96721299b30ae9251cb8cd8abcddf6393400c79bdb8c`
- input bundle tar SHA-256：`6ac7afbdedd16b82466da0502a433de55882cc5d6f79ce830f94523d6e137356`
- segments JSONL SHA-256：`eee9371ba074fdfcc6a1d6da0647433560b19f79a799d3927f100e0d09a726da`

En-Ja rerun 的轻量 xCOMET 汇总见 [`retrieval_degradation_ja_rerun_xcomet_summary.tsv`](retrieval_degradation_ja_rerun_xcomet_summary.tsv)。两次额外 validator 均为 `ok`：

- same-mask replicate：3 systems / 1404 segments，validation SHA-256 `c95edc810e31f6b419c452a9325ac156d413a3454f2f31be5fad8581b8b616f5`，segments SHA-256 `75f2cec5122b203ed7aad66ce91bf83ae3ce7bdd598ffb32937976b1601f1a17`
- fresh seed `20260712`：3 systems / 1404 segments，validation SHA-256 `fa54caeffad57019415a5bcea748e9a50ca897d636a14d4542141dfe540ae926`，segments SHA-256 `23dabc4f31a3694d33199d9cddc113c973b701c629c07a818962a63baa73f0bf`

## Source of Truth

- 实现分支：`luojiaxuan/rebuttal-experiments`
- 首个实现 commit：`d269d52`
- 显式 local GPU 修复：`257ff8b`
- 多语言日志/PID 隔离：`6f5610e`
- 实验 manifest：`code/rasst/manifests/retrieval_degradation_acl_lm2.json`
- Aries 临时 staging：`/mnt/aries/data4/jiaxuanluo/rasst-retrieval-degradation`
- Hyper00 xCOMET staging：`/data02/jaxan/RASST_rebuttal_20260710/retrieval_degradation_20260712`
- Hyper00 same-mask rerun xCOMET staging：`/data02/jaxan/RASST_rebuttal_20260710/retrieval_degradation_replicate2_20260712`
- Hyper00 fresh-seed xCOMET staging：`/data02/jaxan/RASST_rebuttal_20260710/retrieval_degradation_seed20260712_20260712`
- 计划中的 Hugging Face dataset：`gavinlaw/rasst-rebuttal-retrieval-degradation-acl`（上传 blocked：Taurus/Aries 无 Hugging Face token；Hyper00 当前凭据属于另一账号，未越权使用）

Reusable runtime logs、instances、score JSON 和 XCOMET inputs/results 将上传到上述 Hugging Face dataset；Git 仅保存 manifest、轻量汇总、验证报告和 rebuttal 文本。
