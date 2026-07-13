# Ja `lm=1` curriculum retrain：ACL 评测

## 结论

增加短 chunk `lm=1` 训练覆盖后，最低延迟点的翻译质量明显恢复，但没有完全追平
InfiniSST。相对原 RASST，新模型的逐句 xCOMET-XXL 提升 `+1.972` points，BLEU
提升 `+1.231`，masked BLEU 提升 `+0.981`；同时 TERM_ACC 下降 `3.93 pp`。
因此结果支持“原 Ja `lm=1` checkpoint 对短 chunk 覆盖不足是质量下降的一个原因”，
但不支持“重训已经消除最低延迟的整体质量差距”。

## 主结果

所有 BLEU、TERM_ACC 和 latency 来自 canonical offline post-eval；不使用推理过程中
打印的 raw SimulEval BLEU。xCOMET 按作者最终指定的逐句协议计算：每个
source/reference sentence 对应一个 mWER-resegmented hypothesis segment，明确设置
`sentences_per_segment=1`，不使用 block-aware 聚合。xCOMET 表中数值乘以 100。

| System | BLEU | Masked BLEU | TERM_ACC | Correct | StreamLAAL | xCOMET-XXL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| InfiniSST | 24.400 | 21.717 | 62.45% | 587/940 | 1613 ms | 63.256 |
| Original RASST | 22.662 | 19.443 | 81.70% | 768/940 | 1478 ms | 59.153 |
| RASST + `lm=1` curriculum | **23.893** | **20.424** | **77.77%** | 731/940 | 1522 ms | **61.124** |

相对原 RASST，新模型把逐句 xCOMET 对 InfiniSST 的差距从 `-4.103` 缩小到
`-2.132` points；BLEU 差距从 `-1.739` 缩小到 `-0.507`。新模型仍比 InfiniSST
高 `+15.32 pp` TERM_ACC（`+144` correct occurrences），但比原 RASST 少 37 个
exact term hits。StreamLAAL 比原 RASST 增加约 `44.7 ms`，仍比 InfiniSST 低约
`90.6 ms`；StreamLAAL_CA 变差约 `196.9 ms`，不能把该 checkpoint 表述为所有
latency/quality 指标的 Pareto win。

## 训练干预

- 原 Ja SFT：12,500 rows，其中 1,048 个 all-`lm=1` rows（8.38%）。
- 新增 curriculum：对同一批 1,048 个 all-`lm=1` rows 使用 seed 43 独立重采样
  term-map distractors，监督 target 不变。
- 最终训练集：13,548 rows，all-`lm=1` 占 2,096 rows（15.47%）。
- 训练：Aries 4x A6000，TP=2 / EP=2，LoRA r32 / alpha32，1 epoch；完成
  `573/573` steps，dev loss `0.29417345`。

详细训练、导出和 ACL artifact hash 见
[`20260713__speech_llm_ja_lm1_curriculum_r32a32_ep1_a6000_4g.md`](../../provenance/slm/20260713__speech_llm_ja_lm1_curriculum_r32a32_ep1_a6000_4g.md)。

## xCOMET 验证

`Unbabel/XCOMET-XXL` 固定 revision
`873bac1b1c461e410c4a6e379f6790d3d1c7c214`。Hyper00 GPU 3/4 对 3 systems、
5 talks/system、468 sentences/system，共 1,404 segments 评分。独立 validator
复算全部 system means 和新模型对 InfiniSST 的 468 strict pairs，状态为 `ok`：

- summary：[`ja_lm1_curriculum_xcomet_sentence_summary.tsv`](ja_lm1_curriculum_xcomet_sentence_summary.tsv)，SHA-256 `5e84ff05e78ceeb336a51f18f3a6dc0bfbd835c79e2086e496dc0d94e58b513b`；
- paired：[`ja_lm1_curriculum_xcomet_sentence_paired.tsv`](ja_lm1_curriculum_xcomet_sentence_paired.tsv)，SHA-256 `b0ddaeecc934e5e64fd1d08beb1eb49d7e784123e7c890e7b272a76989b504c4`；
- validation：[`ja_lm1_curriculum_xcomet_sentence_validation.json`](ja_lm1_curriculum_xcomet_sentence_validation.json)，SHA-256 `0e3a6f3e79e312324835592d05a04ac5bb01dadb1fd2b3875f1844d12750f30a`；
- full segments staging SHA-256：`6ea75fd94d72dba025b8fb16c7c16ec93011018e2642b836d3c3c30505a130d0`。

逐句 win/tie/loss 为新模型 `209 / 8 / 251`，只作为描述性统计；当前只有 5 个
talks，未做显著性声明。

## Source of Truth

- 实现与结果分支：`luojiaxuan/rebuttal-experiments`。
- 评测 manifest：`code/rasst/manifests/rebuttal_ja_lm1_curriculum_eval.json`。
- Aries 临时训练/eval staging：
  `/mnt/aries/data6/jiaxuanluo/RASST_release_runs/ja_lm1_curriculum_20260713`。
- Hyper00 临时 xCOMET staging：
  `/data02/jaxan/RASST_rebuttal_20260710/results/xcomet_ja_lm1_curriculum_sentence_20260713`。
- 训练数据与模型的 Hugging Face destination 见 provenance；当前均为
  **pending upload**，上述共享机路径不是 canonical reusable artifact。
