# En-Ja xCOMET 强负例的 MFA-aligned term-map audit

状态：**已完成。** 本报告分析 ACL En-Ja `lm=2` 的四个强负 xCOMET case，
把 source-sentence MFA 时间边界与实际 RASST `llm_input.references` 对齐。结论是：
四例中只有 ACL 367:16 的主要语义错误可直接归因于 term-map collision；其余三例
主要是 streaming commitment / mWER sentence resegmentation，不能称为错误术语
retrieval 导致的 hallucination。

## 方法与口径

- Source-sentence 时间边界来自现有 ACL MFA 对齐。英语 source audio 对三种目标语言
  相同，因此复用
  `code/legacy/documents/code/simuleval/reports/20260525_de_lm2_rasst_bleu_case_report.tsv`
  中的 source start/end，不使用德语 hypothesis。
- Term-map 来自 ACL En-Ja `lm=2` canonical RASST runtime JSONL 中每个实际
  `llm_input.references`；runtime SHA-256 为
  `cb788310e023e88fbe572a5e88b37eae3fee54df5c4443b440b93d1ebc0bdc13`。
- `prompt_audio_end_sec=(chunk_idx+1)*1.92`。本表沿用 failure-chain 的严格定义：
  prompt audio end 不晚于 MFA sentence end 为 `on_time`，超过 sentence end 为
  `late`；紧邻 sentence start 前的 prompt 另标为 `pre_sentence`。
- xCOMET 输入 hypothesis 经过 mwerSegmenter 重新分句，而 term-map 在 source
  timeline 上更新。不能把 xCOMET 某一行的整段 hypothesis 直接视为同一 source
  sentence 时间内生成的文本；必须同时检查 raw chunk outputs。

逐 prompt 的完整 term-map 与分数见
[`ja_xcomet_mfa_term_map_cases.tsv`](ja_xcomet_mfa_term_map_cases.tsv)。

## 汇总结论

| Case | MFA source span | On-time term-map | Late / boundary-crossing term-map | 诊断 |
| --- | ---: | --- | --- | --- |
| ACL 268:83 | 565.004--570.015 s | `models→モデル`, `CRF→CRF`, `model→モデル` | `model→モデル` at 570.24 s | 未召回完整 `BiLSTM-CRF` 或 `Flair`，但 raw output 正确生成两者；xCOMET 强负主要来自邻句内容被 mWER 分到本句，不是 term noise。 |
| ACL 110:81 | 502.582--510.997 s | `training data/data/training/dataset/pretraining`，随后 `parser`、`online`、`model`、`parsing` | 句末后两个 prompt 为空 | `prefix data` 从未作为 hint 出现；开头带入上一句的 training-data 内容，但重复/续写不是错误 term-map 直接触发。 |
| ACL 117:70 | 451.399--457.716 s | `baselines`、`answer`、`question generation`、`data`、`relation` 等 | `question/parser/questions` at 458.88 s | 句内 hints 均有 source 支持；raw output 包含完整“1つ目のベースライン”。xCOMET 行丢掉句首/句尾主要是 resegmentation/boundary。 |
| ACL 367:16 | 146.157--150.739 s | `input→入力`, `sentence→文章`, `model→モデル` | `document→文章`, `sentence→文章` at 151.68 s | **明确 term-map collision：`sentence` 和 `document` 都被强制为 `文章`，直接对应输出 `文章または文章`。** 前句残尾仍是独立 boundary 问题。 |

## 逐例核对

### 1. ACL 268:83，xCOMET delta -0.8159

MFA span 为 `565.004--570.015 s`。实际 prompt：

| Audio seen | Timing | Term-map |
| ---: | --- | --- |
| 566.40 s | on time | empty |
| 568.32 s | on time | `models→モデル@0.952; CRF→CRF@0.841; model→モデル@0.807` |
| 570.24 s | late by 0.225 s | `model→モデル@0.844` |

Raw source-time output 是
`両方のモデルがBiLSTM-CRFモデルで、Flairを用いていました。`，说明虽然完整
`BiLSTM-CRF` 和 `Flair` 没有 retrieved，模型仍从语音正确生成了它们。相邻 chunks
另有 `BPE埋め込み`、`最適な結果` 和下一句 `一方は`；mWER 重分句把其中一部分
分配给当前 reference，才形成用户看到的强负 xCOMET hypothesis。不能把它描述为
term-map 凭空引入 BPE。

### 2. ACL 110:81，xCOMET delta -0.8046

MFA span 为 `502.582--510.997 s`。句初 prompt 包含上一句残留的
`training data→訓練データ`、`data→データ`、`training→訓練`、
`dataset→データセット`、`pretraining→事前訓練`；随后出现
`parser→パーサー`，句末前出现 `online→オンライン`、`model→モデル`、
`parsing→構文解析`。`prefix data/接頭辞データ` 从未作为 retrieval hint 出现。

Raw chunks 实际生成：

`下の曲線は、オフラインパーサーです。さまざまな長さのプレフィックスデータを混合して、オンラインパーサーにモデルを訓練させます。`

因此 term-map 确实显示开头存在 previous-sentence carryover，但用户看到的重复、
继续下一段和截断主要是 streaming/mWER boundary，不是错误 term translation。

### 3. ACL 117:70，xCOMET delta -0.7932

MFA span 为 `451.399--457.716 s`。在 sentence start 前 0.199 s 已有
`baselines→ベースライン`；句内依次出现 `answer→応答`、
`question generation→質問生成`、`question→質問`、`data→データ`、
`relation→リレーション` 等 source-supported hints。Raw chunks 实际生成：

`1つ目のベースラインはランダム回答と質問生成と呼ばれ、元の質問とはリレーションのないデータを追加します。`

所以 raw output 没有丢失“第一个”。xCOMET hypothesis 从 `目のベースライン`
开始且停在残片，是 mWER resegmentation 后的边界切分。句末后的
`parser→パーサー` 属于下一 source 内容，不应归因给当前句。

### 4. ACL 367:16，xCOMET delta -0.7747

MFA span 为 `146.157--150.739 s`。Term-map 的关键更新是：

| Audio seen | Timing | Term-map |
| ---: | --- | --- |
| 147.84 s | on time | `input→入力@0.963` |
| 149.76 s | on time | `sentence→文章@0.972; model→モデル@0.814; input→入力@0.801` |
| 151.68 s | late / boundary-crossing | `document→文章@0.973; sentence→文章@0.970` |

Raw output 为 `モデルへの入力は、文章または文章です。`。这里 target glossary
把两个语义不同的 source terms 都映射到 `文章`，而 reference 需要
`文またはドキュメント`。这是四例中唯一能从 term-map 到错误输出建立直接、
逐 token 对应关系的 case。输出开头的上一句残尾则仍应单独标为 boundary artifact。

## 可用于回复的保守结论

> An MFA-aligned audit of four strongly negative En-Ja xCOMET cases shows that
> only one is directly explained by a harmful term map: both `sentence` and
> `document` were mapped to `文章`, yielding `文章または文章`. In the other
> three cases, the retrieved hints were source-supported or did not contain
> the allegedly hallucinated phrase; the discrepancy arose primarily from
> streaming commitment and mWER sentence resegmentation. We therefore do not
> attribute the En-Ja xCOMET decrease wholesale to retrieval noise.

## Block-aware 回归

固定 5-sentence block 的重新评分已验证上述诊断：前三例的 paired delta 变为
`+0.0341/-0.0023/+0.1507`，而真实 collision case ACL 367:16 仍为
`-0.2776`。详细协议与 artifacts 见
[`xcomet_acl_block5_report.md`](xcomet_acl_block5_report.md)。
