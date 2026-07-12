# Gemini LLM-as-a-judge 100-request pilot

状态：**pilot 已完成；完整 22,728-request run 尚未提交。**

## 结论

- 完整矩阵已经包含 RASST 和 InfiniSST baseline，不需要再把账单乘以 2：ACL
  11,232 requests，Medicine En-De 11,496 requests，合计 22,728。
- `gemini-2.5-pro` 对当前 Gemini API 账号返回 `404 NOT_FOUND`，说明该模型不再向
  new users 开放。因此当前不能 exact reproduce WMT25 的 Gemini 2.5 Pro judge。
- 本 pilot 使用当前可调用的 `gemini-3.1-pro-preview` 作为明确标注的 Pro proxy，
  并与 `gemini-2.5-flash` 比较。两者均使用 WMT25 Appendix A 原始 prompt、空
  generation config 和 model-default thinking；不能把 Pro 3.1 proxy 写成 WMT25
  Gemini 2.5 Pro reproduction。
- 按当前官方价格和 16 cells 的真实 request 数加权，完整 Batch API 预计费用为：
  - `gemini-3.1-pro-preview`：`$109.06`，分层 sample-bootstrap 95% 区间
    `$101.95–$116.08`；普通 API 约 `$218.12`。
  - `gemini-2.5-flash`：`$48.19`，分层 sample-bootstrap 95% 区间
    `$43.72–$52.93`；普通 API 约 `$96.39`。
  - 若两个模型都跑完整 Batch，点估计合计 `$157.25`。
- 100 条上两模型评分均值非常接近（Pro `49.02`、Flash `48.74`），Pearson
  `0.9241`，mean absolute difference `6.58`，exact agreement `65/100`。Pro higher /
  tie / Flash higher 为 `14/65/21`。这只表示模型间 agreement，不是相对人工评分的
  accuracy，也不能用于挑选对 RASST 更有利的 judge。

## Protocol

评测 prompt 逐字复用 [WMT25 Task 1 Appendix A](https://aclanthology.org/2025.wmt-1.24.pdf)：

```text
Score the following translation from {source_lang} to
{target_lang} on a scale from 0 to 100, where a score of 0 means a
broken or poor translation; 33 indicates a flawed translation
with significant issues; 66 indicates a good translation with
only minor issues in grammar, fluency, or consistency; and 100
represents a perfect translation in both meaning and grammar.
Answer with only a whole number representing the score, and
nothing else.
{source_lang} source text:
{source_seg}
{target_lang} translation:
{target_seg}
```

- Source language name 固定为 `English`；target names 为 `Chinese`、`German`、
  `Japanese`。
- Prompt 不包含 reference、method name、LM setting、glossary 或 term tags。
- 该 judge 是 reference-free；Medicine 只纳入 En-De 是为了与 human-reference
  sentence alignment 和 revision 主结果范围一致，不是因为 Gemini 读取 reference。
- 100 个 system outputs 来自 50 个 paired source segments，每个同时包含 RASST 和
  InfiniSST。按完整矩阵 cell 大小做 deterministic proportional stratification：ACL
  24 pairs，Medicine 26 pairs。
- 正式费用使用 [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
  2026-07-12 的价格：Pro 3.1 Preview Batch input/output 为 `$1/$6` per million；
  Flash 2.5 Batch input/output 为 `$0.15/$1.25` per million。Output 价格包含 thinking
  tokens。

## Token usage 与费用

| Model | Requests | Mean input | Mean visible output | Mean thinking | Pilot standard cost | Full Batch projection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemini-3.1-pro-preview` | 100 | 158.54 | 1.95 | 771.14 | `$0.9594` | `$109.06` (`$101.95–$116.08`) |
| `gemini-2.5-flash` | 100 | 158.54 | 2.02 | 1,680.43 | `$0.4254` | `$48.19` (`$43.72–$52.93`) |

Flash 单价较低，但默认 thinking tokens 在本 sample 中约为 Pro proxy 的 2.18 倍；
因此不能只按 input tokens 或可见整数答案估算账单。区间是固定 16-cell 分层 sample
bootstrap，反映 100-request sample 的不确定性，不包含未来 API 价格、model alias 或
模型随机性变化。

## Provenance 与 staging

- Runner：
  `code/rasst/analysis/rebuttal/run_gemini_llm_judge_pilot.py`
- Pilot request SHA-256：
  `85a13ab3e6b8aa831378af88884915034c3adb96dcf524bdbcec3123e669bef8`
- Manifest SHA-256：
  `417c4866239279d68dc2371c2353fc8842cecd15d7ec53a0899ab88487f1ce47`
- Gemini 2.5 Pro unavailable smoke-result SHA-256：
  `47f69bb2fa9b0a87b8a4183282c82a31229a2f603209835199efb86612b81e2d`
- Pro raw result SHA-256：
  `83398a28265017175bca817f2707e1ade5dc5ef84101b067bdd5a2f2dbfa0bf9`
- Flash raw result SHA-256：
  `0552d61e690206963f291895df67da2166ebf96ae5be8b273c1f12e7bcd2af99`
- Machine-readable report SHA-256：
  `851847a1a298bd25c45e7fe065ec41df5fa1e36fcdf4ccdd1215fa01ab12d6e3`
- Taurus staging：
  `/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/rebuttal_2026/llm_judge_wmt25/pilot_100`
- Intended Hugging Face destination：`gavinlaw/rasst-main-result-data` 下的 versioned
  rebuttal artifact；当前 upload status 为 **pending**。在上传完成前，Taurus
  staging 不是 canonical reusable artifact。

输入严格合并两套已验证逐句数据：旧 full-matrix 文件只选 ACL 11,232 rows；
Medicine 使用 submitted-paper exact En-De 11,496-row 文件。没有使用旧文件中的
release-cache Medicine rows。

## Decision guardrail

正式实验应在查看完整分数前只选择一个 judge 配置并冻结。若目标是性价比，当前
证据支持 `gemini-2.5-flash` Batch，预算建议 `$55–$60`；若目标是更强的 current Pro
proxy，预算建议 `$120`。不能同时跑完两个模型后只报告对 RASST 更有利的一个。
