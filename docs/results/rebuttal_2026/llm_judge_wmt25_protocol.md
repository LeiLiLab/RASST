# WMT25 prompt 的 Gemini LLM-as-a-judge 协议

## 当前状态

状态：**full Batch 尚未提交，等待作者确认约 50 USD 预算并轮换 Gemini key**。

100-request paired pilot 已完成。当前建议用 `gemini-2.5-flash` 和 API defaults
完成全量评分：在本账号上，`gemini-2.5-pro` 的 smoke request 返回 404，说明该
模型已不再向 new users 开放；可用的 `gemini-3.1-pro-preview` 不是 WMT25 论文所用
模型，而且 pilot 预计全量 Batch 费用更高。作者确认模型/预算和轮换后的私有 key
之前，不提交任何 full Batch，也不产生可用于 rebuttal 的全量结果。

## Prompt

逐句评分复用 WMT25 shared task paper Appendix A 的 Task 1 prompt。花括号只替换为
语言名称、source segment 和 system translation；不加 system prompt、reference、
glossary、术语标签或额外评分说明。

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

- Source：[WMT25 shared-task paper](https://aclanthology.org/2025.wmt-1.24.pdf)，
  Appendix A, “LLM Prompt for Task 1”。
- `source_lang` 固定为 `English`；`zh/de/ja` 分别展开为
  `Chinese/German/Japanese`。
- 上述 UTF-8 template（末尾无换行）的 SHA-256 为
  `56c396ed097093f51c8febc748c8862f1866ca5c83516ef74f6667c5d682e859`。
- 只接受去除首尾空白后严格匹配 `0`--`100` 的一个整数；其他 response 必须显式
  记为失败，不能截取、四舍五入或静默修复。

## Model 与 generation 口径

WMT25 Task 1 的 Gemini judge 使用 `gemini-2.5-pro`。因此，若该模型可用，它才是
最接近论文的 model reproduction；仅复用 prompt 并不等于复现论文中的 Gemini
system。当前账号实测 `gemini-2.5-pro` 返回 404，不能作为可执行方案。

当前可执行方案的口径如下：

| 方案 | 与 WMT25 的关系 | 100-request pilot / 全量 Batch 费用投影 | 当前结论 |
| --- | --- | --- | --- |
| `gemini-2.5-pro` | WMT25 paper 对应模型 | 本账号 1-request smoke 即 404 | 不可用 |
| `gemini-2.5-flash` | 相同 prompt，但不是相同模型 | 约 `$48.19`，bootstrap 95% 区间 `$43.72–$52.93` | **当前建议** |
| `gemini-3.1-pro-preview` | 可用的 Pro proxy；不是 WMT25 reproduction | 约 `$109.06`，bootstrap 95% 区间 `$101.95–$116.08` | 仅作更昂贵备选 |

两种可用模型的 100-request paired pilot score Pearson 为 `0.9241`，但 mean absolute
difference 为 `6.580`，所以不能把 Flash 和 Pro-preview 分数混在同一主表，也不能
把 Flash 结果写成 WMT25 的 Gemini 2.5 Pro 复现。若采用 Flash，论文和 rebuttal
应明确写成“WMT25 prompt with Gemini 2.5 Flash”。

Pilot manifest 固定 `generation_config={}`：不显式设置 `temperature=0`、
`max_output_tokens`、`seed`、response schema 或 `thinking_budget`，而是保留 Gemini
API defaults 和 model-default thinking。这一点与早期拟定的 `temperature=0` 方案
不同；full run 必须沿用实际 pilot 的空 generation config，避免在 pilot 后改变协议。
visible answer 只取非-thinking text；usage 中的 thinking tokens 仍需完整保存并计费。

## 输入与 reference-free 边界

Judge request 只发送 `source` 与 `hypothesis`，不会发送 `reference`、xCOMET score
或术语表，因此模型评分本身是 **reference-free**。但逐句 `source/hypothesis` 来自
现有 sentence-aligned xCOMET artifacts；streaming hypothesis 的句界由
`mwerSegmenter` 借助人工 reference 对齐。因此更准确的表述是：judge inference
reference-free，但 segment construction / evaluation unit 仍间接依赖 human
reference。逐句记录中的 reference hash 只用于 alignment provenance 和配对核验，
绝不进入 prompt。

只允许以下两个已验证 artifact：

| Scope | Taurus staging input | Full-file SHA-256 | 选择规则 | 选中 rows |
| --- | --- | --- | --- | ---: |
| ACL 5 talks，En-Zh/De/Ja | `/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/rebuttal_2026/llm_judge_wmt25/input/xcomet_release_cache_segments.jsonl` | `43b5581bc79e8c389383e6fb84b684f4f7207334c114a0cc0b8b19d47d2a459b` | 仅 `dataset=acl_tagged_raw`；不能使用该文件中的旧 medicine rows | 11,232 |
| ESO/Medicine 5 talks，En-De | `/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/rebuttal_2026/llm_judge_wmt25/input/xcomet_paper_exact_eso_de_segments.jsonl` | `40454563ea39d20f04970b8a733dc0a370d8ab0e0a4cb25c0f7720bf5491e997` | 仅 `dataset=medicine_hardraw`，使用 submitted-paper exact 四档 `30/30` replacement | 11,496 |

第一个文件全量有 22,728 rows，但其中 Medicine/ESO 是旧 release-cache 口径；它
只能提供 ACL 的 11,232 rows。ESO/Medicine 必须从第二个 paper-exact artifact
取 11,496 rows。直接对第一个文件全量评分会让 ESO lm3/lm4 回到错误输入。

## 评分矩阵与 Batch 分片

| Track | Pair cells / shards | 每 shard requests | System rows | Requests |
| --- | ---: | ---: | ---: | ---: |
| ACL：3 languages × 4 LM | 12 | 936（468 segments × 2 systems） | 24 | 11,232 |
| ESO/Medicine：De × 4 LM | 4 | 2,874（1,437 segments × 2 systems） | 8 | 11,496 |
| **合计** | **16** | — | **32** | **22,728** |

每个 shard 对应一个严格 pair `(dataset, language, lm)`，同时包含 RASST 与
InfiniSST；共 16 shards、16 strict pairs。不得跨 system 去重相同文本。聚合输出
至少包含 32-row system summary、16-row paired delta、talk-macro 与逐句 paired
win/tie/loss；缺任一 shard、重复 request key 或非整数 response 时禁止发布汇总。

## Batch、费用与凭据安全

- Gemini Batch API 需要 paid tier；官方标称相对 standard API 价格减半，但实际
  费用取决于 input、visible output 和 thinking tokens。此次 Flash 的约 `$48.19`
  只是基于 100 requests 的分层 paired pilot 投影，不是费用上限。
- Pilot 中 Flash 平均每 request 约 `158.54` input tokens、`2.02` visible output
  tokens 和 `1,680.43` thinking tokens；默认 thinking 是主要成本来源。提交前必须
  由作者明确接受约 50 USD 预算。
- Pilot 对全量 input 的 cell-weighted 投影约为 3.60M tokens，高于官方当前
  Tier 1 `gemini-2.5-flash` 的 3.00M active enqueued-token 上限。实际账号 quota 需在
  AI Studio 再核对；若仍是 Tier 1，应分两波提交（先 ACL 12 shards 加 Medicine
  `lm=1/2`，完成并释放 quota 后再提交 Medicine `lm=3/4`），不能一次 enqueue 全部
  16 shards。见 [Gemini Batch rate limits](https://ai.google.dev/gemini-api/docs/rate-limits#batch-api-rate-limits)。
- Batch create 不是幂等操作。每个 shard 只允许在持久 state ledger 中从
  `PREPARED` 提交一次；网络超时后不能盲目重提，否则可能重复计费。
- API key 只能通过显式 `--api-key-file` 读取 owner-only regular file（mode
  `0600`）；不能使用环境变量，也不能把 key 值写入 Git、命令行、manifest、request
  JSONL、response、日志或文档。聊天中出现过的 key 一律视为已暴露，必须轮换。
- Taurus 当前 candidate path 为
  `/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/rebuttal_2026/llm_judge_wmt25/.secrets/gemini_key.txt`；
  文件存在和权限正确不等于 key 已安全轮换或获准用于付费 full run。作者确认前保持
  blocked，本文不记录也不校验 key 值。

## Runtime 与 Source of Truth

- Taurus staging root：
  `/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/rebuttal_2026/llm_judge_wmt25`。
- Pinned runtime：
  `/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/rebuttal_2026/llm_judge_wmt25/.venv-google-genai-2.11.0`
  （`google-genai==2.11.0`）。本目录下的 `.venv` 和所有 run products 都是本地
  staging，不是 canonical artifact。
- 100-request pilot manifest：`pilot_100/manifest.json`，SHA-256
  `417c4866239279d68dc2371c2353fc8842cecd15d7ec53a0899ab88487f1ce47`；其 request
  sample 为确定性的 proportional paired stratification，不能当作全量质量结果。
- Full request/response JSONL、逐句 score、usage/cost、state ledger 与验证 manifest
  的预定 Hugging Face dataset 目标是
  [`gavinlaw/rasst-main-result-data`](https://huggingface.co/datasets/gavinlaw/rasst-main-result-data)
  下的 versioned rebuttal artifact，状态为 **pending**。
- Git 只保存本协议、可复现代码、轻量 summary/validation 和 HF revision 链接。
  上传完成并在 Git 中记录 revision 前，Taurus staging 不能视为 source of truth。
