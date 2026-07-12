# WMT25 prompt 的 Gemini LLM-as-a-judge 协议

## 当前状态

状态（2026-07-12 01:27 PDT）：**full Flash run 已提交；13/16 shards 已完成并下载，
Medicine En-De `lm=2/3/4` 已创建但 8,622 requests 仍全部 pending。当前 Gemini project
同时对单条标准请求返回 `429 prepayment credits depleted`，因此剩余 Batch 是否继续取决于
补充预付余额。**

已下载的 14,106 个原始回答中，14,105 个严格满足“只返回一个整数”；ACL En-Zh
`lm=4` 有 1 个回答给出解释并在末行写出分数。该末行数字没有被事后截取。一次使用完全
相同 prompt、model 与 API-default config 的 format-only retry 已尝试，但同样被余额 429
拒绝且未产生新评分；原始违例与失败尝试均保留。完整汇总仍 blocked，不能用不完整或
后验解析的数字替代。

100-request paired pilot 已完成。当前建议用 `gemini-2.5-flash` 和 API defaults
完成全量评分：在本账号上，`gemini-2.5-pro` 的 smoke request 返回 404，说明该
模型已不再向 new users 开放；可用的 `gemini-3.1-pro-preview` 不是 WMT25 论文所用
模型，而且 pilot 预计全量 Batch 费用更高。作者确认模型/预算和轮换后的私有 key
之前，不提交任何 full Batch，也不产生可用于 rebuttal 的全量结果。

正式 bundle 已在看见任何 full-run score 之前冻结为
`gemini-2.5-flash`、`generation_config={}` 和 model-default thinking。Preparation
只生成 request/sidecar/state artifacts，不调用 Gemini API，也不产生费用；随后提交与
收集严格使用该冻结配置。

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

只允许以下四个已验证 artifact，并按 method/lm 显式选择：

| Scope | Taurus staging input | Full-file SHA-256 | 选择规则 | 选中 rows |
| --- | --- | --- | --- | ---: |
| ACL 5 talks，En-Zh/De/Ja | `/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/rebuttal_2026/llm_judge_wmt25/input/xcomet_release_cache_segments.jsonl` | `43b5581bc79e8c389383e6fb84b684f4f7207334c114a0cc0b8b19d47d2a459b` | 仅 `dataset=acl_tagged_raw`；不能使用该文件中的旧 medicine rows | 11,232 |
| ESO new InfiniSST，lm1--3 | `/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/rebuttal_2026/llm_judge_wmt25/input/xcomet_new_infinisst_lm123_segments.jsonl` | `244a8fb4d5e3538ff4bb0bdbf6860a95ca4f88306ff96d7259f89dccb99fcca3` | `dataset=medicine_hardraw, method=InfiniSST, lm in {1,2,3}` | 4,311 / 8,622 |
| ESO new InfiniSST，lm4 | `/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/rebuttal_2026/llm_judge_wmt25/input/xcomet_new_infinisst_lm4_segments.jsonl` | `50708b41855173ba2231c23f567de824342da687c87a840c89320bc5f3615df9` | `dataset=medicine_hardraw, method=InfiniSST, lm=4` | 1,437 / 2,874 |
| ESO paper-exact RASST，lm1--4 | `/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/rebuttal_2026/llm_judge_wmt25/input/xcomet_paper_exact_eso_de_segments.jsonl` | `40454563ea39d20f04970b8a733dc0a370d8ab0e0a4cb25c0f7720bf5491e997` | `dataset=medicine_hardraw, method=RASST, lm in {1,2,3,4}` | 5,748 / 11,496 |

第一个文件全量有 22,728 rows，但其中 Medicine/ESO 是旧 release-cache 口径；它
只能提供 ACL 的 11,232 rows。两个 new-InfiniSST artifacts 内部还包含配对评分时使用的
release-cache RASST rows，必须按 `method=InfiniSST` 过滤。paper-exact artifact 内的旧
InfiniSST rows 也必须排除，只选 submitted-paper exact RASST。这样组成的新 ESO 矩阵
恰为 11,496 rows；任何整文件直读都会混入口径错误的 system outputs。

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
  Tier 1 `gemini-2.5-flash` 的 3.00M active enqueued-token 上限。实际提交 ACL 12
  shards + Medicine lm1 后，再创建 Medicine lm2 得到 429；远端 list audit 确认该次
  429 没有生成 Batch job。ACL 与 lm1 完成释放 quota 后，lm2--4 才安全提交。见
  [Gemini Batch rate limits](https://ai.google.dev/gemini-api/docs/rate-limits#batch-api-rate-limits)。
- Batch create 不是幂等操作。每个 shard 只允许在持久 state ledger 中从
  `PREPARED` 提交一次；网络超时后不能盲目重提，否则可能重复计费。
- API key 只能通过显式 `--api-key-file` 读取 owner-only regular file（mode
  `0600`）；不能使用环境变量，也不能把 key 值写入 Git、命令行、manifest、request
  JSONL、response、日志或文档。聊天中出现过的 key 一律视为已暴露，必须轮换。
- 本次 full run 从本机 owner-only 文件安全复制到 Taurus owner-only path
  `/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/rebuttal_2026/llm_judge_wmt25/.secrets/gemini_key_mac_20260712.txt`；
  两端均为 mode `0600`。本文只记录路径、owner/mode 与 40-byte 文件大小，不记录、
  打印、hash 或校验 key 值。

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
- 当前 full Flash bundle：
  `/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/rebuttal_2026/llm_judge_wmt25/full_flash_api_default_corrected_eso`。
  它由 Git commit `e229a4257ef1099e97b5f33d2eab17e89b9da84f` 的 runner 生成，
  `run_config_sha256` 为
  `5d01d8c2790be654272dc848aba265239a4c34838c91490b850148a76232628a`，
  `run_manifest.json` SHA-256 为
  `67d01bde0d93f748312ccdc2308d9056bfcef490c543a3f27f761769543ddad3`。
  16 request files 共 20,039,532 bytes，16 sidecars 共 30,646,531 bytes；22,728
  opaque keys 全部唯一，83 个真实空 hypothesis 保留，request payload 的
  `reference/method/glossary` 字段检查为 0。独立 pre-submit validator 核对 32 systems /
  16 pairs / 22,728 segments / 4 source artifacts，报告 SHA-256 为
  `844e1d85d49b8e55aed374ce21650430e570ffd795c9cbd09b2dfc22125ed38e`。
- 旧 bundle 使用了过时 ESO InfiniSST baseline，且 16 states 均停留在 `PREPARED`、
  没有任何 remote job name；它已重命名为
  `superseded_old_infinisst_full_flash_api_default/`，不得提交或用于结果。
- Full request/response JSONL、逐句 score、usage/cost、state ledger 与验证 manifest
  的预定 Hugging Face dataset 目标是
  [`gavinlaw/rasst-main-result-data`](https://huggingface.co/datasets/gavinlaw/rasst-main-result-data)
  下的 versioned rebuttal artifact，状态为 **pending**。
- Git 只保存本协议、可复现代码、轻量 summary/validation 和 HF revision 链接。
  上传完成并在 Git 中记录 revision 前，Taurus staging 不能视为 source of truth。
