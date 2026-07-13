# ACL xCOMET streaming boundary 修复

状态：**已完成并独立验证。** 原 sentence-level 流程把整场 streaming prediction
对逐句 target reference 运行 `mwerSegmenter`，再按行号与固定 source/reference
配对。输出有 delayed commitment、漏句或跨句续写时，mWER 可能把邻句内容移入
当前 hypothesis，导致 xCOMET 实际看到不对应的 `src/mt/ref`。

## 修复

`score_sentence_aligned_xcomet.py` 新增显式参数
`--sentences-per-segment`。本次设为 `5`：每个 talk 内先把连续五个
source/reference 句组成固定、无重叠 block，再对完整 prediction 只求这些 block
边界。这样相邻句之间的 streaming/mWER 漂移留在同一个 xCOMET pair 内；不修改、
删除或按 reference 重写任何系统输出。旧逐句行为仍可用显式值 `1` 复现。

- ACL：24 systems，12 paired cells，5 talks/system。
- 每个 system 96 blocks，共 2,304 blocks。
- Metric：`Unbabel/XCOMET-XXL`，revision
  `873bac1b1c461e410c4a6e379f6790d3d1c7c214`。
- 输入长度审计（XLM-R tokenizer，special tokens included）：source 最大 200、
  hypothesis 最大 275、reference 最大 215 tokens；没有字段超过 512 tokens。
- 独立 validator：24 systems / 12 pairs / 2,304 blocks，状态 `ok`。

## 结果

xCOMET 乘以 100；每种语言对四个 `lm` cells 等权平均。

| Language | RASST | InfiniSST | Delta | Block W/T/L |
|---|---:|---:|---:|---:|
| En-Zh | 57.9680 | 58.3042 | -0.3362 | 176/2/206 |
| En-De | 63.3204 | 64.3765 | -1.0561 | 176/2/206 |
| En-Ja | 52.2048 | 53.4221 | -1.2174 | 183/0/201 |
| **Macro (12 cells)** | **57.8310** | **58.7009** | **-0.8699** | — |

逐 cell 数据见
[`xcomet_acl_block5_paired.tsv`](xcomet_acl_block5_paired.tsv)。这些 block scores
与旧 sentence-level xCOMET 的绝对值不可直接比较；二者评测粒度不同。修复后的
结果不能支持“RASST overall xCOMET 提升”的 rebuttal claim。

## 四个 Ja 强负例回归

| Case | Old sentence delta | New 5-sentence block delta | 结论 |
|---|---:|---:|---|
| ACL 268:83 | -0.8159 | +0.0341 | 邻句串位被吸收，原强负不是该句的真实翻译失败 |
| ACL 110:81 | -0.8046 | -0.0023 | 原重复/截断强负基本消失 |
| ACL 117:70 | -0.7932 | +0.1507 | 完整 `1つ目のベースライン` 回到对应上下文 |
| ACL 367:16 | -0.7747 | -0.2776 | 仍为真实失败；`sentence/document→文章/文章` collision 保留 |

这证明逐句 mWER 确实污染了 case-level failure analysis，但它不是 En-Ja aggregate
下降的主要解释：Ja `lm=2` 的 block-aware delta 仍为 `-3.0956` points。因此不要用
前三个旧 case 论证 retrieval noise，也不要声称修复边界后 xCOMET 变成 overall win。
机器可读回归表见
[`xcomet_acl_block5_case_audit.tsv`](xcomet_acl_block5_case_audit.tsv)。

## Provenance 与 artifact 状态

| Artifact | SHA-256 |
|---|---|
| `xcomet_acl_block5_summary.tsv` | `43bc268f9c9a197f2b7f24896cbd7bf9f6961fc0b463d8770893aeb926ecaddf` |
| `xcomet_acl_block5_paired.tsv` | `a148e55a51e114e3c3bd0b5b61f675bcbb36f03e83340a16f0211499b41e3db5` |
| full `segments.jsonl` | `21ccef92d1721dc66631b8d604d6a50a4dd05a31166dcc6371a082c2964a385a` |
| `xcomet_acl_block5_validation.json` | `44d754d1526ef6872b19626619b259c48278cac3a190edd003f9e85a5b3e57f5` |

- Code：`luojiaxuan/rebuttal-experiments@285ebd1`。
- Hyper00 staging：
  `/data02/jaxan/RASST_rebuttal_20260710/results/xcomet_acl_block5_20260713/`。
- Taurus backup staging：
  `/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/rebuttal_2026/xcomet_acl_block5_20260713/`；
  `segments.jsonl` 已复核为 2,304 rows，SHA-256 与 Hyper00 相同。
- Run container：`sglang-omni-jaxan-07130915`；physical GPUs 0/3；稳定推理
  10 秒窗口平均 utilization 98%--100%；任务完成后容器已停止。
- 完整 2,304-row JSONL 的预定 Hugging Face 目标为
  `gavinlaw/rasst-main-result-data` 下的 versioned rebuttal artifact。当前没有可用
  写入凭据，状态为 **pending upload**；Hyper00 staging 不是 canonical artifact。

## Rebuttal 使用建议

这是一个有价值的评测修复和内部诊断，但不是明显 win。建议在 rebuttal 中：

1. 不再引用旧 sentence-level ACL `+0.1158` 作为 overall xCOMET improvement；
2. 不展开 block-aware `-0.8699`，除非 reviewer 明确追问 alignment protocol；
3. 主要使用 verified masked BLEU 说明移除术语 token 后普通翻译质量没有下降；
4. 若讨论 failure cases，只保留 ACL 367:16 为 confirmed term-map failure，并把另外
   三例标为 sentence-boundary artifacts。
