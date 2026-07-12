# ESO En-De paper-exact xCOMET-XXL 复算

## 结论

此前 rebuttal xCOMET 汇总对 ESO En-De 使用了 release-canonical
`30/30, 30/30, 20/20, 20/20` cache 输出，而不是 submitted-paper exact 输出。
本次从 Taurus 的 InfiniSST provenance 与历史运行目录恢复论文四个 RASST
`instances.log`，并对实际 BLEU 使用的 `instances.strip_term.log` 重新评分。

paper-exact ESO En-De 的 RASST cell-macro xCOMET 为 **75.9398**，InfiniSST 为
**78.0410**，平均差值为 **-2.1012**，四个 cells 均为负。旧 release-cache
口径的差值为 `-3.1716`。修正使平均差值改善 `+1.0703`，主要来自 lm4；负向
结论仍然成立，但不应继续把 `-3.1716` 写成 submitted-paper exact 结果。

| lm | InfiniSST | RASST paper exact | Paper-exact delta | 旧 release-cache delta | Delta 修正量 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 76.4411 | 70.8779 | -5.5631 | -5.5631 | +0.0000 |
| 2 | 78.7102 | 77.5068 | -1.2034 | -1.2034 | +0.0000 |
| 3 | 78.3695 | 76.9881 | -1.3814 | -1.5909 | +0.2095 |
| 4 | 78.6433 | 78.3863 | -0.2571 | -4.3289 | +4.0718 |
| **平均** | **78.0410** | **75.9398** | **-2.1012** | **-3.1716** | **+1.0703** |

ACL 12 cells 不变。将其与 paper-exact ESO En-De 合并后，保留的 16-cell macro
为 RASST `78.0131`、InfiniSST `78.4515`、差值 `-0.4385`，8/16 cells 为正；旧
release-cache 合并差值为 `-0.7060`。

`xcomet_paper_exact_combined_{summary,paired}.tsv` 是将原运行中独立验证的 ACL
24 systems 与本次独立验证的 ESO 8 systems 合并得到的轻量视图。两个运行使用
同一模型、checkpoint、segmenter 和评分代码，但物理 GPU ids 不同，因此保留各自
的 `scoring_config_sha256`；combined TSV 本身不冒充一次单独的 32-system validator
运行。

## 输入核对

四个 paper-exact cells 均为 cache `30/30`。lm1、lm2、lm4 是 serial 输出；lm3
是论文选择的 batch-vLLM 输出。恢复后的 BLEU 分别为 `22.6187`、`26.7696`、
`26.9742`、`28.9154`，与论文主表 provenance 完全一致。原始路径、event ID、
BLEU、TERM_ACC，以及 `eval_results.tsv`、`instances.log`、
`instances.strip_term.log` 的 SHA-256 见
[`xcomet_paper_exact_eso_de_input_provenance.tsv`](xcomet_paper_exact_eso_de_input_provenance.tsv)。

xCOMET 对 `instances.strip_term.log` 评分，因为它是论文 BLEU/TERM_ACC 评测实际
使用的去标签 hypothesis；原始 `instances.log` 仅保留为 provenance。

## 评分与验证

- Model：`Unbabel/XCOMET-XXL` revision
  `873bac1b1c461e410c4a6e379f6790d3d1c7c214`。
- Checkpoint SHA-256：
  `e760e1f568af69b7a1bf7aeb46d8f3be21f01be7cbda480f8225ee81eb0af27a`。
- Scorer SHA-256：
  `53557f2bf9c538202b703437aedf6b04715b99e7ddb2c72500bfa6b6d547bf73`。
- Host/container：Hyper00 `node-radixark-16-0000`，
  `sglang-omni-jaxan-07120847`。
- Physical GPUs：2、3；2-way DDP，batch size 16，`num_workers=0`。
- 稳定推理阶段连续 10 秒窗口的双卡利用率为 99%--100%。
- 评分规模：8 systems、4 strict pairs、11,496 sentence segments。
- 独立 validator 结果：8 systems、4 pairs、11,496 segments，全部与逐句 JSONL
  重算一致。

Git-tracked 轻量产物：

| Artifact | SHA-256 |
| --- | --- |
| [`xcomet_paper_exact_combined_summary.tsv`](xcomet_paper_exact_combined_summary.tsv) | `e84a55a2e779c59f9e63e2fb33ec2c699b2bfb125828876f8d66e2ba6069142d` |
| [`xcomet_paper_exact_combined_paired.tsv`](xcomet_paper_exact_combined_paired.tsv) | `3c0dbdb2f46f115fcc4ea1551075e302834026ee07bb0ea7a02377955df4df21` |
| [`xcomet_paper_exact_eso_de_summary.tsv`](xcomet_paper_exact_eso_de_summary.tsv) | `d39004098ef158430d80f0389475eaa114f919f5c9d28b043cd2ec28eaea3279` |
| [`xcomet_paper_exact_eso_de_paired.tsv`](xcomet_paper_exact_eso_de_paired.tsv) | `c19292c432dea966e631f36366370797ea9c60305c95c962a4e4bf20ebb00e8d` |
| [`xcomet_paper_exact_eso_de_validation.json`](xcomet_paper_exact_eso_de_validation.json) | `3f1fdc38a940f915202e59611ca8d983e9ae102fb63a4fc2085fdf09af4be3c2` |
| [`xcomet_paper_exact_eso_de_manifest.inference.tsv`](xcomet_paper_exact_eso_de_manifest.inference.tsv) | `6d3dab49c397142c0d094effadc7935d91377bd333f0975157be177a55bc75ff` |
| [`xcomet_paper_exact_eso_de_manifest.portable.tsv`](xcomet_paper_exact_eso_de_manifest.portable.tsv) | `220f31ef2909b4d27b92bf9a54111d8aa3aef75254860ace2803ba1fbdc27415` |

推理 manifest 与 corrected portable manifest 只在 lm3 的 content-addressed 目录名
上不同。首次 Taurus→Hyper00 NFS 传输尚未关闭时曾记录到中间态 hash
`86e6670...`；scorer 实际读取、summary/segments 记录、Taurus 源文件复核和 corrected
portable bundle 均为最终 hash
`f7bc53e9ffd5a4275dd5f027cb76340a9015786296797be8bd3d8423cbc1ae29`。
因此该差异不改变评分输入或结果；保留 inference manifest 是为了不改写运行历史。

## Local staging 与上传状态

逐句结果位于 Hyper00：

```text
/data02/jaxan/RASST_rebuttal_20260710/results/xcomet_paper_exact_eso_de_20260712/segments.jsonl
```

文件大小约 28 MiB，SHA-256 为
`40454563ea39d20f04970b8a733dc0a370d8ab0e0a4cb25c0f7720bf5491e997`。
portable bundle 位于：

```text
/data02/jaxan/RASST_rebuttal_20260710/xcomet_paper_exact_eso_de_bundle/
```

两者预定上传到 `gavinlaw/rasst-main-result-data` 的 versioned rebuttal artifact。
当前无可用 Hugging Face 写入凭据，状态为 **pending upload**；Hyper00 路径只是
staging，不是 canonical source of truth。
