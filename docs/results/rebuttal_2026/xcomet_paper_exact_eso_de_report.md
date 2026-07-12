# ESO En-De paper-exact xCOMET-XXL 复算

## 结论

最初 rebuttal xCOMET 汇总同时使用了旧 InfiniSST baseline 和 release-canonical
`30/30, 30/30, 20/20, 20/20` RASST cache 输出。后续先恢复了 submitted-paper
exact RASST 四档 `30/30` 输出，又重新生成并独立验证了 InfiniSST baseline。

当前推荐口径是 **new InfiniSST 对比 paper-exact RASST**：RASST cell-macro
xCOMET 为 **75.9398**，InfiniSST 为 **77.3245**，平均差值为 **-1.3848**，
四个 cells 均为负。lm4 基本持平（`-0.2287`）；主要负差来自 lm1（`-2.5040`）
与 lm3（`-1.8294`）。

| lm | New InfiniSST | RASST paper exact | RASST - InfiniSST |
| ---: | ---: | ---: | ---: |
| 1 | 73.3819 | 70.8779 | -2.5040 |
| 2 | 78.4838 | 77.5068 | -0.9770 |
| 3 | 78.8175 | 76.9881 | -1.8294 |
| 4 | 78.6150 | 78.3863 | -0.2287 |
| **平均** | **77.3245** | **75.9398** | **-1.3848** |

三次口径修正应明确区分：

| Comparison | ESO En-De mean delta |
| --- | ---: |
| Old InfiniSST vs release-cache RASST | -3.1716 |
| Old InfiniSST vs paper-exact RASST | -2.1012 |
| **New InfiniSST vs paper-exact RASST（推荐）** | **-1.3848** |

ACL 12 cells 不变。与推荐 ESO 结果合并后，16-cell macro 为 RASST `78.0131`、
InfiniSST `78.2724`、差值 `-0.2593`，8/16 cells 为正。

[`xcomet_new_infinisst_vs_paper_exact_rasst.tsv`](xcomet_new_infinisst_vs_paper_exact_rasst.tsv)
由各自独立验证的 system means 交叉组合。两边使用同一 xCOMET model revision、
checkpoint、segmenter 和 scorer，但不是同一次联合 scorer run，因此只报告 system
means 与 cell-macro delta，不从该交叉表声称 combined sentence-level
win/tie/loss。`xcomet_paper_exact_combined_{summary,paired}.tsv` 保留为旧 InfiniSST
baseline 的历史视图，不再是推荐结果。

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
- Paper-exact RASST 来源运行：8 systems、4 pairs、11,496 segments；独立
  validator 全部重算一致。
- New InfiniSST lm1--3 来源运行：6 systems、3 pairs、8,622 segments；exit 0，
  validator `status=ok`。
- New InfiniSST lm4 来源运行：2 systems、1 pair、2,874 segments；exit 0，
  validator `status=ok`。
- New InfiniSST 来源运行内部同时评分了 release-cache RASST；推荐交叉表只取其中
  InfiniSST system rows，再与独立验证的 paper-exact RASST system rows 组合。

Git-tracked 轻量产物：

| Artifact | SHA-256 |
| --- | --- |
| [`xcomet_new_infinisst_vs_paper_exact_rasst.tsv`](xcomet_new_infinisst_vs_paper_exact_rasst.tsv) | `94e88b680c104697900df961b18a047d876e9b445a63894c3f2af5628073e0e6` |
| [`xcomet_new_infinisst_lm123_summary.tsv`](xcomet_new_infinisst_lm123_summary.tsv) | `14d3dea52d9243d4a730e7bd22a7455492be10f7469c910d20b6eed1d5f8b61e` |
| [`xcomet_new_infinisst_lm123_manifest.tsv`](xcomet_new_infinisst_lm123_manifest.tsv) | `ead6cf9be73be8ed3c6a27a6b10c38a013ec84133f5f67f49d081455bfda428e` |
| [`xcomet_new_infinisst_lm123_validation.json`](xcomet_new_infinisst_lm123_validation.json) | `b7c62d5a6a9449079ca8b0c36800e5fd96ac405c8fa287ff9f18be009b5bb561` |
| [`xcomet_new_infinisst_lm4_summary.tsv`](xcomet_new_infinisst_lm4_summary.tsv) | `42f57b77feb658c81747fdc874f9986ae76a2e1c01f0399b4f580db3a0c90087` |
| [`xcomet_new_infinisst_lm4_manifest.tsv`](xcomet_new_infinisst_lm4_manifest.tsv) | `59cb5e26eda0b5c74838cd1a52b5e00db4352dabe8e7f10db379f2d28e8b701c` |
| [`xcomet_new_infinisst_lm4_validation.json`](xcomet_new_infinisst_lm4_validation.json) | `86b522fa35a093febe78251b4000bae260a62ab7a6473cd210349f3ad19e08a2` |
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
/data02/jaxan/RASST_rebuttal_20260710/results/xcomet_eso_de_infinisst_rerun_lm123_20260712/segments.jsonl
/data02/jaxan/RASST_rebuttal_20260710/results/xcomet_eso_de_infinisst_rerun_lm4_20260712/segments.jsonl
```

paper-exact RASST 来源文件大小约 28 MiB，SHA-256 为
`40454563ea39d20f04970b8a733dc0a370d8ab0e0a4cb25c0f7720bf5491e997`；new
InfiniSST lm1--3 与 lm4 逐句文件的 SHA-256 分别为
`244a8fb4d5e3538ff4bb0bdbf6860a95ca4f88306ff96d7259f89dccb99fcca3` 与
`50708b41855173ba2231c23f567de824342da687c87a840c89320bc5f3615df9`。
paper-exact portable bundle 位于：

```text
/data02/jaxan/RASST_rebuttal_20260710/xcomet_paper_exact_eso_de_bundle/
```

两者预定上传到 `gavinlaw/rasst-main-result-data` 的 versioned rebuttal artifact。
当前无可用 Hugging Face 写入凭据，状态为 **pending upload**；Hyper00 路径只是
staging，不是 canonical source of truth。
