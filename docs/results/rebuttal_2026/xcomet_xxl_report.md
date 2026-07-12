# xCOMET-XXL rebuttal 评测报告

> 口径说明（2026-07-11）：本报告中的 ESO En-De RASST 四行使用
> release-canonical `30/30, 30/30, 20/20, 20/20` cache 输出，不是
> submitted-paper exact 四档 `30/30` 输出。paper-exact 复算将 ESO 平均差值从
> `-3.1716` 修正为 `-2.1012`；rebuttal 应引用
> [`xcomet_paper_exact_eso_de_report.md`](xcomet_paper_exact_eso_de_report.md)。
> 本报告保留为 release-cache 诊断与首次全矩阵运行 provenance。

## 结论

本次评测使用 `Unbabel/XCOMET-XXL` revision
`873bac1b1c461e410c4a6e379f6790d3d1c7c214`，覆盖 ACL 60/60
En-Zh/De/Ja 与 ESO En-De 的 RASST/InfiniSST、4 个 latency settings。总计
32 个 system rows、16 个严格配对 cells、22,728 个已评分 hypothesis-reference
segments。

下表按 cell 等权平均，xCOMET 乘以 100 便于阅读；没有做显著性声明。

| 分组 | Cells | RASST | InfiniSST | Delta | 正/负/零 cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACL En-Zh | 4 | 82.4860 | 81.8504 | +0.6356 | 4/0/0 |
| ACL En-De | 4 | 82.8855 | 82.8805 | +0.0050 | 2/2/0 |
| ACL En-Ja | 4 | 70.7410 | 71.0342 | -0.2933 | 2/2/0 |
| **ACL 合计** | **12** | **78.7042** | **78.5884** | **+0.1158** | **8/4/0** |
| **ESO En-De** | **4** | **74.8694** | **78.0410** | **-3.1716** | **0/4/0** |
| 16 cells 合计 | 16 | 77.7455 | 78.4515 | -0.7060 | 8/8/0 |

结果是 mixed，而不是“xCOMET 也普遍提升”：ACL 的 cell-macro 平均差值很小，
正向结果主要来自 En-Zh；En-De 几乎持平，En-Ja 为负。ESO En-De 四个 cells
全部为负且平均差值较大。因此 rebuttal 应把主张收窄为 terminology handling
提升，并明确 overall contextual translation quality 依赖数据集和语言，不能用本表
声称稳定或显著提升。

## 可审计产物

- 32-system summary：[`xcomet_xxl_summary.tsv`](xcomet_xxl_summary.tsv)，
  SHA-256 `4d952042599d387ddfc3fbc844dd2ebe75734a3fa3fe2e5dfaa4e34c187402de`。
- 16-pair table：[`xcomet_xxl_paired.tsv`](xcomet_xxl_paired.tsv)，
  SHA-256 `a83db3684d7aaf4efeabc6141d5dc0b96af7c1c1e3ad605effbb04a585438324`。
- 独立验证报告：[`xcomet_xxl_validation.json`](xcomet_xxl_validation.json)，
  SHA-256 `72baf201cf12a984a38deea03a65f4ff03184ee0e23c733d86e249c6aad39608`。
- 逐句 JSONL：
  `/data02/jaxan/RASST_rebuttal_20260710/results/xcomet_xxl_recovered_attempt2/segments.jsonl`，
  65,900,597 bytes、22,728 rows，SHA-256
  `43b5581bc79e8c389383e6fb84b684f4f7207334c114a0cc0b8b19d47d2a459b`。
- 成功 recovery log：同目录 `run.log`，39 rows，SHA-256
  `e5a3b9f6e01b53f99b6dd61e4ee208ab7ca0ea8aef433ba3e70d81301840a7cf`；
  exit code 为 0，包含三条 `[DONE]`，不含 `Traceback` 或 `[ERROR]`。

独立 validator 从逐句 JSONL 重新计算每个 system 的 segment mean、乘 100
分数、talk-macro mean，以及每个严格配对的 RASST/InfiniSST mean、delta、sample
standard deviation 和 win/tie/loss。验证结果为 32 systems、16 pairs、22,728
segments，全部与 TSV 一致。逐句 JSONL 与 DDP prediction 文件仍是 Hyper00
staging artifact；预定上传到 `gavinlaw/rasst-main-result-data` 的 versioned rebuttal
artifact。目前没有可用的 Hugging Face 写入凭据，上传状态为 **pending**，该
staging 路径不能视为 canonical source of truth。

## 输入与模型 provenance

- Portable manifest SHA-256：
  `dbf07fd4f8fc6460f2edd0cf1c167ffe61740da6cab63cab9307d3775ad84e89`。
- Portable provenance SHA-256：
  `22f6a653354d0cc9f6b0634bb961e9c32231c023fca790ecbfa21ac869436514`。
- Portable bundle：40 个 content-addressed payloads，共 26,100,339 bytes。
- Model：[`Unbabel/XCOMET-XXL`](https://huggingface.co/Unbabel/XCOMET-XXL)，
  revision `873bac1b1c461e410c4a6e379f6790d3d1c7c214`。
- Checkpoint：42,868,157,218 bytes，SHA-256
  `e760e1f568af69b7a1bf7aeb46d8f3be21f01be7cbda480f8225ee81eb0af27a`。
- `hparams.yaml` SHA-256：
  `0519fd6b5ad74bb15c87894b2b862e1a005219939ad2e474e63eeff5aa6b2214`。
- Encoder tokenizer/config：`facebook/xlm-roberta-xxl` revision
  `03e0fb540c3c9afd4bdda0072e7cb82d2eafd060`。
- `mwerSegmenter` SHA-256：
  `09da1798c65b89d299a6110160d91b1258425928a6164c9d6e3c12ce6057a157`。
- Scoring config SHA-256：
  `d64f05d42ec971cca84771dc0232c45681bedb13a7b89b0500867aefa917a3ae`。
- Inference runner SHA-256：
  `8d4a0d014282128299d996617bb50329cbaa7d00a77d5f02e7dc982e21525a0c`。
- Recovery runner SHA-256：
  `53557f2bf9c538202b703437aedf6b04715b99e7ddb2c72500bfa6b6d547bf73`。
- Validator SHA-256：
  `1c22bf4df4b8be3baa3699d7a9200569e91e204af4f9d9f7a7f01b9f524737b1`。

## 运行环境

- Host：Hyper00，`node-radixark-16-0000`。
- Container：`sglang-omni-jaxan-07112024`。
- Image：
  `hongccc/sglang-omni@sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`。
- GPUs：physical GPU 0
  `GPU-e10a085b-e20e-a454-6fda-17063cf68418` 与 physical GPU 1
  `GPU-2396c1ff-41ff-385f-30fe-0272d99c5d6c`。
- Python 3.12.3；PyTorch 2.11.0+cu130；`unbabel-comet` 2.2.7；
  Transformers 4.57.3；`kernels` 0.11.7；`huggingface-hub` 0.36.2。
- Batch size 16，2-way DDP，`num_workers=0`。稳定推理阶段各 10 秒窗口
  平均 GPU utilization 为 97%--100%。

## 最终恢复与验证命令

以下命令均在 Hyper00 的 `sglang-omni-jaxan-07112024` 容器中运行，所有配置
通过显式参数传入：

```bash
/data/work/xcomet-gpu5-venv/bin/python \
  /data/RASST/code/rasst/analysis/rebuttal/score_sentence_aligned_xcomet.py \
  --manifest /data/xcomet_input_bundle/manifest.tsv \
  --mwer-segmenter /data/tools/mwerSegmenter \
  --checkpoint /data/models/XCOMET-XXL/873bac1b1c461e410c4a6e379f6790d3d1c7c214/checkpoints/model.ckpt \
  --model-id Unbabel/XCOMET-XXL \
  --model-revision 873bac1b1c461e410c4a6e379f6790d3d1c7c214 \
  --devices 0 1 \
  --batch-size 16 \
  --num-workers 0 \
  --prediction-gather-dir /data/results/xcomet_xxl_attempt2/prediction_gather_files \
  --inference-runner-sha256 8d4a0d014282128299d996617bb50329cbaa7d00a77d5f02e7dc982e21525a0c \
  --summary-tsv /data/results/xcomet_xxl_recovered_attempt2/summary.tsv \
  --paired-tsv /data/results/xcomet_xxl_recovered_attempt2/paired.tsv \
  --segments-jsonl /data/results/xcomet_xxl_recovered_attempt2/segments.jsonl \
  --no-progress-bar

/data/work/xcomet-gpu5-venv/bin/python \
  /data/RASST/code/rasst/analysis/rebuttal/validate_xcomet_outputs.py \
  --manifest /data/xcomet_input_bundle/manifest.tsv \
  --summary-tsv /data/results/xcomet_xxl_recovered_attempt2/summary.tsv \
  --paired-tsv /data/results/xcomet_xxl_recovered_attempt2/paired.tsv \
  --segments-jsonl /data/results/xcomet_xxl_recovered_attempt2/segments.jsonl \
  --expected-systems 32 \
  --expected-pairs 16 \
  --expected-segments 22728 \
  --report-json /data/results/xcomet_xxl_recovered_attempt2/validation.json
```

## DDP gather 恢复记录

模型推理全部完成后，COMET 2.2.7 在 DDP gather 中使用未显式指定
`weights_only` 的 `torch.load`。PyTorch 2.6+ 默认 `weights_only=True`，而其
weights-only unpickler 不能恢复 COMET 的 `Prediction(ModelOutput)` dict subclass，
因此初次 gather 失败；模型计算本身没有失败。

COMET 在失败前已经完整写出两个 ranks 的 prediction 与 batch-index 文件。四个
文件被复制到持久化目录并在恢复前后逐个复核：

| 文件 | Bytes | SHA-256 |
| --- | ---: | --- |
| `batch_indices_0.pt` | 40,221 | `987aeabfc30b7818992c589c38b2ec44db38d2627614e7dd03e1affd7090246b` |
| `batch_indices_1.pt` | 40,221 | `e3a13aed080269521533b68d322d567bd1a9ebdbb1cc007f3cdc6dc53b215150` |
| `pred_0.pt` | 3,080,075 | `bbe2abc2d46139c7e593d352093609ed2dc67f6e54041797e8f30b4b4849cc3b` |
| `pred_1.pt` | 3,071,947 | `abe0045cbb14c7821a3df5d572cccccce6db6a8328d342267d4072d06f0d823f` |

Recovery runner 仅在 `CustomWriter.gather_all_predictions` 期间、且仅对 resolve
后直属指定目录并严格命名为 `pred_<rank>.pt` 或
`batch_indices_<rank>.pt` 的普通文件使用 `weights_only=False`。Checkpoint 与
其他 `torch.load` 路径不受影响；recovery 不调用 cleanup，也不删除输入。
Gather 恢复了恰好 22,728 个 scores，`error_spans`、`src_scores`、`ref_scores`、
`unified_scores`、`mqm_scores` 也各为 22,728 条。

失败 inference log SHA-256 为
`05d71ee5960d95ab799027dfb73e33bc8c485b1ec640315a6fe9f7025122eb0b`；
其 exit code 为 1，失败仅发生在 gather。最终逐句记录同时保存 inference runner、
recovery runner 与四个 rank 文件的哈希，确保恢复链条可追踪。
