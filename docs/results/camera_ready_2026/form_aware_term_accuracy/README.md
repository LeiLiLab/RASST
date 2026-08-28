# Camera-ready form-aware terminology accuracy

本目录保存 camera-ready revision 使用的轻量、可复算结果。主指标仍是论文中
原有的 exact-form terminology accuracy；form-aware accuracy 是保守的补充诊断，
用于回答 exact match 是否把大小写、Unicode/全半角、空格/连字符和德语屈折变化
误计为错误。该诊断不接受同义词、意译或 LLM 语义判断。

## 协议

- 评分输入为已存在的 `instances.log` / `instances.strip_term.log`，没有重新运行模型
  推理。
- 分母复现 `stream_laal_term.py`：按 target translation 去重；source term 必须出现
  在 aligned source sentence 中，target translation 必须出现在 reference 中；同一
  target form 每句至多计一次。
- 所有语言先做 NFKC、case、空格、连字符和宽度归一化。德语再使用
  `spaCy==3.8.5` 与 `de_core_news_sm==3.8.0` 比较连续 lemma sequence。
- `summary.tsv` 共 32 个 system rows、27,296 个 occurrence rows。28 个有
  `term_correct/term_total` provenance 的 rows 均与原 exact scorer 逐格一致。
- ACL En-Zh 的四个 InfiniSST 主表数字是早期 user-supplied aggregate，原记录没有
  `term_correct/term_total`。当前可得 hypothesis artifacts 能复现 BLEU 与
  StreamLAAL，但重算 exact TERM_ACC 为 `71.91/75.17/74.04/75.51`，与主表
  `74.31/76.55/76.75/77.54` 不完全一致。因此论文中的 form-aware 诊断不使用这
  四行做新的系统间 headline comparison；ACL En-Zh 主结果继续保留已提交口径。

## Camera-ready 读数

在 latency multiplier 2 下，RASST 的 exact/form-aware TERM_ACC 为：ACL En-Zh
`88.99/89.10`、En-Ja `84.57/84.57`、En-De `82.99/87.38`，以及 ESO En-De
`78.21/82.23`。ACL En-De 的 InfiniSST 为 `67.59/74.44`，因此保守归一化后
RASST 的优势仍为 `+12.94` points。完整四档结果见 `summary.tsv`。

## Provenance 与产物状态

- 运行清单、输入哈希、glossary 哈希、依赖版本和输出哈希见
  `run_manifest.json`。
- Taurus 运行根目录：
  `/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/camera_ready_term_metric_20260828`。
- 逐 occurrence 文件 `occurrences.jsonl` 为 22 MB，保留在上述 persistent root，
  不进入 Git。预定 Hugging Face 目标为
  `gavinlaw/rasst-main-result-data` 的 camera-ready versioned artifact；当前状态为
  `PENDING_HF_UPLOAD`。

