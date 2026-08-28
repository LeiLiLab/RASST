# RASST camera-ready research log

## 2026-08-28：rebuttal revision 与 form-aware terminology metric

- **假设：** strict exact-match 会把大小写、空格、连字符和德语屈折/复合变化误计为
  术语错误，但保守归一化后 RASST 相对 InfiniSST 的术语优势应保留。
- **症状：** rebuttal 的人工检查中，127 个 reverse exact losses 有 43 个属于同义、
  语法表面变化、空格或大小写差异；另有 48 个来自 mWER segmentation boundary。
- **原因：** 原 TERM\_ACC 只检查 reference target form 是否逐字出现在 aligned
  hypothesis 中，不处理正字法或形态变化。
- **代码/配置：** 新增
  `code/rasst/eval/offline_sst_eval/compute_form_aware_term_accuracy.py`，保留原分母并
  加入 NFKC/case/spacing/hyphen normalization；德语使用固定版本
  `spaCy==3.8.5`、`de_core_news_sm==3.8.0` 的 contiguous lemma sequence。
- **结果：** 32 个 system rows、27,296 个 occurrences 完成；`lm=2` 的 RASST
  ACL En-De 从 `82.99` 升至 `87.38`，InfiniSST 从 `67.59` 升至 `74.44`，优势
  仍为 `+12.94` points。ESO En-De 对应优势为 `+23.03` points。
- **决定：** 保留 exact-form TERM\_ACC 为主指标，form-aware 只作保守诊断；不采用
  LLM-as-a-Judge 作为 headline metric。论文移除 ESO En-Zh/En-Ja reference-based
  结果，加入 XCOMET、term-tag、qualitative、end-to-end retrieval、paper-derived
  glossary 和 retrieval-degradation 分析。完整 occurrence artifact 暂存 Taurus，
  状态 `PENDING_HF_UPLOAD`。

