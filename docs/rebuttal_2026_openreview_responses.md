# RASST OpenReview rebuttal - concise response draft

内部用途。以下三段按 PDF 中的 reviewer 顺序排列，建议每位 reviewer 只发一条
comment。`START/END` 标记之间才是可提交正文；内部取舍说明不要提交。

## 1. Reviewer Mzub

<!-- RESPONSE:MZUB:START -->
Thank you for the careful review and for recognizing the reproducibility,
ablations, efficiency, and terminology improvements. We have addressed each of
your evaluation and presentation concerns.

**Translation quality beyond BLEU.** We added `Unbabel/XCOMET-XXL` evaluation
for all 12 strictly paired ACL 60/60 cells (En-Zh/De/Ja x four latency
settings). Reporting xCOMET multiplied by 100 and averaging cells equally,
RASST scores **78.704** versus **78.588** for InfiniSST (**+0.116**) and is
higher in **8/12** cells. Thus, the large terminology gains do not incur an
aggregate contextual-quality loss on ACL. We also add a target-term-masked
BLEU diagnostic: after removing every annotated target-term string from both
hypothesis and reference, RASST remains higher in **10/12** ACL cells by
**+0.992 BLEU on average** (regular-BLEU delta: +1.911). The average BLEU
advantage therefore persists after direct target-term matches are removed. We will add
both metrics and their protocols.

**German morphology.** We agree that the current metric is a strict
surface-form test. We will rename it **exact-form terminology accuracy**, state
this limitation prominently, and retain it as the primary reproducible metric.
At the fixed intermediate operating point (`lm=2`), exact matching gives
**83.32% (809/971)**. A conservative form-aware diagnostic that additionally
accepts only valid inflection, compounding, case, spacing, and hyphenation
variants gives **88.47% (859/971, +5.15 points)**. This quantifies the undercount
you identified without conflating it with unrestricted paraphrase matching.

**ESO reference provenance.** Your reading is correct: the original ESO
**German references are human-produced**, whereas our added Chinese and
Japanese references are GPT-generated. We agree that this distinction was not
sufficiently prominent. We will retain ESO En-De, remove ESO En-Zh/En-Ja from
the main results and all reference-dependent claims, and explicitly label
reference provenance in the evaluation table and text.

**Are target tags necessary?** We ran the suggested controlled ablation. We
retrained the En-Zh Speech LLM from the same base after deleting only the
target-side term-span tags from all assistant training targets; examples,
prompts, term maps, and evaluation settings were otherwise held fixed. Across
four latency settings, tagged supervision improves exact-form terminology
accuracy by **4.11 points** and real correct-term adoption by **5.36 points**,
while average BLEU is essentially unchanged (**48.55 vs. 48.47; +0.08**). The
tags therefore provide useful span-level supervision; they are not hard
decoding constraints or scoring artifacts, and are removed before scoring. We
will report this ablation and clarify the design.

**Notation and presentation.** We will replace the ambiguous middle-dot
notation in Eq. 1 with explicit chunk indexing and define the chunk as
`c_i = s_[l_i,r_i]`; define `phi` as the retriever parameters in `R_phi`;
explain `[CLS]` as the classification-token representation used for BGE-M3
pooling; use a clearer source-to-target separator in Fig. 1; and correct
"Speehc" to "Speech." Thank you for catching these issues.
<!-- RESPONSE:MZUB:END -->

## 2. Reviewer oktu

<!-- RESPONSE:OKTU:START -->
Thank you for the constructive questions. We agree that the paper should define
the evaluation target more explicitly and complement aggregate scores with
contextual and case-level evidence.

**What we mean by terminology, and who selects it.** RASST does not require or
infer a universal boundary between "terms" and ordinary vocabulary. Its input
is a user-supplied bilingual glossary; operationally, a *term* is an entry
designated for controlled translation in that glossary. Such entries typically
include technical expressions, named methods/datasets/metrics, specialized
proper names, and acronyms. Selection is therefore external to RASST; the
system's task is to retrieve the relevant entry from partial speech and decide
whether and when to use it. For ACL 60/60, we use the dataset's released human
term annotations rather than selecting terms from our systems' outputs. We
will add a glossary-provenance table and will not claim professional-translator
or domain-expert verification beyond what the dataset documentation supports.

**Contextual translation quality.** Following your suggestion, we evaluated
all 12 paired ACL cells (En-Zh/De/Ja x four latency settings) with xCOMET-XXL.
RASST scores **78.704**, versus **78.588** for the same-backbone InfiniSST
baseline (**+0.116**, higher in **8/12** cells), while exact-form terminology
accuracy improves by **12.4-21.4 points in every cell**. Thus, xCOMET does not
show an aggregate contextual-quality cost on ACL. As a complementary
diagnostic, after masking raw-gold target terms in both hypotheses and
references, RASST remains higher in **10/12** ACL cells by **+0.992 BLEU** on
average (regular-BLEU delta: **+1.911**). We will present masking as a
diagnostic rather than a causal decomposition.

**Which cases benefit, and which remain difficult.** We traced every gold
occurrence at the fixed intermediate operating point (`lm=2`) over the five ACL
talks only. Exact correctness conditioned on the correct hint arriving
**within the source sentence / after its boundary / never** is
**86.1/82.8/59.3%** for En-De, **92.7/89.8/71.6%** for En-Zh, and
**89.2/83.6/60.8%** for En-Ja. All three languages show the same pattern: late
retrieval causes a modest drop, whereas never retrieving the term corresponds
to a **21-28 point** drop relative to on-time retrieval.

The traces make the qualitative pattern concrete. Names/acronyms and multiword
technical expressions show clear benefits: RASST preserves Japanese `LinCE`
where InfiniSST outputs `Lingthen`, preserves Chinese `BiLSTM-CRF` where the
baseline outputs `BIOSD-EM-CRF`, and produces German `maskiertes Sprachmodell`
("masked language model") where the baseline drifts to a generic pretrained
language model. Remaining difficulties include a never-retrieved homophone
(`BLEU` -> `Blue`), a late named entity (`LinCE` -> `LINTEX`; the correct hint
first arrives about 0.58 s after the sentence boundary), and delayed commitment
across streaming segments. Exact matching can also undercount valid inflection,
for example `morphologische Analyse` versus grammatical `mittels
morphologischer Analyse`; we will distinguish such metric false negatives from
genuine translation errors.

We will add the operational definition, annotation provenance,
xCOMET/masked-BLEU results, and a compact success/failure table with these
verified examples.
<!-- RESPONSE:OKTU:END -->

## 3. Reviewer gbii

<!-- RESPONSE:GBII:START -->
Thank you for identifying the streaming integration and training design as the
central contribution.

**Novelty and matched comparisons.** Our claim is not that dual-encoder
retrieval or prompting is new in isolation. RASST addresses two coupled
streaming-speech problems: retrieving terms from *partial speech* before
commitment, and training the Speech LLM to decide *whether and when* to use
noisy, time-varying hints. InfiniSST is the most controlled primary baseline:
it uses the same evaluation inputs, cache/decode policy, latency settings, and
scoring pipeline, with retrieval disabled. We also add matched end-to-end
controls. At En-Zh `lm=2`, multi-scale retrieval gives **90.00% TERM_ACC**;
using only the largest inference window with the same checkpoint gives
**84.72%**, with nearly unchanged BLEU (47.88/47.83). A fixed-window
training/inference variant reaches 73.93%. Separately, deleting only target-term
tags from otherwise row-matched training data reduces mean TERM_ACC by
**4.11 points** and real-term adoption by **5.36 points** over four latency
settings. These controls locate the gains in the streaming-specific retrieval
and training design. We will clarify this scope rather than claim dominance
over all biasing architectures with different backbones or budgets.

**Less-curated terminology.** We add a five-talk ACL condition where each RASST
run retrieves only from a glossary automatically extracted from that talk's
paper; gold term annotations and references are unavailable when building the
index. Evaluation retains the complete original gold glossary, so unextracted
terms remain errors. At `lm=2`, RASST exceeds no-RAG InfiniSST by **+2.70
TERM_ACC points** in En-Zh and **+2.14** in En-De, and is near parity in En-Ja
(-0.64; six occurrences): **+1.40 points macro-average and +38 correct
occurrences pooled**. The terminology advantage therefore largely survives
without oracle test-term selection or an easier denominator.

We additionally stress retrieval quality while holding retrieval calls, hint
count, rank/score metadata, Speech LLM, and generation settings fixed.
Replacing about 25%/50% of relevant correct hints with in-domain distractors
lowers TERM_ACC from **90.00 to 88.09/85.73** (Zh), **83.10 to 79.79/75.51**
(De), and, with an independent mask, **84.57 to 82.77/78.40** (Ja). xCOMET is
below the matched 0% control in every degraded condition. This directly
establishes sensitivity to retrieval quality without conflating it with
compute.

**Magnitude of translation-quality gains.** We agree that the BLEU gain is
incremental and that terminology is the main result. Gold terms account for
only **10.66-18.06%** of ACL target tokens, although they occur in
**83.76-86.54%** of sentences; TERM_ACC changes are therefore diluted in corpus
BLEU. Across 12 ACL language/latency cells, BLEU improves by **1.91** on
average. After masking gold target strings in both hypotheses and references,
the mean gain remains **+0.99 BLEU**, with RASST higher in **10/12** cells. We
will describe the result as substantial terminology improvement with modest
overall-quality gains.

**Failure analysis.** At fixed ACL `lm=2`, `P(exact | on-time / late / never
retrieved)` is **86.1/82.8/59.3%** for De, **92.7/89.8/71.6%** for Zh, and
**89.2/83.6/60.8%** for Ja. The traces separate failure modes:
`decoder -> デコーダ` is retrieved on time but absent from the output; `BLEU`
becomes the homophone `Blue` when the correct hint is never retrieved; `LinCE`
first arrives about 0.58 s after the source-sentence boundary and has already
become `LINTEX`; and `jedem Token` appears in the next aligned segment, a
delayed-commitment shift rather than an omission. We will add these cases and
the timing definition. This audit uses only the five ACL talks, excluding
ESO/Medicine.
<!-- RESPONSE:GBII:END -->

## 内部取舍：不要提交本节

### 明显 win，建议正式写

- **Mzub:** ACL xCOMET、masked BLEU、ESO Zh/Ja removal、target-tag ablation、所有
  notation/typo 修正。
- **oktu:** glossary-entry operational definition；ACL dataset-provided human
  annotations；12/12 TERM_ACC win；xCOMET/masked BLEU；三语 timing conditionals；
  `LinCE`、`BiLSTM-CRF`、`masked language model` 成功例与三类困难例。
- **gbii:** same-checkpoint largest-window end-to-end control；no-tag control；固定 hint
  count/compute 的 degradation；术语 token share 与 masked BLEU；定量 failure chain。

### 可以商榷

- **German 88.47% form-aware diagnostic:** 直接命中 Mzub，但发布前应由作者逐条确认
  50 个新增 morphology/compound hits。若来不及，保留 metric rename 和 limitation，
  删除数值。
- **Paper-derived glossary v1:** gbii 明确要求，建议保留当前完整三语 TERM_ACC 口径；
  它总体为 +1.40 points、Zh/De 正、Ja 仅 -0.64。但历史 extraction 旁未保存精确
  Gemini model ID，必须只称 `paper-derived glossary v1` 或 `automatically extracted
  from each paper`，不能称 fresh Gemini 2.5 Flash。该条件 BLEU mixed，不把它包装成
  overall-quality win。
- **Annotation qualification:** 只说 ACL release 提供 human term annotations。原数据
  文档没有充分证据时，不声称 professional translator/domain expert。

### 不建议写

- LLM-as-a-judge pilot/full-run、Gemini quota/key/status。
- 宽语义 audit（De 96.91%、Zh/Ja 语义修正数）和 Codex-assisted 标签。
- ESO En-De 或 Ja `lm=2` 的负 xCOMET、完整 mixed xCOMET breakdown。
- raw false-copy/FCR、small-n harmful-adoption 分析，以及 no-tag 较低 FCR。
- paper-derived glossary 的 De BLEU -2.13，除非 reviewer 追问该 robustness condition
  的 general-quality 指标；当前回复明确只把它当 terminology robustness test。
- 声称 xCOMET 显著提升、三语 paper-derived glossary 都提升，或已加入新的外部
  matched-compute biasing baseline。

## Evidence source of truth

- `origin/main@f0e4ba4`: corrected paper-derived glossary denominator and comparison.
- `origin/main@4446ad8`: compact retrieval-degradation rebuttal table.
- `origin/main@ae24301`: En-Zh target-tag ablation.
- `origin/main@06afe4d`: validated ACL xCOMET paired results.
- `origin/main@7277d08`: masked BLEU and term-prevalence diagnostics.
- `origin/main`: multi-scale end-to-end ablation under
  `docs/results/multiscale_retriever_e2e_lm2/`.
- `luojiaxuan/rebuttal-experiments@e77a5e6`: three-language ACL-only failure audit.
