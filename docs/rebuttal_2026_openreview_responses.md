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
advantage therefore persists after direct target-term matches are removed. We
will add both metrics and their protocols.

**German morphology.** We agree that the current metric is a strict
surface-form test. We will rename it **exact-form terminology accuracy**, state
this limitation prominently, and retain it as the primary reproducible metric.
At the fixed intermediate operating point (`lm=2`), exact matching gives
**83.32% (809/971)**. A conservative form-aware author diagnostic that
additionally accepts only checked inflection, compounding, case, spacing, and
hyphenation variants gives **88.47% (859/971, +5.15 points)**. This quantifies
the undercount you identified without conflating it with unrestricted
paraphrase matching.

**ESO reference provenance.** We confirm the distinction:

| ESO slice | Sentence-reference provenance | Revision treatment |
| --- | --- | --- |
| En-De | Original manual translation in the [released ESO dataset](https://github.com/mllpresearch/ESO-dataset) | Retain as the ESO reference-based evaluation |
| En-Zh/En-Ja | Added by us with GPT-5.4 | Remove BLEU, xCOMET, and other synthetic-reference-dependent claims from the main results |

We will also separate **reference provenance** from **glossary provenance**.
The ESO hard-term inventory was produced with GPT-5.4 in Stages 1/2/5 and then
manually checked; it will be labeled **GPT-assisted and manually checked**, not
presented as fully human-authored or domain-expert annotation. Any retained
term-only readout will be clearly separated from reference-based
translation-quality metrics.

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

**Notation clarification and camera-ready fixes.** We will make the method
definitions explicit: replace the ambiguous middle-dot in Eq. 1 with chunk
indexing (`c_i = s_[l_i,r_i]`), define `phi` in `R_phi` as the retriever
parameters, and explain `[CLS]` as the classification token whose BGE-M3
representation is used for text pooling. The remaining points are presentation
fixes for the camera-ready: Fig. 1 will use a clearer
`source term -> target term` separator, and "Speehc" will be corrected to
"Speech."
<!-- RESPONSE:MZUB:END -->

## 2. Reviewer oktu

<!-- RESPONSE:OKTU:START -->
Thank you for the constructive questions. We agree that the paper should define
the evaluation target explicitly and complement aggregate scores with
contextual and case-level evidence.

**Definition and provenance.** RASST does not need to infer a universal
boundary between "terminology" and ordinary vocabulary. Its input is a
user-supplied bilingual glossary; operationally, a *term* is an entry designated
for controlled translation in that glossary. Typical entries are technical
expressions, named methods/datasets/metrics, specialized proper names, and
acronyms. Selection is external to RASST: the system retrieves relevant entries
from partial speech and decides whether and when to use them.

We will add this provenance distinction:

| Glossary | Selection and verification | Qualification we claim |
| --- | --- | --- |
| **ACL 60/60** | Non-exhaustive lists created by the 60/60 initiative; source spans were automatically tagged. Professional post-editors corrected and tagged the target spans. | The English transcript's technical terms were checked by a domain expert; target translations were professionally post-edited by a target-language-native annotator and second-reviewed. |
| **ESO hard terms** | Seven-stage hybrid pipeline; final glossary manually checked. | GPT-5.4-assisted and manually checked; **not** claimed as fully human-authored, professional-translator-verified, or domain-expert annotation. |

For ESO, GPT-5.4 generated Zh/Ja translations and candidate terms in Stage 1
(10 sentences/batch), enforced exact-span and abbreviation consistency in
Stage 2, and compared the no-RAG baseline in Stage 5 (3 sentences/batch) to
identify hard terms. Non-LLM Stages 3/4 restored sample-level consistency and
filtered source-unmatched terms; Stage 6 manually checked the glossary and
Stage 7 emitted the final output. The ACL qualifications above follow
[Salesky et al. (2023), Secs. 3.4-3.7](https://aclanthology.org/2023.iwslt-1.2/);
we do not conflate source-term expertise with target-translation review.

**Contextual translation quality.** Following your suggestion, we evaluated
all 12 paired ACL cells (En-Zh/De/Ja x four latency settings) with xCOMET-XXL.
RASST scores **78.704**, versus **78.588** for the same-backbone InfiniSST
baseline (**+0.116**, higher in **8/12** cells), while exact-form terminology
accuracy improves by **12.4-21.4 points in every cell**. Thus, xCOMET does not
show an aggregate contextual-quality cost on ACL, although we will not claim a
large universal gain in general quality. As a complementary diagnostic, after
masking raw-gold target terms in both hypotheses and references, RASST remains
higher in **10/12** ACL cells by **+0.992 BLEU** on average (regular-BLEU delta:
**+1.911**). We present masking as a diagnostic, not a causal decomposition.

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

We will add the operational definition, provenance table, xCOMET/masked-BLEU
results, and a compact success/failure table with these verified examples.
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
scoring pipeline, with retrieval disabled. Our new controlled evidence is:

| ACL evidence | Matched comparison | Result |
| --- | --- | --- |
| Multi-scale, Zh `lm=2` | Largest-infer / fixed-window train+infer | TERM_ACC **90.00 vs. 84.72/73.93**; BLEU 47.88 vs. 47.83/45.80 |
| Target-tag SFT, Zh `lm=1-4` | Target tags removed only | **+4.11 pp** TERM_ACC; **+5.36 pp** correct-term adoption |
| 25%/50% hint corruption, `lm=2` | 0%; same hint count/retrieval compute | Delta TERM_ACC Zh: -1.91/-4.27; De: -3.31/-7.59; Ja: -1.80/-6.17 pp; xCOMET lower in all six |
| Target-masked BLEU, 12 cells | InfiniSST | **+0.99** average; **10/12** wins (regular BLEU: +1.91) |

The first two rows locate the gains in the streaming-specific retrieval and
training design, not merely appending glossary entries to a prompt. We will
clarify this scope rather than claim dominance over all biasing architectures
with different backbones or budgets.

**Less-curated terminology.** In the five-talk paper-derived condition, each
RASST run retrieves only from a glossary built from that talk's paper; neither
the ACL tagged-raw glossary nor references are used to build the index. Both
systems are scored against the unchanged ACL tagged-raw glossary, so terms
absent from the paper-derived index remain errors. At the default `lm=2`
operating point:

| Language | TERM_ACC: RASST / InfiniSST (delta) | BLEU: RASST / InfiniSST (delta) |
| --- | ---: | ---: |
| En-Zh | **77.87 / 75.17 (+2.70 pp)** | **46.3280 / 45.8268 (+0.5012)** |
| En-Ja | **65.32 / 65.96 (-0.64 pp)** | **27.7656 / 27.7202 (+0.0455)** |
| En-De | **70.91 / 68.21 (+2.70 pp)** | **29.2086 / 30.2743 (-1.0657)** |

The term-accuracy macro delta is **+1.59 points**: the advantage survives in
En-Zh and En-De, while En-Ja is near parity. BLEU is preserved in En-Zh/En-Ja
and lower in En-De, so we treat this as a terminology-robustness result rather
than a uniform overall-quality claim.

**Sensitivity to retrieval quality.** The corruption test holds retrieval
calls, hint count, rank/score metadata, Speech LLM, and generation settings
fixed. TERM_ACC and xCOMET are below the matched 0% control in every degraded
language/level, directly establishing sensitivity without conflating it with
compute.

**Magnitude of quality gains.** We agree that BLEU improves incrementally and
terminology is the main result. Gold terms account for only **10.66-18.06%** of
ACL target tokens, although they occur in **83.76-86.54%** of sentences; gains
are therefore diluted in corpus BLEU. The masked-BLEU result shows that direct
gold-term strings do not explain the entire difference. We will describe the
result as substantial terminology improvement with modest overall-quality
gains.

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

- **Mzub:** ACL xCOMET、masked BLEU、ESO Zh/Ja reference-based quality metric
  removal、target-tag ablation、所有 notation/typo 修正。
- **oktu:** glossary-entry operational definition；ACL dataset-provided human
  annotations 的官方 qualification（source technical terms 经 domain expert
  check；target 由母语 professional post-editor + second reviewer）；12/12
  TERM_ACC win；xCOMET/masked BLEU；三语 timing conditionals；`LinCE`、
  `BiLSTM-CRF`、`masked language model` 成功例与三类困难例。
- **gbii:** same-checkpoint largest-window end-to-end control；no-tag control；固定 hint
  count/compute 的 degradation；术语 token share 与 masked BLEU；定量 failure chain。

### 可以商榷

- **German 88.47% form-aware diagnostic:** 直接命中 Mzub，但发布前应由作者逐条确认
  50 个新增 morphology/compound hits。若来不及，保留 metric rename 和 limitation，
  删除数值。
- **Paper-derived glossary v1:** gbii 明确要求，建议保留当前完整三语 TERM_ACC 口径；
  作者确认的 reported macro delta 为 +1.59 points、Zh/De 正、Ja 仅 -0.64。但
  De 的 pooled counts `652/935` 和 `632/935` 对应 69.73% 与 67.59%，并不等于
  reported 70.91% 与 68.21%；内部 SoT 必须把 reported TERM_ACC 与 pooled counts
  分列，正式 rebuttal 只报作者确认的 reported metric。历史 extraction 旁未保存精确
  Gemini model ID，必须只称 `paper-derived glossary v1` 或 `automatically extracted
  from each paper`，不能称 fresh Gemini 2.5 Flash。BLEU 为
  `+0.5012/+0.0455/-1.0657`（Zh/Ja/De），应明确是 mixed，不包装成 overall-quality
  win。
- **ESO 七阶段 provenance:** oktu 直接问术语由谁选择/验证，因此建议用一行表格和
  一段流程透明披露。Stage 5 虽然是 GPT 对 baseline 的判断，但它只用于定义
  hard-term subset，不是本次 rebuttal 的通用 translation-quality evaluation。
  若字符紧张，可缩短各 Stage 描述，不能删掉 “GPT-5.4-assisted and manually
  checked” 的总口径。
- **ACL qualification 边界:** 可以且应该写 source technical terms 经 domain
  expert check、target translation 由目标语言母语 professional post-editor 完成并
  second-reviewed；不能合并成 “all target terms were domain-expert translated”。

### 不建议写

- LLM-as-a-judge pilot/full-run、Gemini quota/key/status。
- 宽语义 audit（De 96.91%、Zh/Ja 语义修正数）和 Codex-assisted 标签。
- ESO En-De 或 Ja `lm=2` 的负 xCOMET、完整 mixed xCOMET breakdown。
- raw false-copy/FCR、small-n harmful-adoption 分析，以及 no-tag 较低 FCR。
- 声称 xCOMET 显著提升、三语 paper-derived glossary 都提升，或已加入新的外部
  matched-compute biasing baseline。
- 把 ESO hard-term inventory 称为 fully human-authored、professional-translator
  verified 或 domain-expert annotation。

## Evidence source of truth

- `origin/main:docs/results/acl_paper_extracted_lm2/author_reported_lm2_update.tsv`:
  author-confirmed default-`lm=2` paper-derived TERM_ACC/BLEU, with
  reported-vs-pooled aggregation kept separate.
- `origin/main@4446ad8`: compact retrieval-degradation rebuttal table.
- `origin/main@ae24301`: En-Zh target-tag ablation.
- `origin/main@06afe4d`: validated ACL xCOMET paired results.
- `origin/main@7277d08`: masked BLEU and term-prevalence diagnostics.
- `origin/main`: multi-scale end-to-end ablation under
  `docs/results/multiscale_retriever_e2e_lm2/`.
- `luojiaxuan/rebuttal-experiments@e77a5e6`: three-language ACL-only failure audit.
- `docs/results/rebuttal_2026/eso_hard_term_pipeline/`: 七阶段 ESO hard-term
  protocol、Stage 1/2/5 exact prompts 与 SHA-256。
- [Salesky et al. (2023), Secs. 3.4--3.7 and App. A.5](https://aclanthology.org/2023.iwslt-1.2/):
  ACL 60/60 annotation and professional post-editing qualifications.
