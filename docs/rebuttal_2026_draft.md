# RASST Rebuttal 2026 — Working Draft

> 内部说明：方括号中的 `PENDING` 字段必须在提交前替换为已验证结果；不要把
> 占位符或本文末尾的事实核对清单提交到 OpenReview。以下英文段落按三位
> reviewer 组织，并只使用当前可审计的结果。

## Shared response: additional evaluation and corrected scope

We thank all reviewers for the careful and constructive feedback. We agree that
the current evaluation overemphasizes exact surface-form terminology accuracy
and BLEU, and that the provenance and realism of the evaluation glossaries and
references need to be clearer. We therefore add the following analyses and will
revise the paper accordingly.

### Contextual translation quality beyond BLEU

We evaluate every sentence-aligned RASST and InfiniSST hypothesis with xCOMET,
a contextual neural metric that also provides error-span information
([Guerreiro et al., 2024](https://aclanthology.org/2024.tacl-1.54/)). The
comparison covers all four latency settings for ACL 60/60 En-Zh/De/Ja and ESO
En-De, i.e., 16 strictly paired RASST--InfiniSST cells. We exclude ESO En-Zh and
En-Ja for the reference-provenance reason discussed below.

Using `Unbabel/XCOMET-XXL` at immutable revision
`873bac1b1c461e410c4a6e379f6790d3d1c7c214`, we score 22,728
sentence-level system hypotheses, each aligned to its source and reference. We
report xCOMET multiplied by 100 and average
the paired cells equally. On ACL, RASST scores **78.704** versus **78.588** for
InfiniSST, a mean paired difference of **+0.116**, with RASST higher in **8/12**
cells. The language-level differences are **+0.636** for En-Zh, **+0.005** for
En-De, and **-0.293** for En-Ja. On human-reference ESO En-De, RASST scores
**74.869** versus **78.041**, a difference of **-3.172**, and is lower in all
four cells. Across all 16 retained cells, the cell-macro difference is
**-0.706** and 8/16 cells are positive. Thus, the contextual-metric evidence is
mixed: it does not support a claim of uniformly improved overall translation
quality, and we will narrow the main claim to terminology handling rather than
general quality improvement. We make no significance claim for this analysis.

As a complementary diagnostic, we recomputed BLEU after removing every raw-gold
target-term string from both the aligned hypothesis and reference. This asks
whether the BLEU difference survives when the directly matched terminology is
excluded. On ACL 60/60, RASST has higher masked BLEU in 10/12 cells, with a mean
delta of **+0.9919 BLEU** versus InfiniSST (the corresponding regular-BLEU mean
delta is **+1.9110**). On human-reference ESO En-De, the masked-BLEU result is
positive in 2/4 cells, with a mean delta of **-0.2047** (regular BLEU:
**+0.2351**). Across the 16 retained cells, masked BLEU is positive in
**12/16** cells and improves by **+0.6927 BLEU on average**, compared with
**+1.4920** regular BLEU. Thus, correct terminology contributes materially to
the BLEU gains, but does not explain all of them on average. We will present
masked BLEU as a diagnostic rather than a causal decomposition, because
deleting terms also changes sentence length and surrounding n-grams.

### A non-oracle, paper-derived glossary condition

We agree that giving the system all human-annotated ACL terms is an oracle-like
preparation condition. We therefore construct a pre-event glossary for each of
the five ACL talks using only its corresponding paper PDF. Gemini 2.5 Flash is
prompted to extract likely technical expressions and provide En-Zh/De/Ja
translations; neither the raw-gold term annotations nor the test references are
available to the extractor. At inference, RASST retrieves only from this
paper-derived glossary. At evaluation, terminology accuracy is still scored
against the unchanged human-annotated raw-gold glossary, so missing extracted
terms remain errors and the denominator does not become easier.

**[PENDING REALISTIC GLOSSARY PROVENANCE — replace before submission: exact API
model identifier/version, PDF hashes or release identifier, extraction prompt,
terms per paper/union after deduplication, and raw-gold term coverage.]**

**[PENDING REALISTIC GLOSSARY RESULTS — replace before submission: verified
TERM_ACC, BLEU, latency, and xCOMET for En-Zh/De/Ja across the reported latency
settings, plus paired InfiniSST deltas. State explicitly whether all four
latency settings or only a preregistered subset were run.]**

This condition removes access to oracle term selection and better represents
conference preparation from public materials. It is not a complete simulation
of field deployment: the extracted glossary translations are still
model-generated, and paper availability does not capture ASR corruption,
last-minute topic changes, or glossary edits by human interpreters. We will
state this limitation rather than describe the condition as fully realistic.

### Why terminology gains can be much larger than BLEU gains

We agree that the BLEU gains are moderate and will temper the corresponding
claim. To quantify how much of each test set is directly covered by annotated
terminology, we counted longest non-overlapping raw-gold glossary matches using
the same language-specific SacreBLEU tokenizers used by evaluation. The exact
counts are:

| Evaluation set | Source term-token share | Aligned target term-token share | Sentences with an aligned target term |
| --- | ---: | ---: | ---: |
| ACL 60/60 En-Zh | 13.90% | 18.06% (2,568/14,222) | 86.54% (405/468) |
| ACL 60/60 En-De | 13.90% | 10.66% (934/8,763) | 83.76% (392/468) |
| ACL 60/60 En-Ja | 13.85% | 12.23% (1,423/11,631) | 85.04% (398/468) |
| ESO En-De | 3.76% | 2.20% (807/36,764) | 31.59% (454/1,437) |

These statistics support a narrower explanation: terminology is widespread
across ACL sentences but remains a minority of target tokens (10.66--18.06%),
and it is especially sparse in ESO En-De (2.20%). A large change in exact
terminology accuracy therefore need not translate into a similarly large
corpus-BLEU change. This is an explanation of metric sensitivity, not evidence
that overall translation quality improves dramatically; the xCOMET and masked
BLEU results provide the direct quality checks.

### ESO reference provenance and corrected presentation

Thank you for flagging the unclear provenance. The original
[ESO dataset](https://github.com/mllpresearch/ESO-dataset) provides manual
German translations (as well as its other original target languages), whereas
we used GPT-5.4 to add Chinese and Japanese translations. Although the paper
mentions this in the evaluation setup, we agree that the distinction was not
prominent enough in the result presentation. In the revision, we will retain
ESO En-De and remove ESO En-Zh/En-Ja from the main results and claims, including
reference-dependent BLEU/xCOMET, exact-form terminology accuracy, and
reference-aligned latency readouts.

We will also separate reference provenance from glossary provenance. The ESO
German reference is human-produced, but the hard-term candidate inventory was
selected with a GPT-assisted Chinese error judge and then manually checked. We
will describe that process explicitly and correct the glossary size: the
artifact has **215 pre-deduplication rows and 212 unique terms**, not 217. We
will not present this author-checked inventory as domain-expert annotation.

## Response to Reviewer Mzub

Thank you for the positive assessment of the method, training description,
ablations, and efficiency, and for identifying several places where the
evaluation and notation need correction.

**Translation-quality metric.** We agree that BLEU alone is insufficient. We
add xCOMET on the 16 retained paired cells and the target-term-masked BLEU
diagnostic summarized above. ACL is nearly unchanged on average under xCOMET
(+0.116 points; 8/12 cells positive), whereas ESO En-De decreases by 3.172
points in all four cells. We will report this negative result and temper the
quality claim accordingly.

**Morphology and exact matching.** We agree that exact-form terminology
accuracy undercounts valid inflected forms, especially in German, and can also
miss legitimate paraphrases. We will rename/describe it consistently as
*exact-form terminology accuracy*, explain that it is a strict and
reference-dependent measure, and add this limitation to the main discussion.
We will not claim that the current experiment resolves morphology. **[PENDING
OPTIONAL ANALYSIS: insert a verified morphology-aware or manually audited
German subset only if completed; otherwise leave it as a limitation.]**

**ESO references.** The German references are manual, while the added Chinese
and Japanese references are GPT-generated. We will retain only ESO En-De in
the main evaluation and make both reference and glossary provenance explicit,
as detailed above.

**Notation, tags, and presentation.** We will define the middle-dot operation
and speech-chunk boundaries in Eq. 1, define \(\phi\) in \(R_\phi\), define the
`[CLS]` representation, improve the source/target separator in Figure 1, and
correct “Speehc.” The `<term>` tags are internal supervision markers and are
removed before scoring; we will explain their purpose and clarify that they do
not make the exact-form metric morphology-aware. We will also discuss direct
untagged generation as a useful future ablation rather than imply that it was
already tested.

## Response to Reviewer oktu

Thank you for highlighting the need to define terminology, document who
selected it, use a contextual metric, and analyze concrete successes and
failures.

**Operational definition and provenance.** In this work, *terminology* is an
operational dataset category rather than a claim of a universal lexical
boundary: it includes domain-specific technical expressions, named methods,
datasets, metrics, acronyms, and specialized names that are supplied in an
evaluation glossary. ACL 60/60 provides human-annotated terminology; the ESO
hard inventory is GPT-assisted and author-checked; the new robustness condition
uses only terms extracted from the corresponding paper. We will distinguish
these sources in a provenance table and state the verifier qualifications
without implying professional-translator or domain-expert verification where
we do not have it.

**Contextual and qualitative evaluation.** We add xCOMET and masked BLEU as
described above. We will also add a compact error analysis that separates at
least: (i) successful rare or multiword terms, (ii) inflectional variants
penalized by exact matching, (iii) homophone/recognition failures, (iv) correct
retrieval but non-use, and (v) delayed-commitment errors. **[PENDING QUALITATIVE
ANALYSIS — insert only verified examples and category counts; do not invent
examples from memory.]**

## Response to Reviewer gbii

Thank you for recognizing the practical problem, streaming integration, and
controlled ablations, and for asking for a less curated robustness test and a
more careful interpretation of the BLEU gains.

**Realistic glossary robustness.** The new paper-derived condition removes the
assumption that an organizer has exposed every raw-gold term: only terms
extracted from each talk's paper are available at inference, while evaluation
keeps the full raw-gold denominator. **[PENDING: insert the verified glossary
coverage and results.]** We will report this separately from the oracle-like
human-annotated glossary condition and describe the residual limitations of
model-generated translations.

**Magnitude of the quality gain.** We agree that the overall translation gain
is incremental rather than transformative. The token-share analysis and
masked-BLEU results above explain why terminology accuracy can move much more
than BLEU while also showing that the effect is not uniformly positive: the
human-reference ESO En-De masked-BLEU mean is -0.2047. We will report this
negative result, avoid overgeneralizing the ACL average, and narrow the main
claim to improved terminology handling with mixed translation-quality effects
that depend on the language and evaluation set. This is also consistent with
the negative xCOMET result on ESO En-De.

**Novelty and comparison scope.** Our intended contribution is the streaming
integration: cross-modal retrieval from partial speech, multi-scale windows,
and training the Speech LLM to decide whether and when to use noisy retrieved
terms. InfiniSST is the principal baseline because it provides a controlled
same-backbone, same-data comparison that isolates retrieval. We agree that this
does not replace broader matched-compute comparisons to biasing and retrieval
variants; we will state that limitation and temper the breadth of the claim.

**Failure analysis.** The revised qualitative analysis will include retrieved-
but-unused terms, homophones, and delayed commitment, as requested. **[PENDING:
insert verified cases/counts.]**

## Planned manuscript changes

Before submission, the revision will:

1. add verified xCOMET results and exact model/checkpoint provenance;
2. add the fully verified masked-BLEU protocol and the term-prevalence table;
3. add the paper-derived glossary protocol, coverage, and results while keeping
   the raw-gold evaluation denominator fixed;
4. retain ESO En-De but remove ESO En-Zh/En-Ja reference-dependent results and
   correct the medicine glossary size from 217 to 215 rows / 212 unique terms;
5. define terminology operationally and distinguish dataset, LLM, author, and
   human-reference provenance;
6. qualify terminology accuracy as exact-form matching and add morphology and
   paraphrase limitations;
7. add only verified qualitative examples and failure categories; and
8. fix the notation, `[CLS]` explanation, `<term>`-tag description, figure
   separator, and typo identified by Reviewer Mzub.

## Evidence used by this draft

- Masked BLEU protocol and exact aggregate:
  `docs/results/rebuttal_2026/masked_bleu_global_cache_snapshot.md`
- Per-cell masked BLEU:
  `docs/results/rebuttal_2026/masked_terms_quality_compare_vs_infinisst_global_cache30_30_20_20_snapshot.tsv`
- Token and sentence prevalence:
  `docs/results/rebuttal_2026/term_prevalence.tsv`
- Verified xCOMET report, paired table, and independent validation:
  `docs/results/rebuttal_2026/xcomet_xxl_report.md`,
  `docs/results/rebuttal_2026/xcomet_xxl_paired.tsv`, and
  `docs/results/rebuttal_2026/xcomet_xxl_validation.json`
- Official xCOMET paper and implementation provenance:
  [TACL paper](https://aclanthology.org/2024.tacl-1.54/),
  [COMET repository](https://github.com/Unbabel/COMET)
- Official ESO data provenance:
  [mllpresearch/ESO-dataset](https://github.com/mllpresearch/ESO-dataset)
- Gemini model documentation for the planned paper-only extractor:
  [Gemini models](https://ai.google.dev/gemini-api/docs/models)

## Internal factual-consistency checklist (remove before submission)

- **Do not say all ESO references are synthetic.** The retained German
  reference is manual; only the added Chinese/Japanese references are
  GPT-generated.
- **Do not call the ESO hard glossary fully human-annotated.** Its candidate
  selection is GPT-assisted and Chinese-error-conditioned; the later check was
  manual. Confirm and report who performed that check and their qualifications.
- **Correct 217.** The current artifact records 215 rows before deduplication
  and 212 unique terms.
- **Do not say terms are rare in every ACL sentence.** They occur in 83.76--
  86.54% of aligned ACL sentences, although they account for only 10.66--18.06%
  of target tokens. ESO En-De is much sparser (2.20% of target tokens; 31.59%
  of sentences).
- **Do not describe masked BLEU as uniformly positive.** The retained aggregate
  is positive in 12/16 cells, but ESO En-De averages -0.2047 and only 2/4 cells
  are positive.
- **Do not present masked BLEU as a causal decomposition.** Removing strings
  changes length and neighboring n-grams.
- **Do not describe xCOMET as uniformly positive.** ACL averages +0.116 points
  with 8/12 positive cells, but En-Ja averages -0.293 and ESO En-De averages
  -3.172 with 0/4 positive cells; the 16-cell macro average is -0.706.
- **Do not report paper-derived glossary numbers until fresh artifacts have
  been generated and validated.** Those fields remain deliberate placeholders.
- **Do not reuse the legacy `acl_paper_extracted` cells as fresh evidence.** The
  tracked table labels them `user_supplied_reusable`, and the exact
  `instances.log` files for those 12 RASST cells are not recoverable. The fresh
  extractor is currently blocked because the legacy Gemini key returns 403 as
  leaked; a new private key is required.
- **Do not call the paper-derived glossary fully real-world or human-created.**
  It removes gold-term selection leakage, but its translations remain
  Gemini-generated.
- **Do not imply morphology is solved.** Current TERM_ACC is exact-form matching;
  a morphology-aware claim requires a separately verified analysis.
- **Check the final removal scope for ESO En-Zh/En-Ja.** BLEU, xCOMET,
  exact-reference TERM_ACC, and reference-aligned latency all depend on the
  synthetic references or their exact forms; removing only BLEU would leave the
  provenance issue partly unresolved.
