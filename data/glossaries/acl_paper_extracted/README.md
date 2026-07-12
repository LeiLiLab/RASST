# ACL Paper-Extracted Glossaries

These five glossaries are the realistic-glossary rebuttal condition. Each
glossary was extracted from one ACL6060 source paper with the original Gemini
extraction pipeline, then translated into `zh`, `de`, and `ja`. Unlike the
tagged-ACL glossary, the entries are selected from the paper content rather
than from reference-side annotations.

`manifest.json` records term counts, checksums, source papers, and the limits of
the recovered extraction provenance. The exact historical Gemini invocation
was not stored with the JSON artifacts; the extraction tooling default was
Gemini 1.5 Flash, but the manifest does not claim that default as a verified
run-time fact.

The rebuttal evaluation record is in
[`docs/results/acl_paper_extracted_lm2`](../../../docs/results/acl_paper_extracted_lm2/).
