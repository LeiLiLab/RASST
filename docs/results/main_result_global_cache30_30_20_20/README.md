# Main Result Snapshot: Global Cache 30/30/20/20

This directory is the tracked release snapshot for the final RASST main result
policy:

```text
lm=1,2 -> max_chunks=keep_chunks=30
lm=3,4 -> max_chunks=keep_chunks=20
```

The runtime source files remain under ignored `outputs/` and `figures/`; this
directory is the small Git-tracked copy used for release documentation.

## Files

| File | Contents |
| --- | --- |
| `main_result.tsv` | Full merged main-result table with baselines and RASST rows. |
| `rasst24.tsv` | The 24 RASST cells used for the release main result. |
| `compare_vs_infinisst_and_paper.tsv` | Per-cell BLEU and TERM_ACC deltas versus InfiniSST and the paper-exact RASST table. |
| `masked_terms_quality.tsv` | Per-row target-term-masked BLEU computed from available hypothesis artifacts. |
| `masked_terms_quality_compare_vs_infinisst.tsv` | Per-cell RASST versus InfiniSST target-term-masked BLEU comparison. |
| `masked_terms_artifacts.tsv` | Local artifact map for masked-term hypothesis logs not named in `main_result.tsv`. |
| `masked_terms_bleu_global_cache30_30_20_20.pdf/png` | Combined ACL tagged and Medicine hard/raw target-term-masked BLEU figure. |
| `artifacts/acl_tagged_raw_infinisst_zh/` | Local copies of ACL6060 tagged En-Zh InfiniSST no-RAG validation TSVs and ignored `instances.log` files. |
| `new_main_result_tagged_global_cache30_30_20_20.pdf/png` | ACL tagged main-result figure. |
| `medicine_main_result_global_cache30_30_20_20.pdf/png` | Medicine hardraw main-result figure. |

## Runtime Sources

```text
/mnt/taurus/data2/jiaxuanluo/RASST/outputs/canonical/main_result/paper_global_cache30_30_20_20_main_result.tsv
/mnt/taurus/data2/jiaxuanluo/RASST/outputs/canonical/main_result/paper_global_cache30_30_20_20_rasst24.tsv
/mnt/taurus/data2/jiaxuanluo/RASST/outputs/canonical/main_result/paper_global_cache30_30_20_20_compare.tsv
/mnt/taurus/data2/jiaxuanluo/RASST/figures/main_result_global_cache30_30_20_20/
```

## Summary

The global policy improves BLEU over InfiniSST in 19 of 24 RASST cells.

| Dataset | Lang | BLEU wins | Avg delta BLEU | Avg delta TERM_ACC |
| --- | --- | ---: | ---: | ---: |
| `acl_tagged_raw` | `de` | 3/4 | +0.6793 | +0.1824 |
| `acl_tagged_raw` | `ja` | 3/4 | +1.6487 | +0.1946 |
| `acl_tagged_raw` | `zh` | 4/4 | +3.4049 | +0.1329 |
| `medicine_hardraw` | `de` | 2/4 | +0.2350 | +0.2790 |
| `medicine_hardraw` | `ja` | 3/4 | +1.3330 | +0.4405 |
| `medicine_hardraw` | `zh` | 4/4 | +3.1231 | +0.3540 |

## Masked-Term BLEU

`MASKED_TERMS_BLEU` removes target-side glossary translations from both the
hypothesis and reference after the same mWER resegmentation used by the regular
BLEU scorer. It is a non-term quality check: if the RASST delta disappears after
masking, the original BLEU gain is mostly terminology-driven.

Among the 24 cells with artifact-backed InfiniSST hypothesis logs, RASST keeps
positive masked-term BLEU deltas in 19/24 cells. The average delta drops from
`+1.7373` regular BLEU to `+1.1238` masked-term BLEU, so terminology accounts
for part of the BLEU gain but does not fully explain it in the artifact-backed
comparison.

| Track | Cells | Masked BLEU wins | Avg delta regular BLEU | Avg delta masked-term BLEU |
| --- | ---: | ---: | ---: | ---: |
| ACL6060 tagged | 12 | 10/12 | +1.9110 | +0.9919 |
| Medicine hard/raw | 12 | 9/12 | +1.5637 | +1.2557 |
| Artifact-backed total | 24 | 19/24 | +1.7373 | +1.1238 |

ACL6060 tagged zh InfiniSST hypothesis logs were backfilled from the old
InfiniSST rank16 baseline no-RAG ACL6060-only v3 run. The copied logs reproduce
the published En-Zh InfiniSST BLEU and StreamLAAL rows; `masked_terms_artifacts.tsv`
maps those local copies into the masked-term scorer without changing
`main_result.tsv` provenance.

## Checksums

```text
1929cc8d7883c99f136d104adb6efe92199ca529d21a7f6b9bf683ab4f50f95b  main_result.tsv
278af3606fe863c2f93d3281dd1714f19a4a3dffe4e6749ff3190b7d4c306406  rasst24.tsv
11221fabdde32e42862014454b81faa56ad9822646b790cc250565d60f1941d0  compare_vs_infinisst_and_paper.tsv
3265c5b3c808c44f5227ebe5626b5a7ff194da6535267a17eb88f7e65604af14  masked_terms_quality.tsv
ac06dec613a35a7c1cc1d04a56dbe7534834178ef229d89cb3357b6f2a727c6b  masked_terms_quality_compare_vs_infinisst.tsv
2964438ed82a43a8ba7e394ce9b1ceacc9c0788a6a6726b502b3768401215b7a  masked_terms_artifacts.tsv
7c1147cd60c5bfef2007a71405d53af5c8891e5ac3f24e2a8366eae87e30349f  masked_terms_bleu_global_cache30_30_20_20.pdf
e55f6c7d2fe309ac1125a47400b28975e259554285af8e9a48300aca41c6c680  masked_terms_bleu_global_cache30_30_20_20.png
f2c9d4df06dc0888e347a59418cb7b480ea0d102a34b29142e36c258c09aa31b  new_main_result_tagged_global_cache30_30_20_20.pdf
319cc377ba4bda7ddd1311f3e537be51029b6543c3007498df496353eca6de3a  new_main_result_tagged_global_cache30_30_20_20.png
845d904908343ea6036fa79fd3aebd06ef96daad6459b8096f9d609c868bcd10  medicine_main_result_global_cache30_30_20_20.pdf
1eb9678437c7fe1e89b3a548cd9300b837922ae2e282bdc3a55f244b463a78e6  medicine_main_result_global_cache30_30_20_20.png
```
