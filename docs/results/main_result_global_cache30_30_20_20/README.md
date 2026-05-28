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

## Checksums

```text
1929cc8d7883c99f136d104adb6efe92199ca529d21a7f6b9bf683ab4f50f95b  main_result.tsv
278af3606fe863c2f93d3281dd1714f19a4a3dffe4e6749ff3190b7d4c306406  rasst24.tsv
11221fabdde32e42862014454b81faa56ad9822646b790cc250565d60f1941d0  compare_vs_infinisst_and_paper.tsv
f2c9d4df06dc0888e347a59418cb7b480ea0d102a34b29142e36c258c09aa31b  new_main_result_tagged_global_cache30_30_20_20.pdf
319cc377ba4bda7ddd1311f3e537be51029b6543c3007498df496353eca6de3a  new_main_result_tagged_global_cache30_30_20_20.png
845d904908343ea6036fa79fd3aebd06ef96daad6459b8096f9d609c868bcd10  medicine_main_result_global_cache30_30_20_20.pdf
1eb9678437c7fe1e89b3a548cd9300b837922ae2e282bdc3a55f244b463a78e6  medicine_main_result_global_cache30_30_20_20.png
```
