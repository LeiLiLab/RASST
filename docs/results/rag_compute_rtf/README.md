# RAG Compute RTF

This directory tracks the RAG retriever compute real-time-factor figure used to
audit retrieval overhead as the streaming chunk/cadence changes.

## Files

| File | Contents |
| --- | --- |
| `data.tsv` | Frozen plotted summary for Medicine En-Zh hard/raw, `lm=1..4`. |
| `rag_compute_rtf.pdf/png` | RAG compute RTF figure. |
| `plot_rag_compute_rtf.py` | Paper-local plotting script that regenerates the local PDF/PNG from `data.tsv`. |

## Metric

The reported RTF is:

```text
RAG compute RTF = retriever call time / (0.96s * LM)
```

The retriever encodes the current vLLM generation span plus a fixed 1.92s
look-back. The plotted input spans are therefore 2.88s, 3.84s, 4.80s, and 5.76s
for `lm=1,2,3,4`.

## Summary

| LM | vLLM cadence | Retriever input span | Median retrieve time | Median RAG RTF |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.96s | 2.88s | 36.957 ms | 3.8497% |
| 2 | 1.92s | 3.84s | 42.345 ms | 2.2055% |
| 3 | 2.88s | 4.80s | 42.560 ms | 1.4778% |
| 4 | 3.84s | 5.76s | 43.645 ms | 1.1366% |

## Provenance

This package was copied from the frozen InfiniSST paper-local figure package:

```text
/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/plot/figure_03_rag_compute_rtf/
```

Original analysis manifest:

```text
/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/manifests/2026/05/20260525T0149__analysis__rag_compute_rtf_figure.json
```

The manifest records the source runtime traces under:

```text
/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_hn1024_tau078_new_v9_batch_20260524T0242/
```

## Checksums

```text
4cc153356f41f1b0ebf105565432129ed927b794937d6a73dc0635b6f54f6b0b  data.tsv
ab6f2151195c5130bd3744515c93f5ecbf6ff5c1d1e13ef9e6fc908f5099aff9  rag_compute_rtf.pdf
da445178665f7122f8b7eda63193906f6665f29ef240c0cf2b93745114ac5356  rag_compute_rtf.png
3abd389cffe938016ba6cb1bd352048590f24f2624870ac53f36fad1def3e245  plot_rag_compute_rtf.py
```
