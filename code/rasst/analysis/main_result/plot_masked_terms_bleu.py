#!/usr/bin/env python3
"""Plot target-term-masked BLEU for the final main-result snapshot."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[4]
RESULT_DIR = REPO_ROOT / "docs/results/main_result_global_cache30_30_20_20"
FIGURE_DIR = REPO_ROOT / "figures/main_result_global_cache30_30_20_20"
DEFAULT_MASKED_DATA = RESULT_DIR / "masked_terms_quality.tsv"
DEFAULT_MAIN_DATA = RESULT_DIR / "main_result.tsv"
DEFAULT_PREFIX = FIGURE_DIR / "masked_terms_bleu_global_cache30_30_20_20"
DEFAULT_DOCS_PREFIX = RESULT_DIR / "masked_terms_bleu_global_cache30_30_20_20"

DATASETS: Sequence[Tuple[str, str]] = (
    ("acl_tagged_raw", "ACL6060 tagged"),
    ("medicine_hardraw", "Medicine hard/raw"),
)
LANGS: Sequence[Tuple[str, str]] = (("zh", "En-Zh"), ("de", "En-De"), ("ja", "En-Ja"))
METHODS = ("InfiniSST", "RASST")

METHOD_STYLES = {
    "InfiniSST": {
        "color": "#2b6cb0",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 2.8,
        "markersize": 9.5,
    },
    "RASST": {
        "color": "#d62728",
        "marker": "*",
        "linestyle": "-",
        "linewidth": 3.0,
        "markersize": 13.5,
    },
}


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def finite(value: str) -> float | None:
    if value in {"", "NA"}:
        return None
    return float(value)


def row_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (row["dataset"], row["method"], row["lang"], row["lm"])


def merge_rows(
    masked_rows: Sequence[Dict[str, str]],
    main_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    main_by_key = {row_key(row): row for row in main_rows if row.get("lm") != "NA"}
    merged: List[Dict[str, str]] = []
    for row in masked_rows:
        base = main_by_key.get(row_key(row))
        if base is None:
            continue
        out = dict(row)
        out["StreamLAAL"] = base.get("StreamLAAL", "")
        merged.append(out)
    return merged


def plot_masked_bleu(rows: Sequence[Dict[str, str]], output_prefix: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 17,
            "legend.fontsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.4))
    handles: List[object] = []
    labels: List[str] = []

    for row_idx, (dataset, dataset_label) in enumerate(DATASETS):
        dataset_rows = [r for r in rows if r["dataset"] == dataset]
        for col, (lang, lang_label) in enumerate(LANGS):
            ax = axes[row_idx][col]
            lang_rows = [r for r in dataset_rows if r["lang"] == lang]
            x_values: List[float] = []
            y_values: List[float] = []
            plotted_methods = set()

            for method in METHODS:
                method_rows = [
                    r
                    for r in lang_rows
                    if r["method"] == method and r.get("status") == "ok"
                ]
                points: List[Tuple[float, float]] = []
                for item in sorted(
                    method_rows,
                    key=lambda value: int(value["lm"]) if value["lm"].isdigit() else 99,
                ):
                    x = finite(item.get("StreamLAAL", ""))
                    y = finite(item.get("MASKED_TERMS_BLEU", ""))
                    if x is None or y is None:
                        continue
                    points.append((x, y))
                if not points:
                    continue
                plotted_methods.add(method)
                line = ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    label=method,
                    **METHOD_STYLES[method],
                )[0]
                x_values.extend(point[0] for point in points)
                y_values.extend(point[1] for point in points)
                if method not in labels:
                    handles.append(line)
                    labels.append(method)

            if dataset == "acl_tagged_raw" and lang == "zh" and "InfiniSST" not in plotted_methods:
                ax.text(
                    0.03,
                    0.96,
                    "InfiniSST artifact\nunavailable",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=10.5,
                    color="#555555",
                )

            if x_values:
                x_low = min(x_values)
                x_high = max(x_values)
                pad = max((x_high - x_low) * 0.10, 90.0)
                ax.set_xlim(x_low - pad, x_high + pad)
            if y_values:
                y_low = min(y_values)
                y_high = max(y_values)
                pad = max((y_high - y_low) * 0.12, 1.1)
                ax.set_ylim(y_low - pad, y_high + pad)

            ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
            if row_idx == 0:
                ax.set_title(lang_label, fontweight="bold")
            if row_idx == len(DATASETS) - 1:
                ax.set_xlabel("StreamLAAL (ms)")
            if col == 0:
                ax.set_ylabel(f"{dataset_label}\nMasked-Term BLEU")

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            frameon=True,
            bbox_to_anchor=(0.5, 0.01),
            columnspacing=1.8,
            handlelength=2.4,
        )
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0), w_pad=1.25, h_pad=1.5)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masked-data", type=Path, default=DEFAULT_MASKED_DATA)
    parser.add_argument("--main-data", type=Path, default=DEFAULT_MAIN_DATA)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--docs-prefix", type=Path, default=DEFAULT_DOCS_PREFIX)
    args = parser.parse_args()

    rows = merge_rows(load_rows(args.masked_data), load_rows(args.main_data))
    plot_masked_bleu(rows, args.out_prefix)
    print(f"wrote {args.out_prefix.with_suffix('.pdf')}")
    print(f"wrote {args.out_prefix.with_suffix('.png')}")

    if args.docs_prefix:
        args.docs_prefix.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.out_prefix.with_suffix(".pdf"), args.docs_prefix.with_suffix(".pdf"))
        shutil.copy2(args.out_prefix.with_suffix(".png"), args.docs_prefix.with_suffix(".png"))
        print(f"updated {args.docs_prefix.with_suffix('.pdf')}")
        print(f"updated {args.docs_prefix.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
