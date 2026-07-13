#!/usr/bin/env python3
"""Summarize ACL terminology outcomes by reproducible surface-form types."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


LANGUAGES = ("de", "zh", "ja")
TERM_TYPES = ("acronym_or_symbolic_name", "multiword_expression", "single_word_term")
OUTCOMES = ("gain", "loss", "both_correct", "both_wrong")
GENUINE_ERROR_LABELS = {"wrong_translation", "omitted_term"}
METRIC_OR_BOUNDARY_LABELS = {
    "valid_alignment_boundary",
    "valid_compound_or_orthography",
    "valid_morphology",
    "valid_paraphrase",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--occurrences-root", type=Path, required=True)
    parser.add_argument("--lm", type=int, default=2)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--loss-audit-out", type=Path, required=True)
    parser.add_argument("--term-outcomes-out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_term(term: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", term)
    is_symbolic = any(re.fullmatch(r"[A-Z]{2,}[0-9]*", token) for token in tokens)
    is_symbolic = is_symbolic or any(
        any(character.isupper() for character in token[1:]) for token in tokens
    )
    is_symbolic = is_symbolic or any(character.isdigit() for character in term)
    if is_symbolic:
        return "acronym_or_symbolic_name"
    if len(tokens) >= 2:
        return "multiword_expression"
    return "single_word_term"


def classify_outcome(row: Mapping[str, str]) -> str:
    rasst_correct = row["exact_correct"] == "True"
    baseline_correct = row["baseline_exact_correct"] == "True"
    if rasst_correct and not baseline_correct:
        return "gain"
    if baseline_correct and not rasst_correct:
        return "loss"
    if rasst_correct:
        return "both_correct"
    return "both_wrong"


def load_occurrences(
    root: Path,
    lm: int,
) -> tuple[List[Dict[str, str]], Dict[str, Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    hashes: Dict[str, Dict[str, str]] = {}
    for language in LANGUAGES:
        path = root / language / "occurrences.tsv"
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[language] = {"path": str(path), "sha256": sha256_file(path)}
        with path.open(encoding="utf-8", newline="") as handle:
            language_rows = list(csv.DictReader(handle, delimiter="\t"))
        if not language_rows:
            raise ValueError(f"No occurrence rows in {path}")
        for row in language_rows:
            if row["dataset"] != "acl_tagged_raw" or row["lm"] != str(lm):
                raise ValueError(f"Unexpected dataset/lm row in {path}: {row}")
            if row["lang"] != language:
                raise ValueError(f"Unexpected language row in {path}: {row['lang']}")
            row["term_type"] = classify_term(row["term"])
            row["outcome"] = classify_outcome(row)
        rows.extend(language_rows)
    return rows, hashes


def write_tsv(path: Path, fieldnames: Iterable[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for language in (*LANGUAGES, "all"):
        language_rows = rows if language == "all" else [row for row in rows if row["lang"] == language]
        language_outcomes = Counter(row["outcome"] for row in language_rows)
        for term_type in TERM_TYPES:
            selected = [row for row in language_rows if row["term_type"] == term_type]
            outcomes = Counter(row["outcome"] for row in selected)
            total = len(selected)
            rasst_correct = outcomes["gain"] + outcomes["both_correct"]
            baseline_correct = outcomes["loss"] + outcomes["both_correct"]
            output.append(
                {
                    "language": language,
                    "term_type": term_type,
                    "occurrences": total,
                    "rasst_correct": rasst_correct,
                    "rasst_accuracy_pct": f"{100.0 * rasst_correct / total:.4f}",
                    "baseline_correct": baseline_correct,
                    "baseline_accuracy_pct": f"{100.0 * baseline_correct / total:.4f}",
                    "delta_percentage_points": f"{100.0 * (rasst_correct - baseline_correct) / total:.4f}",
                    "gain_rate_pct": f"{100.0 * outcomes['gain'] / total:.4f}",
                    "loss_rate_pct": f"{100.0 * outcomes['loss'] / total:.4f}",
                    "both_wrong_rate_pct": f"{100.0 * outcomes['both_wrong'] / total:.4f}",
                    "share_of_language_gains_pct": (
                        f"{100.0 * outcomes['gain'] / language_outcomes['gain']:.4f}"
                        if language_outcomes["gain"]
                        else "0.0000"
                    ),
                    "share_of_language_losses_pct": (
                        f"{100.0 * outcomes['loss'] / language_outcomes['loss']:.4f}"
                        if language_outcomes["loss"]
                        else "0.0000"
                    ),
                    **{outcome: outcomes[outcome] for outcome in OUTCOMES},
                }
            )
    return output


def build_loss_audit(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for language in (*LANGUAGES, "all"):
        language_rows = rows if language == "all" else [row for row in rows if row["lang"] == language]
        losses = [row for row in language_rows if row["outcome"] == "loss"]
        for term_type in TERM_TYPES:
            selected = [row for row in losses if row["term_type"] == term_type]
            labels = Counter(row["audit_label"] or "unlabeled" for row in selected)
            output.append(
                {
                    "language": language,
                    "term_type": term_type,
                    "exact_losses": len(selected),
                    "genuine_errors": sum(labels[label] for label in GENUINE_ERROR_LABELS),
                    "metric_or_boundary_false_negatives": sum(
                        labels[label] for label in METRIC_OR_BOUNDARY_LABELS
                    ),
                    "wrong_translation": labels["wrong_translation"],
                    "omitted_term": labels["omitted_term"],
                    "valid_paraphrase": labels["valid_paraphrase"],
                    "valid_morphology": labels["valid_morphology"],
                    "valid_compound_or_orthography": labels[
                        "valid_compound_or_orthography"
                    ],
                    "valid_alignment_boundary": labels["valid_alignment_boundary"],
                    "unlabeled": labels["unlabeled"],
                }
            )
    return output


def build_term_outcomes(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    languages: Dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (row["term_type"], row["term"])
        grouped[key][row["outcome"]] += 1
        languages[key].add(row["lang"])
    output: List[Dict[str, object]] = []
    for (term_type, term), counts in grouped.items():
        output.append(
            {
                "term_type": term_type,
                "term": term,
                "languages": ",".join(sorted(languages[(term_type, term)])),
                **{outcome: counts[outcome] for outcome in OUTCOMES},
                "net_gain_minus_loss": counts["gain"] - counts["loss"],
            }
        )
    return sorted(
        output,
        key=lambda row: (
            str(row["term_type"]),
            -int(row["net_gain_minus_loss"]),
            -int(row["gain"]),
            str(row["term"]).casefold(),
        ),
    )


def main() -> None:
    args = parse_args()
    rows, hashes = load_occurrences(args.occurrences_root, args.lm)
    summary = build_summary(rows)
    loss_audit = build_loss_audit(rows)
    term_outcomes = build_term_outcomes(rows)

    write_tsv(
        args.summary_out,
        (
            "language",
            "term_type",
            "occurrences",
            "rasst_correct",
            "rasst_accuracy_pct",
            "baseline_correct",
            "baseline_accuracy_pct",
            "delta_percentage_points",
            "gain_rate_pct",
            "loss_rate_pct",
            "both_wrong_rate_pct",
            "share_of_language_gains_pct",
            "share_of_language_losses_pct",
            *OUTCOMES,
        ),
        summary,
    )
    write_tsv(
        args.loss_audit_out,
        (
            "language",
            "term_type",
            "exact_losses",
            "genuine_errors",
            "metric_or_boundary_false_negatives",
            "wrong_translation",
            "omitted_term",
            "valid_paraphrase",
            "valid_morphology",
            "valid_compound_or_orthography",
            "valid_alignment_boundary",
            "unlabeled",
        ),
        loss_audit,
    )
    write_tsv(
        args.term_outcomes_out,
        (
            "term_type",
            "term",
            "languages",
            *OUTCOMES,
            "net_gain_minus_loss",
        ),
        term_outcomes,
    )

    # note (luojiaxuan): the taxonomy is deliberately based only on the English
    # glossary surface form so that the reviewer-facing aggregate can be exactly
    # reproduced without adding another subjective term-by-term annotation pass.
    manifest = {
        "dataset": "acl_tagged_raw",
        "languages": list(LANGUAGES),
        "lm": args.lm,
        "taxonomy": {
            "acronym_or_symbolic_name": (
                "contains an all-caps token, internal capitalization, or a digit"
            ),
            "multiword_expression": (
                "contains at least two alphanumeric tokens and is not symbolic"
            ),
            "single_word_term": "all remaining glossary terms",
        },
        "occurrence_rows": len(rows),
        "unique_source_terms": len({row["term"] for row in rows}),
        "input_sha256": hashes,
        "outputs": {
            "summary": str(args.summary_out),
            "loss_audit": str(args.loss_audit_out),
            "term_outcomes": str(args.term_outcomes_out),
        },
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
