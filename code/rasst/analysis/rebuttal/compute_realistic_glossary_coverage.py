#!/usr/bin/env python3
"""Measure exact overlap between paper-derived and ACL raw-gold glossaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


LANGUAGES = ("zh", "ja", "de")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-glossary", type=Path, required=True)
    parser.add_argument("--paper-glossary-dir", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--audio-yaml", type=Path, required=True)
    parser.add_argument(
        "--reference",
        action="append",
        required=True,
        help="Language/path pair, for example zh=/path/to/ref.txt",
    )
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--per-paper-out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_source(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def normalize_target(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def source_contains(source_text: str, term: str) -> bool:
    source_norm = " ".join(source_text.split()).casefold()
    term_norm = " ".join(term.split()).casefold()
    if not source_norm or not term_norm:
        return False
    if re.fullmatch(r"[a-z0-9][a-z0-9 ._+/#-]*", term_norm):
        pattern = r"(?<![a-z0-9])" + re.escape(term_norm) + r"(?![a-z0-9])"
        return re.search(pattern, source_norm) is not None
    return term_norm in source_norm


def load_glossary_entries(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values: Iterable[Any]
    if isinstance(data, list):
        values = data
    elif isinstance(data, dict):
        values = data.values()
    else:
        raise ValueError(f"Unsupported glossary root in {path}: {type(data).__name__}")
    entries = [dict(entry) for entry in values if isinstance(entry, dict)]
    if not entries:
        raise ValueError(f"No glossary entries in {path}")
    return entries


def load_paper_glossaries(directory: Path) -> Dict[str, List[Dict[str, Any]]]:
    glossaries: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(directory.glob("extracted_glossary__*.json")):
        paper_id = path.stem.split("__", 1)[1]
        glossaries[paper_id] = load_glossary_entries(path)
    if not glossaries:
        raise ValueError(f"No paper glossaries in {directory}")
    return glossaries


def load_wavs_from_audio_yaml(path: Path) -> List[str]:
    wavs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("wav:"):
            wavs.append(stripped.split("wav:", 1)[1].strip())
    if not wavs:
        raise ValueError(f"No wav rows found in {path}")
    return wavs


def parse_references(values: Sequence[str]) -> Dict[str, Path]:
    references: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --reference value: {value!r}")
        language, raw_path = value.split("=", 1)
        if language not in LANGUAGES or language in references:
            raise ValueError(f"Invalid or duplicate reference language: {language!r}")
        references[language] = Path(raw_path)
    if set(references) != set(LANGUAGES):
        raise ValueError(f"Expected references for {LANGUAGES}; received {sorted(references)}")
    return references


def percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        raise ValueError("Coverage denominator must be positive")
    return f"{100.0 * numerator / denominator:.4f}"


def write_tsv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
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


def main() -> None:
    args = parse_args()
    references = parse_references(args.reference)
    raw_entries = load_glossary_entries(args.raw_glossary)
    paper_glossaries = load_paper_glossaries(args.paper_glossary_dir)
    sources = args.source_file.read_text(encoding="utf-8").splitlines()
    wavs = load_wavs_from_audio_yaml(args.audio_yaml)
    reference_lines = {
        language: path.read_text(encoding="utf-8").splitlines()
        for language, path in references.items()
    }
    if len(sources) != len(wavs) or any(
        len(lines) != len(sources) for lines in reference_lines.values()
    ):
        raise ValueError("Aligned source/reference/audio lengths do not match")

    raw_by_source = {normalize_source(entry["term"]): entry for entry in raw_entries}
    extracted_union: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for paper_id, entries in paper_glossaries.items():
        for entry in entries:
            extracted_union[normalize_source(entry["term"])].append((paper_id, entry))
    source_overlap = set(raw_by_source) & set(extracted_union)

    summary_rows: List[Dict[str, Any]] = []
    per_paper_rows: List[Dict[str, Any]] = []
    for language in LANGUAGES:
        pair_overlap = {
            source
            for source in source_overlap
            if any(
                normalize_target(entry["target_translations"][language])
                == normalize_target(raw_by_source[source]["target_translations"][language])
                for _, entry in extracted_union[source]
            )
        }
        summary_rows.append(
            {
                "scope": "global_unique_raw_entries",
                "language": language,
                "denominator": len(raw_by_source),
                "source_match": len(source_overlap),
                "source_match_pct": percent(len(source_overlap), len(raw_by_source)),
                "source_target_pair_match": len(pair_overlap),
                "source_target_pair_match_pct": percent(len(pair_overlap), len(raw_by_source)),
                "extracted_entry_count": sum(len(rows) for rows in paper_glossaries.values()),
                "extracted_unique_source_count": len(extracted_union),
            }
        )

        occurrences: List[Dict[str, Any]] = []
        unique_gold_pairs: Dict[Tuple[str, str, str], Tuple[bool, bool]] = {}
        for source_text, reference, wav in zip(
            sources,
            reference_lines[language],
            wavs,
        ):
            paper_id = Path(wav).stem
            paper_by_source = {
                normalize_source(entry["term"]): entry
                for entry in paper_glossaries[paper_id]
            }
            for raw_entry in raw_entries:
                source_term = str(raw_entry["term"])
                target_term = str(raw_entry["target_translations"][language])
                if not source_contains(source_text, source_term) or target_term not in reference:
                    continue
                source_key = normalize_source(source_term)
                source_match = source_key in paper_by_source
                pair_match = source_match and (
                    normalize_target(
                        paper_by_source[source_key]["target_translations"][language]
                    )
                    == normalize_target(target_term)
                )
                occurrence = {
                    "paper_id": paper_id,
                    "source_match": source_match,
                    "pair_match": pair_match,
                }
                occurrences.append(occurrence)
                unique_gold_pairs[(paper_id, source_key, normalize_target(target_term))] = (
                    source_match,
                    pair_match,
                )

        occurrence_source = sum(row["source_match"] for row in occurrences)
        occurrence_pair = sum(row["pair_match"] for row in occurrences)
        summary_rows.append(
            {
                "scope": "talk_aware_gold_occurrences",
                "language": language,
                "denominator": len(occurrences),
                "source_match": occurrence_source,
                "source_match_pct": percent(occurrence_source, len(occurrences)),
                "source_target_pair_match": occurrence_pair,
                "source_target_pair_match_pct": percent(occurrence_pair, len(occurrences)),
                "extracted_entry_count": sum(len(rows) for rows in paper_glossaries.values()),
                "extracted_unique_source_count": len(extracted_union),
            }
        )
        unique_source = sum(value[0] for value in unique_gold_pairs.values())
        unique_pair = sum(value[1] for value in unique_gold_pairs.values())
        summary_rows.append(
            {
                "scope": "talk_aware_unique_gold_pairs",
                "language": language,
                "denominator": len(unique_gold_pairs),
                "source_match": unique_source,
                "source_match_pct": percent(unique_source, len(unique_gold_pairs)),
                "source_target_pair_match": unique_pair,
                "source_target_pair_match_pct": percent(unique_pair, len(unique_gold_pairs)),
                "extracted_entry_count": sum(len(rows) for rows in paper_glossaries.values()),
                "extracted_unique_source_count": len(extracted_union),
            }
        )

        for paper_id in sorted(paper_glossaries):
            selected = [row for row in occurrences if row["paper_id"] == paper_id]
            source_count = sum(row["source_match"] for row in selected)
            pair_count = sum(row["pair_match"] for row in selected)
            per_paper_rows.append(
                {
                    "language": language,
                    "paper_id": paper_id,
                    "gold_occurrences": len(selected),
                    "source_match": source_count,
                    "source_match_pct": percent(source_count, len(selected)),
                    "source_target_pair_match": pair_count,
                    "source_target_pair_match_pct": percent(pair_count, len(selected)),
                    "paper_glossary_entries": len(paper_glossaries[paper_id]),
                }
            )

    summary_fields = (
        "scope",
        "language",
        "denominator",
        "source_match",
        "source_match_pct",
        "source_target_pair_match",
        "source_target_pair_match_pct",
        "extracted_entry_count",
        "extracted_unique_source_count",
    )
    per_paper_fields = (
        "language",
        "paper_id",
        "gold_occurrences",
        "source_match",
        "source_match_pct",
        "source_target_pair_match",
        "source_target_pair_match_pct",
        "paper_glossary_entries",
    )
    write_tsv(args.summary_out, summary_fields, summary_rows)
    write_tsv(args.per_paper_out, per_paper_fields, per_paper_rows)

    # note (luojiaxuan): the global entry overlap is the reviewer-facing
    # glossary statistic. Talk-aware rows are stricter diagnostics because a
    # term extracted from one paper cannot be used for a different talk.
    manifest = {
        "dataset": "acl_tagged_raw",
        "normalization": {
            "source": "Unicode NFKC, whitespace collapse, casefold, exact equality",
            "target": "Unicode NFKC, whitespace collapse, exact equality",
            "gold_occurrence": (
                "source boundary match and exact raw-gold target substring in reference"
            ),
        },
        "inputs": {
            "raw_glossary": {
                "path": str(args.raw_glossary),
                "sha256": sha256_file(args.raw_glossary),
            },
            "paper_glossaries": {
                paper_id: {
                    "path": str(
                        args.paper_glossary_dir
                        / f"extracted_glossary__{paper_id}.json"
                    ),
                    "sha256": sha256_file(
                        args.paper_glossary_dir
                        / f"extracted_glossary__{paper_id}.json"
                    ),
                }
                for paper_id in sorted(paper_glossaries)
            },
            "source": {
                "path": str(args.source_file),
                "sha256": sha256_file(args.source_file),
            },
            "audio_yaml": {
                "path": str(args.audio_yaml),
                "sha256": sha256_file(args.audio_yaml),
            },
            "references": {
                language: {"path": str(path), "sha256": sha256_file(path)}
                for language, path in sorted(references.items())
            },
        },
        "counts": {
            "raw_unique_sources": len(raw_by_source),
            "paper_entries_total": sum(len(rows) for rows in paper_glossaries.values()),
            "paper_unique_sources": len(extracted_union),
            "global_source_overlap": len(source_overlap),
        },
        "outputs": {
            "summary": str(args.summary_out),
            "per_paper": str(args.per_paper_out),
        },
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
