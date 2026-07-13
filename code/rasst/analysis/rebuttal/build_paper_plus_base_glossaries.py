#!/usr/bin/env python3
"""Build per-paper runtime glossaries by overlaying paper terms on a base bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-glossary-dir", type=Path, required=True)
    parser.add_argument("--base-glossary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entries(data: Any) -> Iterable[Dict[str, Any]]:
    values: Iterable[Any]
    if isinstance(data, list):
        values = data
    elif isinstance(data, dict):
        values = data.values()
    else:
        raise ValueError(f"Unsupported glossary root: {type(data).__name__}")
    for entry in values:
        if isinstance(entry, dict):
            yield dict(entry)


def source_key(entry: Mapping[str, Any]) -> str:
    term = unicodedata.normalize("NFKC", str(entry.get("term") or ""))
    return " ".join(term.split()).casefold()


def validate_entry(entry: Mapping[str, Any], *, source: str) -> None:
    key = source_key(entry)
    translations = entry.get("target_translations")
    if not key or not isinstance(translations, dict):
        raise ValueError(f"Malformed {source} glossary entry: {entry!r}")
    missing = [language for language in ("zh", "ja", "de") if not translations.get(language)]
    if missing:
        raise ValueError(f"Missing translations {missing} in {source} entry: {entry!r}")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    base_data = json.loads(args.base_glossary.read_text(encoding="utf-8"))
    base_by_source: Dict[str, Dict[str, Any]] = {}
    for entry in entries(base_data):
        validate_entry(entry, source="base")
        key = source_key(entry)
        if key in base_by_source:
            raise ValueError(f"Duplicate normalized base term: {entry['term']!r}")
        copied = dict(entry)
        copied["runtime_source"] = "nlp_ai_cs_10k"
        base_by_source[key] = copied
    if len(base_by_source) != 10000:
        raise ValueError(f"Expected 10,000 unique base terms; found {len(base_by_source)}")

    output_records: Dict[str, Dict[str, Any]] = {}
    for paper_path in sorted(args.paper_glossary_dir.glob("extracted_glossary__*.json")):
        paper_id = paper_path.stem.split("__", 1)[1]
        paper_data = json.loads(paper_path.read_text(encoding="utf-8"))
        combined = dict(base_by_source)
        paper_count = 0
        paper_overrides = 0
        for entry in entries(paper_data):
            validate_entry(entry, source=paper_id)
            key = source_key(entry)
            paper_count += 1
            paper_overrides += int(key in combined)
            copied = dict(entry)
            copied["runtime_source"] = "paper_extracted_override"
            copied["runtime_paper_id"] = paper_id
            combined[key] = copied
        output_path = args.output_dir / f"extracted_glossary__{paper_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_records[paper_id] = {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "base_terms": len(base_by_source),
            "paper_terms": paper_count,
            "paper_overrides": paper_overrides,
            "final_terms": len(combined),
        }
    if len(output_records) != 5:
        raise ValueError(f"Expected five paper glossaries; found {len(output_records)}")

    # note (luojiaxuan): paper-specific entries deliberately override same-source
    # base entries so the index never contains two target translations for the
    # same normalized source term.
    manifest = {
        "schema_version": 1,
        "kind": "rasst_paper_plus_nlp_ai_cs_10k_glossaries",
        "merge_policy": "paper-specific source term overrides the base entry",
        "base_glossary": {
            "path": str(args.base_glossary),
            "sha256": sha256_file(args.base_glossary),
            "unique_terms": len(base_by_source),
        },
        "paper_glossary_dir": str(args.paper_glossary_dir),
        "outputs": output_records,
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
