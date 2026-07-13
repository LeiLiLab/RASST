#!/usr/bin/env python3
"""Prepare ACL shards from prebuilt per-paper runtime glossaries.

This entry point is for realistic glossary conditions assembled outside the
Gemini extraction runner, such as a paper-derived glossary overlaid on a fixed
10k NLP/AI/CS glossary. The ACL raw-gold glossary remains evaluation-only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from prepare_realistic_glossary_eval import (
    SUPPORTED_LANGUAGES,
    _file_record,
    _load_json,
    _normalise_text,
    _prepare_language_shards,
    _validate_gold_glossary,
    _write_json,
    sha256_file,
)


GLOSSARY_PATTERN = "extracted_glossary__{paper_id}.json"


def _validate_build_manifest(
    path: Path,
    *,
    paper_ids: Sequence[str],
) -> Dict[str, Any]:
    manifest = _load_json(path)
    if not isinstance(manifest, dict) or manifest.get("kind") != (
        "rasst_paper_plus_nlp_ai_cs_10k_glossaries"
    ):
        raise ValueError(f"Unsupported runtime glossary build manifest: {path}")
    base = manifest.get("base_glossary")
    if not isinstance(base, dict) or base.get("unique_terms") != 10_000:
        raise ValueError("Runtime glossary build manifest is not based on exactly 10k terms")
    base_path = Path(str(base.get("path") or ""))
    if sha256_file(base_path) != base.get("sha256"):
        raise ValueError(f"Base glossary hash mismatch: {base_path}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(paper_ids):
        raise ValueError("Runtime glossary build manifest paper set mismatch")
    for paper_id in paper_ids:
        record = outputs[paper_id]
        source_path = Path(str(record.get("path") or ""))
        if sha256_file(source_path) != record.get("sha256"):
            raise ValueError(f"Runtime glossary hash mismatch: {source_path}")
        if int(record.get("final_terms", 0)) < 10_000:
            raise ValueError(f"Runtime glossary has fewer than 10k terms: {paper_id}")
    return manifest


def _language_glossary(
    data: Any,
    *,
    language: str,
    paper_id: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Runtime glossary must be a non-empty object: {paper_id}")
    output: Dict[str, Dict[str, Any]] = {}
    source_counts: Dict[str, int] = {}
    for original_key, value in data.items():
        if not isinstance(value, dict):
            raise ValueError(f"Malformed runtime glossary entry: {paper_id}/{original_key}")
        term = _normalise_text(value.get("term") or original_key)
        translations = value.get("target_translations")
        translation = (
            _normalise_text(translations.get(language))
            if isinstance(translations, dict)
            else ""
        )
        if not term or not translation:
            raise ValueError(
                f"Missing term or {language} translation: {paper_id}/{original_key}"
            )
        key = term.casefold()
        if key in output:
            raise ValueError(f"Duplicate normalized term: {paper_id}/{term}")
        runtime_source = _normalise_text(value.get("runtime_source") or "unspecified")
        source_counts[runtime_source] = source_counts.get(runtime_source, 0) + 1
        output[key] = {
            "term": term,
            "target_translations": {language: translation},
            "runtime_source": runtime_source,
            "source_paper": paper_id,
        }
    return output, source_counts


def prepare(
    *,
    runtime_glossary_dir: Path,
    runtime_glossary_build_manifest_path: Path,
    release_data_root: Path,
    gold_glossary_path: Path,
    output_dir: Path,
    paper_ids: Sequence[str],
    languages: Sequence[str],
    runtime_glossary_policy: str,
    runtime_glossary_tag: str,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    build_manifest = _validate_build_manifest(
        runtime_glossary_build_manifest_path,
        paper_ids=paper_ids,
    )
    gold_record = _validate_gold_glossary(gold_glossary_path, languages)

    source_glossaries: Dict[str, Path] = {}
    glossary_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for paper_id in paper_ids:
        source_path = runtime_glossary_dir / GLOSSARY_PATTERN.format(paper_id=paper_id)
        expected = build_manifest["outputs"][paper_id]
        if sha256_file(source_path) != expected["sha256"]:
            raise ValueError(f"Runtime glossary differs from build manifest: {source_path}")
        source_glossaries[paper_id] = source_path
        data = _load_json(source_path)
        for language in languages:
            runtime_glossary, source_counts = _language_glossary(
                data,
                language=language,
                paper_id=paper_id,
            )
            output_path = output_dir / "runtime_glossaries" / language / f"{paper_id}.json"
            _write_json(output_path, runtime_glossary)
            glossary_records[(language, paper_id)] = {
                **_file_record(output_path),
                "term_count": len(runtime_glossary),
                "runtime_source_counts": source_counts,
                "language": language,
                "paper_id": paper_id,
                "source_runtime_glossary": _file_record(source_path),
            }

    release_records: Dict[str, Any] = {}
    shards = []
    for language in languages:
        release_record, shard_records = _prepare_language_shards(
            release_root=release_data_root,
            output_dir=output_dir,
            language=language,
            paper_ids=paper_ids,
        )
        release_records[language] = release_record
        for paper_id in paper_ids:
            shards.append(
                {
                    **shard_records[paper_id],
                    "language": language,
                    "runtime_glossary": glossary_records[(language, paper_id)],
                    "eval_glossary": gold_record,
                }
            )

    manifest = {
        "schema_version": 1,
        "kind": "rasst_acl_realistic_paper_glossary_prepared",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_ids": list(paper_ids),
        "languages": list(languages),
        "runtime_glossary_policy": runtime_glossary_policy,
        "runtime_glossary_tag": runtime_glossary_tag,
        "runtime_glossary_build_manifest": {
            "path": str(runtime_glossary_build_manifest_path.absolute()),
            "sha256": sha256_file(runtime_glossary_build_manifest_path),
        },
        "source_runtime_glossaries": {
            paper_id: _file_record(source_glossaries[paper_id]) for paper_id in paper_ids
        },
        "release_data_root": str(release_data_root.absolute()),
        "release_inputs": release_records,
        "fixed_raw_gold_eval_glossary": gold_record,
        "separation_policy": {
            "runtime_glossary_source": runtime_glossary_policy,
            "evaluation_denominator": "existing raw gold ACL glossary",
            "gold_used_to_build_runtime_glossary": False,
            "aggregation": (
                "concatenate the five paper outputs in canonical talk order and rerun "
                "offline ACL evaluation against the full raw-gold glossary"
            ),
        },
        "shards": shards,
    }
    manifest_path = output_dir / "prepared_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-glossary-dir", required=True, type=Path)
    parser.add_argument("--runtime-glossary-build-manifest", required=True, type=Path)
    parser.add_argument("--release-data-root", required=True, type=Path)
    parser.add_argument("--gold-glossary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--paper-ids", nargs="+", required=True)
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=SUPPORTED_LANGUAGES,
        default=list(SUPPORTED_LANGUAGES),
    )
    parser.add_argument("--runtime-glossary-policy", required=True)
    parser.add_argument("--runtime-glossary-tag", required=True)
    args = parser.parse_args()
    if len(args.paper_ids) != len(set(args.paper_ids)):
        raise ValueError("--paper-ids contains duplicates")
    if len(args.languages) != len(set(args.languages)):
        raise ValueError("--languages contains duplicates")
    manifest_path = prepare(
        runtime_glossary_dir=args.runtime_glossary_dir,
        runtime_glossary_build_manifest_path=args.runtime_glossary_build_manifest,
        release_data_root=args.release_data_root,
        gold_glossary_path=args.gold_glossary,
        output_dir=args.output_dir,
        paper_ids=args.paper_ids,
        languages=args.languages,
        runtime_glossary_policy=args.runtime_glossary_policy,
        runtime_glossary_tag=args.runtime_glossary_tag,
    )
    print(json.dumps({"prepared_manifest": str(manifest_path.absolute())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
