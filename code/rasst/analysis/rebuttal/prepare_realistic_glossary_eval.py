#!/usr/bin/env python3
"""Prepare paper-specific Gemini glossaries and ACL evaluation shards.

The runtime glossary is derived only from the Gemini extraction artifact.  The
existing tagged ACL glossary is carried separately as the fixed evaluation
denominator and is never consulted while constructing runtime glossaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SUPPORTED_LANGUAGES = ("zh", "de", "ja")
SOURCE_LIST_NAME = {
    "zh": "source.list",
    "de": "source.portable.list",
    "ja": "source.portable.list",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split())


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value[:1] in {'"', "'"}:
        if value[:1] == '"':
            return json.loads(value)
        if value[-1:] != "'":
            raise ValueError(f"Unterminated YAML string: {value!r}")
        return value[1:-1].replace("''", "'")
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_flat_audio_yaml(text: str, *, source: str = "<memory>") -> List[Dict[str, Any]]:
    """Parse the release audio YAML's strict list-of-flat-mappings schema."""
    rows: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        if not raw_line.strip():
            continue
        if raw_line.lstrip().startswith("#"):
            continue
        if raw_line.startswith("- "):
            if current is not None:
                rows.append(current)
            current = {}
            payload = raw_line[2:]
        elif raw_line.startswith("  ") and current is not None:
            payload = raw_line.strip()
        else:
            raise ValueError(f"Unsupported audio YAML structure at {source}:{line_number}")
        if ":" not in payload:
            raise ValueError(f"Malformed audio YAML mapping at {source}:{line_number}")
        key, raw_value = payload.split(":", 1)
        key = key.strip()
        if not key or key in current:
            raise ValueError(f"Missing or duplicate YAML key at {source}:{line_number}: {key!r}")
        current[key] = _parse_yaml_scalar(raw_value)
    if current is not None:
        rows.append(current)
    if not rows:
        raise ValueError(f"Audio YAML is empty: {source}")
    for index, row in enumerate(rows):
        if not _normalise_text(row.get("wav")):
            raise ValueError(f"Audio YAML row {index} has no wav: {source}")
    return rows


def load_flat_audio_yaml(path: Path) -> List[Dict[str, Any]]:
    return parse_flat_audio_yaml(path.read_text(encoding="utf-8"), source=str(path))


def _read_nonempty_lines(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing text file: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"Expected non-empty lines only: {path}")
    return lines


def _paper_id_from_wav(value: Any) -> str:
    paper_id = Path(_normalise_text(value)).stem
    if not paper_id:
        raise ValueError(f"Cannot infer paper ID from wav value: {value!r}")
    return paper_id


def _file_record(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty file: {path}")
    return {
        "path": str(path.absolute()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _validate_extraction_manifest(
    path: Path,
    *,
    paper_ids: Sequence[str],
    expected_model: str,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    manifest = _load_json(path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Extraction manifest must be an object: {path}")
    if manifest.get("model") != expected_model:
        raise ValueError(
            f"Gemini model mismatch: expected {expected_model!r}, got {manifest.get('model')!r}"
        )
    model_info = manifest.get("model_metadata")
    if (
        not isinstance(model_info, dict)
        or model_info.get("lookup_status") != "ok"
        or not _normalise_text(model_info.get("version"))
    ):
        raise ValueError("Extraction manifest lacks verified Gemini model-version metadata")
    if manifest.get("sdk") != "google-genai" or not _normalise_text(
        manifest.get("sdk_version")
    ):
        raise ValueError("Extraction manifest lacks google-genai SDK provenance")
    if not _normalise_text(manifest.get("prompt_sha256")):
        raise ValueError("Extraction manifest lacks a prompt hash")
    policy = manifest.get("data_access_policy")
    if not isinstance(policy, dict) or policy.get("manual_filtering") is not False:
        raise ValueError("Extraction manifest must explicitly record manual_filtering=false")
    excluded = {str(value).casefold() for value in policy.get("excluded", [])}
    if not any("gold" in value and "glossary" in value for value in excluded):
        raise ValueError("Extraction manifest does not exclude the gold evaluation glossary")

    rows = manifest.get("papers")
    if not isinstance(rows, list):
        raise ValueError("Extraction manifest papers must be a list")
    by_paper: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Malformed extraction paper row: {row!r}")
        paper_id = _normalise_text(row.get("paper_id"))
        if not paper_id or paper_id in by_paper:
            raise ValueError(f"Missing or duplicate extraction paper ID: {paper_id!r}")
        glossary_path = Path(str(row.get("glossary_path") or ""))
        response_path = Path(str(row.get("raw_responses_path") or ""))
        for artifact_path, hash_field in (
            (glossary_path, "glossary_sha256"),
            (response_path, "raw_responses_sha256"),
        ):
            expected_hash = _normalise_text(row.get(hash_field))
            if not expected_hash or sha256_file(artifact_path) != expected_hash:
                raise ValueError(f"Extraction artifact hash mismatch: {artifact_path}")
        by_paper[paper_id] = row
    requested = list(paper_ids)
    if set(by_paper) != set(requested):
        raise ValueError(
            f"Extraction paper set mismatch: requested={sorted(requested)}, "
            f"manifest={sorted(by_paper)}"
        )
    return manifest, by_paper


def build_language_glossary(
    glossary_data: Any,
    *,
    paper_id: str,
    language: str,
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(glossary_data, dict):
        raise ValueError(f"Gemini glossary for {paper_id} must be an object")
    output: Dict[str, Dict[str, Any]] = {}
    for original_key, value in glossary_data.items():
        if not isinstance(value, dict):
            raise ValueError(f"Malformed Gemini glossary entry {original_key!r}: {value!r}")
        term = _normalise_text(value.get("term") or original_key)
        translations = value.get("target_translations")
        if not term or not isinstance(translations, dict):
            raise ValueError(f"Malformed Gemini glossary entry {original_key!r}")
        translation = _normalise_text(translations.get(language))
        if not translation:
            raise ValueError(f"Missing {language} translation for {paper_id}/{term}")
        key = term.casefold()
        if key in output:
            raise ValueError(f"Duplicate normalized term for {paper_id}: {term}")
        output[key] = {
            "term": term,
            "target_translations": {language: translation},
            "source": "gemini_paper_extracted",
            "source_paper": paper_id,
        }
    if not output:
        raise ValueError(f"No glossary terms for {paper_id}/{language}")
    return output


def _validate_gold_glossary(path: Path, languages: Sequence[str]) -> Dict[str, Any]:
    data = _load_json(path)
    if not isinstance(data, (dict, list)) or not data:
        raise ValueError(f"Gold glossary must be a non-empty object or array: {path}")
    seen = {language: 0 for language in languages}
    values: Iterable[Any] = data.values() if isinstance(data, dict) else data
    for entry in values:
        if not isinstance(entry, dict):
            continue
        translations = entry.get("target_translations")
        if not isinstance(translations, dict):
            continue
        for language in languages:
            if _normalise_text(translations.get(language)):
                seen[language] += 1
    missing = [language for language, count in seen.items() if count == 0]
    if missing:
        raise ValueError(f"Gold glossary has no translations for: {missing}")
    return {"path": str(path.absolute()), "sha256": sha256_file(path), "translation_rows": seen}


def _release_language_inputs(release_root: Path, language: str) -> Dict[str, Path]:
    input_dir = release_root / "main_result" / "inputs" / f"acl_{language}"
    paths = {
        "source_list": input_dir / SOURCE_LIST_NAME[language],
        "target_list": input_dir / "target.list",
        "source_text": input_dir / "source_text.txt",
        "ref": input_dir / "ref.txt",
        "audio_yaml": input_dir / "audio.yaml",
    }
    for path in paths.values():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing release input for {language}: {path}")
    return paths


def _prepare_language_shards(
    *,
    release_root: Path,
    output_dir: Path,
    language: str,
    paper_ids: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    source_paths = _release_language_inputs(release_root, language)
    source_lines = _read_nonempty_lines(source_paths["source_list"])
    target_lines = _read_nonempty_lines(source_paths["target_list"])
    source_texts = _read_nonempty_lines(source_paths["source_text"])
    refs = _read_nonempty_lines(source_paths["ref"])
    audio_rows = load_flat_audio_yaml(source_paths["audio_yaml"])
    if len(source_lines) != len(target_lines):
        raise ValueError(f"source/target talk mismatch for {language}")
    if len(source_texts) != len(refs) or len(refs) != len(audio_rows):
        raise ValueError(
            f"sentence input mismatch for {language}: source={len(source_texts)}, "
            f"ref={len(refs)}, audio={len(audio_rows)}"
        )

    source_order = [_paper_id_from_wav(line) for line in source_lines]
    if source_order != list(paper_ids):
        raise ValueError(
            f"Canonical paper order mismatch for {language}: "
            f"expected={list(paper_ids)}, observed={source_order}"
        )
    audio_root = release_root / "main_result" / "audio" / "acl6060"
    full_audio_by_paper: Dict[str, Path] = {}
    for paper_id in paper_ids:
        full_audio = audio_root / f"{paper_id}.wav"
        if not full_audio.is_file() or full_audio.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing full-talk audio: {full_audio}")
        full_audio_by_paper[paper_id] = full_audio.absolute()

    indices_by_paper: Dict[str, List[int]] = defaultdict(list)
    rewritten_audio: List[Dict[str, Any]] = []
    for index, row in enumerate(audio_rows):
        paper_id = _paper_id_from_wav(row["wav"])
        if paper_id not in full_audio_by_paper:
            raise ValueError(f"Unexpected paper in {language} audio YAML: {paper_id}")
        copied = dict(row)
        copied["wav"] = str(full_audio_by_paper[paper_id])
        rewritten_audio.append(copied)
        indices_by_paper[paper_id].append(index)
    if set(indices_by_paper) != set(paper_ids):
        raise ValueError(f"Audio YAML paper set mismatch for {language}")

    shard_records: Dict[str, Dict[str, Any]] = {}
    for talk_index, paper_id in enumerate(paper_ids):
        indices = indices_by_paper[paper_id]
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"Non-contiguous sentence rows for {language}/{paper_id}")
        shard_dir = output_dir / "inputs" / language / paper_id
        shard_dir.mkdir(parents=True, exist_ok=True)
        output_paths = {
            "source_list": shard_dir / "source.list",
            "target_list": shard_dir / "target.list",
            "source_text": shard_dir / "source_text.txt",
            "ref": shard_dir / "ref.txt",
            "audio_yaml": shard_dir / "audio.yaml",
        }
        output_paths["source_list"].write_text(
            str(full_audio_by_paper[paper_id]) + "\n", encoding="utf-8"
        )
        output_paths["target_list"].write_text(target_lines[talk_index] + "\n", encoding="utf-8")
        output_paths["source_text"].write_text(
            "\n".join(source_texts[index] for index in indices) + "\n", encoding="utf-8"
        )
        output_paths["ref"].write_text(
            "\n".join(refs[index] for index in indices) + "\n", encoding="utf-8"
        )
        # note (luojiaxuan): JSON is valid YAML and avoids a YAML-dumper dependency.
        output_paths["audio_yaml"].write_text(
            json.dumps(
                [rewritten_audio[index] for index in indices],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        shard_records[paper_id] = {
            "paper_id": paper_id,
            "canonical_talk_index": talk_index,
            "sentence_start": indices[0],
            "sentence_end_exclusive": indices[-1] + 1,
            "sentence_count": len(indices),
            "full_audio": _file_record(full_audio_by_paper[paper_id]),
            "files": {name: _file_record(path) for name, path in output_paths.items()},
        }

    release_record = {
        "language": language,
        "canonical_paper_order": source_order,
        "sentence_count": len(audio_rows),
        "source_files": {name: _file_record(path) for name, path in source_paths.items()},
    }
    return release_record, shard_records


def prepare(
    *,
    extraction_manifest_path: Path,
    release_data_root: Path,
    gold_glossary_path: Path,
    output_dir: Path,
    paper_ids: Sequence[str],
    languages: Sequence[str],
    expected_model: str,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    extraction_manifest, extraction_by_paper = _validate_extraction_manifest(
        extraction_manifest_path,
        paper_ids=paper_ids,
        expected_model=expected_model,
    )
    gold_record = _validate_gold_glossary(gold_glossary_path, languages)

    glossary_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for paper_id in paper_ids:
        source_glossary_path = Path(extraction_by_paper[paper_id]["glossary_path"])
        source_glossary = _load_json(source_glossary_path)
        for language in languages:
            runtime_glossary = build_language_glossary(
                source_glossary,
                paper_id=paper_id,
                language=language,
            )
            output_path = output_dir / "runtime_glossaries" / language / f"{paper_id}.json"
            _write_json(output_path, runtime_glossary)
            glossary_records[(language, paper_id)] = {
                **_file_record(output_path),
                "term_count": len(runtime_glossary),
                "language": language,
                "paper_id": paper_id,
                "source_extraction_glossary": _file_record(source_glossary_path),
            }

    release_records: Dict[str, Any] = {}
    shards: List[Dict[str, Any]] = []
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
        "gemini_model": expected_model,
        "extraction_manifest": {
            "path": str(extraction_manifest_path.absolute()),
            "sha256": sha256_file(extraction_manifest_path),
            "prompt_sha256": extraction_manifest.get("prompt_sha256"),
            "sdk": extraction_manifest.get("sdk"),
            "sdk_version": extraction_manifest.get("sdk_version"),
        },
        "release_data_root": str(release_data_root.absolute()),
        "release_inputs": release_records,
        "fixed_raw_gold_eval_glossary": gold_record,
        "separation_policy": {
            "runtime_glossary_source": "Gemini extraction from the associated paper PDF only",
            "evaluation_denominator": "existing raw gold ACL glossary",
            "gold_used_to_build_runtime_glossary": False,
            "aggregation": (
                "concatenate the five paper outputs in canonical talk order and rerun "
                "offline ACL evaluation against the full raw-gold glossary"
            ),
        },
        "shards": shards,
        "logical_cells": [
            {"language": language, "latency_multiplier": lm, "paper_count": len(paper_ids)}
            for language in languages
            for lm in (1, 2, 3, 4)
        ],
    }
    manifest_path = output_dir / "prepared_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-manifest", required=True, type=Path)
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
    parser.add_argument("--expected-model", default="gemini-2.5-flash")
    args = parser.parse_args()
    if len(args.paper_ids) != len(set(args.paper_ids)):
        raise ValueError("--paper-ids contains duplicates")
    if len(args.languages) != len(set(args.languages)):
        raise ValueError("--languages contains duplicates")
    manifest_path = prepare(
        extraction_manifest_path=args.extraction_manifest,
        release_data_root=args.release_data_root,
        gold_glossary_path=args.gold_glossary,
        output_dir=args.output_dir,
        paper_ids=args.paper_ids,
        languages=args.languages,
        expected_model=args.expected_model,
    )
    print(json.dumps({"prepared_manifest": str(manifest_path.absolute())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
