#!/usr/bin/env python3
"""Independently validate WMT25-style Gemini LLM-judge artifacts.

The validator does not import the scorer or reuse its aggregation helpers.  It
reconstructs every request, joins raw Batch responses to the immutable
sidecars, checks the original source records, and recomputes all published
tables from the collected segment scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "rasst-gemini-llm-judge-wmt25-v1"
VALIDATION_SCHEMA_VERSION = "rasst-gemini-llm-judge-wmt25-validation-v1"
SOURCE_LANGUAGE_NAME = "English"
TARGET_LANGUAGE_NAMES = {"zh": "Chinese", "de": "German", "ja": "Japanese"}
METHODS = ("InfiniSST", "RASST")
LM_SETTINGS = ("1", "2", "3", "4")
ACL_DATASET = "acl_tagged_raw"
MEDICINE_DATASET = "medicine_hardraw"
EXPECTED_SEGMENTS_PER_SYSTEM = {ACL_DATASET: 468, MEDICINE_DATASET: 1_437}
EXPECTED_TALKS_PER_SYSTEM = 5
EXPECTED_SYSTEMS = 32
EXPECTED_PAIRS = 16
EXPECTED_SEGMENTS = 22_728
EXPECTED_ACL_SEGMENTS = 11_232
EXPECTED_MEDICINE_SEGMENTS = 11_496
EXPECTED_TALK_PAIRS = EXPECTED_PAIRS * EXPECTED_TALKS_PER_SYSTEM
DEFAULT_ABSOLUTE_TOLERANCE = 1e-9
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SCORE_RE = re.compile(r"^(?:100|[0-9]{1,2})$")

PROMPT_TEMPLATE = """Score the following translation from {source_lang} to
{target_lang} on a scale from 0 to 100, where a score of 0 means a
broken or poor translation; 33 indicates a flawed translation
with significant issues; 66 indicates a good translation with
only minor issues in grammar, fluency, or consistency; and 100
represents a perfect translation in both meaning and grammar.
Answer with only a whole number representing the score, and
nothing else.
{source_lang} source text:
{source_seg}
{target_lang} translation:
{target_seg}"""

SYSTEM_KEY_FIELDS = ("dataset", "method", "lang", "lm")
PAIR_KEY_FIELDS = ("dataset", "lang", "lm")
TALK_KEY_FIELDS = ("dataset", "lang", "lm", "talk_id")
SUMMARY_FIELDS = (
    *SYSTEM_KEY_FIELDS,
    "talks",
    "segments",
    "llm_judge_mean",
    "llm_judge_talk_macro_mean",
    "judge_model",
    "resolved_model_version",
    "prompt_sha256",
    "generation_config_sha256",
    "run_config_sha256",
)
PAIRED_FIELDS = (
    *PAIR_KEY_FIELDS,
    "rasst_method",
    "infinisst_method",
    "paired_talks",
    "paired_segments",
    "rasst_llm_judge_mean",
    "infinisst_llm_judge_mean",
    "delta_rasst_minus_infinisst",
    "paired_delta_stddev",
    "rasst_wins",
    "ties",
    "infinisst_wins",
)
TALK_FIELDS = (
    *TALK_KEY_FIELDS,
    "paired_segments",
    "rasst_llm_judge_mean",
    "infinisst_llm_judge_mean",
    "delta_rasst_minus_infinisst",
)
GROUP_FIELDS = (
    "group",
    "cells",
    "rasst_cell_macro_mean",
    "infinisst_cell_macro_mean",
    "delta_cell_macro_rasst_minus_infinisst",
    "positive_cells",
    "zero_cells",
    "negative_cells",
)


class GeminiJudgeValidationError(RuntimeError):
    """Raised when an LLM-judge artifact is incomplete or inconsistent."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def lexical_absolute(path: Path) -> Path:
    """Make a path absolute without collapsing host-qualified mount aliases."""

    return Path(os.path.abspath(os.fspath(path)))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def _require_file(path: Path, *, label: str) -> Path:
    lexical = lexical_absolute(path)
    if not lexical.is_file():
        raise GeminiJudgeValidationError(f"{label} is not a file: {lexical}")
    return lexical


def _read_json(path: Path, *, label: str) -> Dict[str, Any]:
    path = _require_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GeminiJudgeValidationError(f"Cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GeminiJudgeValidationError(f"{label} must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> List[Dict[str, Any]]:
    path = _require_file(path, label=label)
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise GeminiJudgeValidationError(
                        f"{label} contains a blank line at {path}:{line_number}"
                    )
                try:
                    row = json.loads(line, parse_constant=_reject_json_constant)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise GeminiJudgeValidationError(
                        f"Invalid JSON in {label} at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise GeminiJudgeValidationError(
                        f"Expected an object in {label} at {path}:{line_number}"
                    )
                rows.append(row)
    except (OSError, UnicodeDecodeError) as exc:
        raise GeminiJudgeValidationError(f"Cannot read {label} {path}: {exc}") from exc
    if not rows:
        raise GeminiJudgeValidationError(f"{label} contains no records: {path}")
    return rows


def _read_tsv(
    path: Path, *, label: str, required_fields: Sequence[str]
) -> List[Dict[str, str]]:
    path = _require_file(path, label=label)
    rows: List[Dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise GeminiJudgeValidationError(f"{label} has no header: {path}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise GeminiJudgeValidationError(f"{label} has duplicate columns: {path}")
            missing = [field for field in required_fields if field not in reader.fieldnames]
            if missing:
                raise GeminiJudgeValidationError(
                    f"{label} is missing columns: {', '.join(missing)}"
                )
            for line_number, raw_row in enumerate(reader, start=2):
                if None in raw_row:
                    raise GeminiJudgeValidationError(
                        f"{label} has extra columns at {path}:{line_number}"
                    )
                row = {str(key): str(value or "") for key, value in raw_row.items()}
                if not any(value.strip() for value in row.values()):
                    raise GeminiJudgeValidationError(
                        f"{label} contains a blank row at {path}:{line_number}"
                    )
                rows.append(row)
    except UnicodeDecodeError as exc:
        raise GeminiJudgeValidationError(f"{label} is not valid UTF-8: {path}") from exc
    if not rows:
        raise GeminiJudgeValidationError(f"{label} contains no rows: {path}")
    return rows


def _nonempty(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GeminiJudgeValidationError(f"{context} must be a non-empty string")
    return value


def _sha256(value: Any, *, context: str) -> str:
    cleaned = _nonempty(value, context=context)
    if SHA256_RE.fullmatch(cleaned) is None:
        raise GeminiJudgeValidationError(f"{context} must be a lowercase SHA-256")
    return cleaned


def _canonical_nonnegative_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        raise GeminiJudgeValidationError(f"{context} must be an integer, got bool")
    text = str(value)
    try:
        parsed = int(text)
    except (TypeError, ValueError) as exc:
        raise GeminiJudgeValidationError(f"{context} must be an integer: {value!r}") from exc
    if parsed < 0 or str(parsed) != text:
        raise GeminiJudgeValidationError(
            f"{context} must be a canonical non-negative integer: {value!r}"
        )
    return parsed


def _positive_int(value: Any, *, context: str) -> int:
    parsed = _canonical_nonnegative_int(value, context=context)
    if parsed == 0:
        raise GeminiJudgeValidationError(f"{context} must be positive")
    return parsed


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise GeminiJudgeValidationError(f"{context} must be finite, got bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GeminiJudgeValidationError(f"{context} must be finite: {value!r}") from exc
    if not math.isfinite(parsed):
        raise GeminiJudgeValidationError(f"{context} must be finite: {value!r}")
    return parsed


def _assert_equal(actual: Any, expected: Any, *, context: str) -> None:
    if actual != expected:
        raise GeminiJudgeValidationError(
            f"{context} mismatch: actual={actual!r}, expected={expected!r}"
        )


def _assert_close(
    actual: Any, expected: float, *, context: str, tolerance: float
) -> None:
    parsed = _finite_float(actual, context=context)
    if not math.isclose(parsed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise GeminiJudgeValidationError(
            f"{context} mismatch: reported={parsed:.17g}, "
            f"recomputed={expected:.17g}, tolerance={tolerance:.3g}"
        )


def _index_unique(
    rows: Iterable[Mapping[str, Any]], fields: Sequence[str], *, label: str
) -> Dict[Tuple[str, ...], Mapping[str, Any]]:
    indexed: Dict[Tuple[str, ...], Mapping[str, Any]] = {}
    for number, row in enumerate(rows, start=1):
        key = tuple(_nonempty(row.get(field), context=f"{label} row {number}.{field}") for field in fields)
        if key in indexed:
            raise GeminiJudgeValidationError(f"Duplicate {label} key: {key!r}")
        indexed[key] = row
    return indexed


def _artifact_path(output_dir: Path, relative: Any, *, label: str) -> Path:
    text = _nonempty(relative, context=f"{label} path")
    candidate = Path(text)
    if candidate.is_absolute():
        raise GeminiJudgeValidationError(f"{label} path must be relative: {text!r}")
    lexical = lexical_absolute(output_dir / candidate)
    try:
        lexical.relative_to(output_dir)
    except ValueError as exc:
        raise GeminiJudgeValidationError(f"{label} path escapes output directory: {text!r}") from exc
    # note (luojiaxuan): The lexical check preserves /mnt/taurus/... in
    # provenance, while this real-path check prevents an in-tree symlink from
    # escaping the validated output directory.
    real_output = output_dir.resolve()
    real_candidate = lexical.resolve()
    try:
        real_candidate.relative_to(real_output)
    except ValueError as exc:
        raise GeminiJudgeValidationError(
            f"{label} resolves outside output directory: {text!r}"
        ) from exc
    return _require_file(lexical, label=label)


def _expected_system_counts() -> Dict[Tuple[str, str, str, str], int]:
    counts: Dict[Tuple[str, str, str, str], int] = {}
    for method in METHODS:
        for lang in TARGET_LANGUAGE_NAMES:
            for lm in LM_SETTINGS:
                counts[(ACL_DATASET, method, lang, lm)] = EXPECTED_SEGMENTS_PER_SYSTEM[ACL_DATASET]
        for lm in LM_SETTINGS:
            counts[(MEDICINE_DATASET, method, "de", lm)] = EXPECTED_SEGMENTS_PER_SYSTEM[
                MEDICINE_DATASET
            ]
    return counts


def _expected_shards() -> Dict[str, Tuple[str, str, str, int]]:
    shards: Dict[str, Tuple[str, str, str, int]] = {}
    for lang in TARGET_LANGUAGE_NAMES:
        for lm in LM_SETTINGS:
            shard_id = f"{ACL_DATASET}__{lang}__lm{lm}"
            shards[shard_id] = (
                ACL_DATASET,
                lang,
                lm,
                2 * EXPECTED_SEGMENTS_PER_SYSTEM[ACL_DATASET],
            )
    for lm in LM_SETTINGS:
        shard_id = f"{MEDICINE_DATASET}__de__lm{lm}"
        shards[shard_id] = (
            MEDICINE_DATASET,
            "de",
            lm,
            2 * EXPECTED_SEGMENTS_PER_SYSTEM[MEDICINE_DATASET],
        )
    return shards


def _format_prompt(lang: str, source: str, hypothesis: str) -> str:
    return PROMPT_TEMPLATE.format(
        source_lang=SOURCE_LANGUAGE_NAME,
        target_lang=TARGET_LANGUAGE_NAMES[lang],
        source_seg=source,
        target_seg=hypothesis,
    )


def _request_key(
    sidecar: Mapping[str, Any], *, prompt_sha256: str, model: str, config_sha256: str
) -> str:
    payload = {
        "identity": (
            sidecar["dataset"],
            sidecar["method"],
            sidecar["lang"],
            sidecar["lm"],
            sidecar["talk_id"],
            sidecar["talk_sentence_index"],
        ),
        "source": sidecar["source"],
        "hypothesis": sidecar["hypothesis"],
        "prompt_sha256": prompt_sha256,
        "model": model,
        "generation_config_sha256": config_sha256,
    }
    return "judge-" + canonical_json_sha256(payload)


def _validate_manifest(output_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    manifest = _read_json(output_dir / "run_manifest.json", label="run manifest")
    _assert_equal(manifest.get("schema_version"), SCHEMA_VERSION, context="manifest schema_version")
    model = _nonempty(manifest.get("model"), context="manifest model")
    methodology = manifest.get("methodology")
    if not isinstance(methodology, dict):
        raise GeminiJudgeValidationError("manifest methodology must be an object")
    _assert_equal(methodology.get("prompt_template"), PROMPT_TEMPLATE, context="prompt template")
    prompt_hash = sha256_bytes(PROMPT_TEMPLATE.encode("utf-8"))
    _assert_equal(
        _sha256(methodology.get("prompt_sha256"), context="manifest prompt_sha256"),
        prompt_hash,
        context="prompt_sha256",
    )
    _assert_equal(methodology.get("source_language"), SOURCE_LANGUAGE_NAME, context="source language")
    _assert_equal(
        methodology.get("target_language_names"), TARGET_LANGUAGE_NAMES, context="target languages"
    )
    _assert_equal(methodology.get("reference_passed_to_model"), False, context="reference flag")
    _assert_equal(methodology.get("request_deduplication"), False, context="deduplication flag")
    mode = methodology.get("generation_config_mode")
    if mode not in {"api-default", "temperature-zero"}:
        raise GeminiJudgeValidationError(
            f"Unsupported generation_config_mode: {mode!r}"
        )
    config = methodology.get("generation_config")
    if not isinstance(config, dict):
        raise GeminiJudgeValidationError("generation_config must be an object")
    expected_config = (
        {}
        if mode == "api-default"
        else {
            "temperature": 0.0,
            "candidateCount": 1,
            "responseMimeType": "text/plain",
        }
    )
    _assert_equal(config, expected_config, context=f"generation_config for mode {mode}")
    config_hash = canonical_json_sha256(config)
    _assert_equal(
        _sha256(
            methodology.get("generation_config_sha256"),
            context="manifest generation_config_sha256",
        ),
        config_hash,
        context="generation_config_sha256",
    )

    matrix = manifest.get("matrix")
    if not isinstance(matrix, dict):
        raise GeminiJudgeValidationError("manifest matrix must be an object")
    expected_matrix = {
        "systems": EXPECTED_SYSTEMS,
        "pairs": EXPECTED_PAIRS,
        "segments": EXPECTED_SEGMENTS,
        "acl_segments": EXPECTED_ACL_SEGMENTS,
        "medicine_segments": EXPECTED_MEDICINE_SEGMENTS,
        "shards": EXPECTED_PAIRS,
    }
    for field, expected in expected_matrix.items():
        _assert_equal(matrix.get(field), expected, context=f"manifest matrix.{field}")
    _canonical_nonnegative_int(matrix.get("empty_hypotheses"), context="matrix.empty_hypotheses")

    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, list) or len(source_artifacts) != 2:
        raise GeminiJudgeValidationError("manifest must contain exactly two source artifacts")
    source_by_role: Dict[str, Dict[str, Any]] = {}
    expected_role_dataset = {
        "acl_release_cache": ACL_DATASET,
        "medicine_paper_exact": MEDICINE_DATASET,
    }
    for artifact in source_artifacts:
        if not isinstance(artifact, dict):
            raise GeminiJudgeValidationError("source artifact entry must be an object")
        role = _nonempty(artifact.get("role"), context="source artifact role")
        if role in source_by_role:
            raise GeminiJudgeValidationError(f"Duplicate source artifact role: {role}")
        if role not in expected_role_dataset:
            raise GeminiJudgeValidationError(f"Unexpected source artifact role: {role}")
        _sha256(artifact.get("sha256"), context=f"source artifact {role} sha256")
        selection = artifact.get("selection")
        if not isinstance(selection, dict):
            raise GeminiJudgeValidationError(f"source artifact {role} selection is malformed")
        _assert_equal(
            selection.get("dataset_equals"),
            expected_role_dataset[role],
            context=f"source artifact {role} selection",
        )
        source_by_role[role] = artifact

    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise GeminiJudgeValidationError("manifest shards must be a list")
    shard_by_id: Dict[str, Dict[str, Any]] = {}
    for entry in shards:
        if not isinstance(entry, dict):
            raise GeminiJudgeValidationError("manifest shard entry must be an object")
        shard_id = _nonempty(entry.get("shard_id"), context="shard_id")
        if shard_id in shard_by_id:
            raise GeminiJudgeValidationError(f"Duplicate manifest shard: {shard_id}")
        shard_by_id[shard_id] = entry
    expected_shards = _expected_shards()
    if set(shard_by_id) != set(expected_shards):
        raise GeminiJudgeValidationError(
            "Manifest shard set mismatch: "
            f"missing={sorted(set(expected_shards) - set(shard_by_id))!r}, "
            f"unexpected={sorted(set(shard_by_id) - set(expected_shards))!r}"
        )
    for shard_id, (dataset, lang, lm, count) in expected_shards.items():
        entry = shard_by_id[shard_id]
        for field, expected in (
            ("dataset", dataset),
            ("lang", lang),
            ("lm", lm),
            ("methods", list(METHODS)),
            ("request_count", count),
        ):
            _assert_equal(entry.get(field), expected, context=f"shard {shard_id}.{field}")
        for kind in ("request", "sidecar"):
            path = _artifact_path(output_dir, entry.get(f"{kind}_path"), label=f"{shard_id} {kind}")
            expected_hash = _sha256(
                entry.get(f"{kind}_sha256"), context=f"shard {shard_id} {kind}_sha256"
            )
            _assert_equal(sha256_file(path), expected_hash, context=f"shard {shard_id} {kind} hash")
            _assert_equal(path.stat().st_size, entry.get(f"{kind}_bytes"), context=f"shard {shard_id} {kind} bytes")

    run_config = {
        "run_id": manifest.get("run_id"),
        "model": model,
        "prompt_sha256": prompt_hash,
        "generation_config": config,
        "source_artifact_hashes": [artifact["sha256"] for artifact in source_artifacts],
        "selected_datasets": [ACL_DATASET, MEDICINE_DATASET],
        "shard_request_hashes": [entry["request_sha256"] for entry in shards],
    }
    _assert_equal(
        _sha256(manifest.get("run_config_sha256"), context="manifest run_config_sha256"),
        canonical_json_sha256(run_config),
        context="run_config_sha256",
    )
    return manifest, shard_by_id


def _parse_batch_response(response: Any, *, context: str) -> Dict[str, Any]:
    if not isinstance(response, dict):
        raise GeminiJudgeValidationError(f"{context}.response must be an object")
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise GeminiJudgeValidationError(f"{context} must contain exactly one candidate")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise GeminiJudgeValidationError(f"{context} candidate must be an object")
    _assert_equal(candidate.get("finishReason"), "STOP", context=f"{context} finishReason")
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list) or not parts:
        raise GeminiJudgeValidationError(f"{context} candidate has no content parts")
    visible: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            raise GeminiJudgeValidationError(f"{context} content part must be an object")
        if part.get("thought") is True:
            continue
        if "text" in part:
            if not isinstance(part["text"], str):
                raise GeminiJudgeValidationError(f"{context} visible text must be a string")
            visible.append(part["text"])
    if len(visible) != 1:
        raise GeminiJudgeValidationError(
            f"{context} must contain exactly one non-thinking text part"
        )
    raw_text = visible[0]
    cleaned = raw_text.strip(" \t\r\n")
    if SCORE_RE.fullmatch(cleaned) is None:
        raise GeminiJudgeValidationError(
            f"{context} score is not one integer in [0,100]: {raw_text!r}"
        )
    model_version = _nonempty(response.get("modelVersion"), context=f"{context} modelVersion")
    usage = response.get("usageMetadata")
    if not isinstance(usage, dict):
        raise GeminiJudgeValidationError(f"{context} usageMetadata must be an object")
    usage_tokens: Dict[str, int] = {}
    for api_field, output_field, required in (
        ("promptTokenCount", "prompt_tokens", True),
        ("candidatesTokenCount", "candidate_tokens", True),
        ("thoughtsTokenCount", "thinking_tokens", False),
        ("totalTokenCount", "total_tokens", True),
    ):
        if api_field not in usage:
            if required:
                raise GeminiJudgeValidationError(
                    f"{context} usageMetadata is missing {api_field}"
                )
            usage_tokens[output_field] = 0
            continue
        usage_tokens[output_field] = _canonical_nonnegative_int(
            usage[api_field], context=f"{context} usageMetadata.{api_field}"
        )
    return {
        "judge_score": int(cleaned),
        "judge_raw_text": raw_text,
        "model_version": model_version,
        "finish_reason": "STOP",
        "usage_metadata": usage,
        "usage_tokens": usage_tokens,
        "response_id": response.get("responseId"),
    }


def _parse_int64_string(value: Any, *, context: str) -> int:
    if not isinstance(value, str) or not value or not value.isascii() or not value.isdigit():
        raise GeminiJudgeValidationError(
            f"{context} must be a canonical non-negative int64 string: {value!r}"
        )
    if len(value) > 1 and value.startswith("0"):
        raise GeminiJudgeValidationError(
            f"{context} must not contain leading zeroes: {value!r}"
        )
    parsed = int(value)
    if parsed > 2**63 - 1:
        raise GeminiJudgeValidationError(f"{context} exceeds signed int64: {value!r}")
    return parsed


def _raw_status_state(raw_status: Mapping[str, Any], *, context: str) -> str:
    top_level = raw_status.get("state")
    metadata = raw_status.get("metadata")
    metadata_state = metadata.get("state") if isinstance(metadata, dict) else None
    candidates = [value for value in (top_level, metadata_state) if value is not None]
    if not candidates:
        raise GeminiJudgeValidationError(f"{context} has no Batch state")
    if any(not isinstance(value, str) or not value for value in candidates):
        raise GeminiJudgeValidationError(f"{context} Batch state is malformed")
    if len(set(candidates)) != 1:
        raise GeminiJudgeValidationError(f"{context} has conflicting Batch states")
    return str(candidates[0])


def _raw_status_response_file(raw_status: Mapping[str, Any], *, context: str) -> str:
    paths = (
        ("dest", "fileName"),
        ("metadata", "output", "responsesFile"),
        ("response", "responsesFile"),
        ("outputConfig", "fileName"),
    )
    found: List[str] = []
    for path in paths:
        value: Any = raw_status
        for component in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(component)
        if value is not None:
            found.append(_nonempty(value, context=f"{context}.{'.'.join(path)}"))
    if not found:
        raise GeminiJudgeValidationError(f"{context} contains no response file name")
    if len(set(found)) != 1:
        raise GeminiJudgeValidationError(f"{context} contains conflicting response file names")
    return found[0]


def _validate_raw_status(
    *,
    output_dir: Path,
    state: Mapping[str, Any],
    shard_id: str,
    expected_requests: int,
) -> None:
    raw_path = _artifact_path(
        output_dir,
        state.get("raw_status_path"),
        label=f"{shard_id} raw status",
    )
    raw_hash = _sha256(
        state.get("raw_status_sha256"), context=f"{shard_id} raw_status_sha256"
    )
    _assert_equal(sha256_file(raw_path), raw_hash, context=f"{shard_id} raw status hash")
    raw_status = _read_json(raw_path, label=f"{shard_id} raw status")
    raw_state = _raw_status_state(raw_status, context=f"{shard_id} raw status")
    if raw_state not in {"BATCH_STATE_SUCCEEDED", "JOB_STATE_SUCCEEDED"}:
        raise GeminiJudgeValidationError(
            f"{shard_id} raw Batch state is not succeeded: {raw_state!r}"
        )
    _assert_equal(state.get("raw_api_state"), raw_state, context=f"{shard_id} state raw_api_state")
    metadata = raw_status.get("metadata")
    if not isinstance(metadata, dict):
        raise GeminiJudgeValidationError(f"{shard_id} raw status metadata must be an object")
    batch_stats = metadata.get("batchStats")
    if not isinstance(batch_stats, dict):
        raise GeminiJudgeValidationError(
            f"{shard_id} raw status metadata.batchStats must be an object"
        )
    expected_stats = {
        "requestCount": expected_requests,
        "successfulRequestCount": expected_requests,
        "failedRequestCount": 0,
        "pendingRequestCount": 0,
    }
    for field, expected in expected_stats.items():
        parsed = _parse_int64_string(
            batch_stats.get(field),
            context=f"{shard_id} metadata.batchStats.{field}",
        )
        _assert_equal(parsed, expected, context=f"{shard_id} metadata.batchStats.{field}")
    response_file = _raw_status_response_file(
        raw_status, context=f"{shard_id} raw status"
    )
    _assert_equal(
        state.get("response_file_name"),
        response_file,
        context=f"{shard_id} state response_file_name",
    )


def _validate_sidecar_and_request(
    *,
    sidecar: Mapping[str, Any],
    request_row: Mapping[str, Any],
    shard_id: str,
    manifest: Mapping[str, Any],
) -> None:
    context = f"sidecar {shard_id}/{sidecar.get('request_key')}"
    required = (
        "schema_version",
        "request_key",
        "shard_id",
        "dataset",
        "method",
        "lang",
        "lm",
        "talk_id",
        "talk_sentence_index",
        "source",
        "hypothesis",
        "source_sha256",
        "hypothesis_sha256",
        "reference_sha256",
        "prompt_sha256",
        "judge_input_sha256",
        "api_request_sha256",
        "source_artifact_role",
        "source_artifact_sha256",
        "source_record_line",
        "source_record_sha256",
    )
    missing = [field for field in required if field not in sidecar]
    if missing:
        raise GeminiJudgeValidationError(f"{context} missing fields: {', '.join(missing)}")
    _assert_equal(sidecar["schema_version"], SCHEMA_VERSION, context=f"{context}.schema_version")
    _assert_equal(sidecar["shard_id"], shard_id, context=f"{context}.shard_id")
    dataset = _nonempty(sidecar["dataset"], context=f"{context}.dataset")
    method = _nonempty(sidecar["method"], context=f"{context}.method")
    lang = _nonempty(sidecar["lang"], context=f"{context}.lang")
    lm = _nonempty(sidecar["lm"], context=f"{context}.lm")
    if (dataset, method, lang, lm) not in _expected_system_counts():
        raise GeminiJudgeValidationError(f"{context} has unexpected system identity")
    _nonempty(sidecar["talk_id"], context=f"{context}.talk_id")
    _canonical_nonnegative_int(sidecar["talk_sentence_index"], context=f"{context}.talk_sentence_index")
    source = _nonempty(sidecar["source"], context=f"{context}.source")
    hypothesis = sidecar["hypothesis"]
    if not isinstance(hypothesis, str):
        raise GeminiJudgeValidationError(f"{context}.hypothesis must be a string")
    _assert_equal(
        _sha256(sidecar["source_sha256"], context=f"{context}.source_sha256"),
        sha256_bytes(source.encode("utf-8")),
        context=f"{context}.source_sha256",
    )
    _assert_equal(
        _sha256(sidecar["hypothesis_sha256"], context=f"{context}.hypothesis_sha256"),
        sha256_bytes(hypothesis.encode("utf-8")),
        context=f"{context}.hypothesis_sha256",
    )
    _sha256(sidecar["reference_sha256"], context=f"{context}.reference_sha256")
    prompt_hash = manifest["methodology"]["prompt_sha256"]
    _assert_equal(sidecar["prompt_sha256"], prompt_hash, context=f"{context}.prompt_sha256")
    expected_judge_input_hash = canonical_json_sha256(
        {
            "source_lang": SOURCE_LANGUAGE_NAME,
            "target_lang": TARGET_LANGUAGE_NAMES[lang],
            "source": source,
            "translation": hypothesis,
            "prompt_template_sha256": prompt_hash,
        }
    )
    _assert_equal(
        _sha256(sidecar["judge_input_sha256"], context=f"{context}.judge_input_sha256"),
        expected_judge_input_hash,
        context=f"{context}.judge_input_sha256",
    )
    config = manifest["methodology"]["generation_config"]
    prompt = _format_prompt(lang, source, hypothesis)
    expected_request = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }
    if config:
        expected_request["generation_config"] = config
    if set(request_row) != {"key", "request"}:
        raise GeminiJudgeValidationError(f"Request row {context} has unexpected fields")
    _assert_equal(request_row.get("request"), expected_request, context=f"request payload {context}")
    api_hash = canonical_json_sha256(expected_request)
    _assert_equal(sidecar["api_request_sha256"], api_hash, context=f"{context}.api_request_sha256")
    expected_key = _request_key(
        sidecar,
        prompt_sha256=prompt_hash,
        model=manifest["model"],
        config_sha256=manifest["methodology"]["generation_config_sha256"],
    )
    _assert_equal(sidecar["request_key"], expected_key, context=f"{context}.request_key")
    _assert_equal(request_row.get("key"), expected_key, context=f"request key {context}")
    _positive_int(sidecar["source_record_line"], context=f"{context}.source_record_line")
    _sha256(sidecar["source_record_sha256"], context=f"{context}.source_record_sha256")


def _validate_shards(
    output_dir: Path,
    manifest: Mapping[str, Any],
    shard_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    expected_segments: Dict[str, Dict[str, Any]] = {}
    sidecars_by_role: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_identities = set()
    resolved_models = set()
    for shard_id in sorted(shard_by_id):
        shard = shard_by_id[shard_id]
        state = _read_json(output_dir / "states" / f"{shard_id}.json", label=f"{shard_id} state")
        _assert_equal(state.get("schema_version"), SCHEMA_VERSION, context=f"{shard_id} state schema")
        _assert_equal(state.get("run_id"), manifest.get("run_id"), context=f"{shard_id} state run_id")
        _assert_equal(state.get("shard_id"), shard_id, context=f"{shard_id} state shard_id")
        _assert_equal(state.get("status"), "DOWNLOADED", context=f"{shard_id} state status")
        _assert_equal(state.get("model"), manifest["model"], context=f"{shard_id} state model")
        _assert_equal(
            state.get("generation_config_sha256"),
            manifest["methodology"]["generation_config_sha256"],
            context=f"{shard_id} state generation config",
        )
        for field in ("request_path", "request_sha256", "request_bytes", "request_count"):
            _assert_equal(state.get(field), shard.get(field), context=f"{shard_id} state {field}")
        _validate_raw_status(
            output_dir=output_dir,
            state=state,
            shard_id=shard_id,
            expected_requests=int(shard["request_count"]),
        )
        response_path = _artifact_path(
            output_dir, state.get("response_path"), label=f"{shard_id} response"
        )
        response_hash = _sha256(
            state.get("response_sha256"), context=f"{shard_id} response_sha256"
        )
        _assert_equal(sha256_file(response_path), response_hash, context=f"{shard_id} response hash")
        _assert_equal(response_path.stat().st_size, state.get("response_bytes"), context=f"{shard_id} response bytes")

        request_rows = _read_jsonl(
            _artifact_path(output_dir, shard["request_path"], label=f"{shard_id} request"),
            label=f"{shard_id} requests",
        )
        sidecar_rows = _read_jsonl(
            _artifact_path(output_dir, shard["sidecar_path"], label=f"{shard_id} sidecar"),
            label=f"{shard_id} sidecars",
        )
        response_rows = _read_jsonl(response_path, label=f"{shard_id} responses")
        expected_count = int(shard["request_count"])
        for label, rows in (
            ("request", request_rows),
            ("sidecar", sidecar_rows),
            ("response", response_rows),
        ):
            _assert_equal(len(rows), expected_count, context=f"{shard_id} {label} row count")
        _assert_equal(state.get("response_rows"), expected_count, context=f"{shard_id} state response_rows")

        def by_key(rows: Sequence[Mapping[str, Any]], field: str, label: str) -> Dict[str, Mapping[str, Any]]:
            indexed: Dict[str, Mapping[str, Any]] = {}
            for number, row in enumerate(rows, start=1):
                key = _nonempty(row.get(field), context=f"{shard_id} {label} row {number}.{field}")
                if key in indexed:
                    raise GeminiJudgeValidationError(f"Duplicate {shard_id} {label} key: {key}")
                indexed[key] = row
            return indexed

        requests = by_key(request_rows, "key", "request")
        sidecars = by_key(sidecar_rows, "request_key", "sidecar")
        responses = by_key(response_rows, "key", "response")
        if set(requests) != set(sidecars) or set(sidecars) != set(responses):
            raise GeminiJudgeValidationError(f"{shard_id} request/sidecar/response key coverage mismatch")
        for request_key in sorted(sidecars):
            if request_key in expected_segments:
                raise GeminiJudgeValidationError(f"Request key appears in multiple shards: {request_key}")
            sidecar = dict(sidecars[request_key])
            _validate_sidecar_and_request(
                sidecar=sidecar,
                request_row=requests[request_key],
                shard_id=shard_id,
                manifest=manifest,
            )
            identity = (
                sidecar["dataset"],
                sidecar["method"],
                sidecar["lang"],
                sidecar["lm"],
                sidecar["talk_id"],
                sidecar["talk_sentence_index"],
            )
            if identity in all_identities:
                raise GeminiJudgeValidationError(f"Duplicate segment identity: {identity!r}")
            all_identities.add(identity)
            role = _nonempty(sidecar["source_artifact_role"], context="source artifact role")
            sidecars_by_role[role].append(sidecar)
            batch_row = responses[request_key]
            if set(batch_row).issuperset({"response", "error"}) or not (
                ("response" in batch_row) ^ ("error" in batch_row)
            ):
                raise GeminiJudgeValidationError(
                    f"Batch response {shard_id}/{request_key} must contain exactly one of response/error"
                )
            if "error" in batch_row:
                raise GeminiJudgeValidationError(
                    f"Batch response {shard_id}/{request_key} contains a per-request error"
                )
            parsed = _parse_batch_response(
                batch_row["response"], context=f"response {shard_id}/{request_key}"
            )
            resolved_models.add(parsed["model_version"])
            expected_segments[request_key] = {
                **sidecar,
                **parsed,
                "judge_model": manifest["model"],
                "generation_config_sha256": manifest["methodology"][
                    "generation_config_sha256"
                ],
                "response_artifact_sha256": response_hash,
            }
    _assert_equal(len(expected_segments), EXPECTED_SEGMENTS, context="raw response segment count")
    if len(resolved_models) != 1:
        raise GeminiJudgeValidationError(
            f"Expected one resolved model version, got {sorted(resolved_models)!r}"
        )
    return expected_segments, sidecars_by_role


def _validate_source_artifacts(
    manifest: Mapping[str, Any], sidecars_by_role: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    artifacts = {artifact["role"]: artifact for artifact in manifest["source_artifacts"]}
    if set(sidecars_by_role) != set(artifacts):
        raise GeminiJudgeValidationError("Sidecar/source-artifact role coverage mismatch")
    for role, artifact in artifacts.items():
        path = _require_file(Path(_nonempty(artifact.get("path"), context=f"{role} source path")), label=f"{role} source artifact")
        actual_hash = sha256_file(path)
        expected_hash = _sha256(artifact.get("sha256"), context=f"{role} source sha256")
        _assert_equal(actual_hash, expected_hash, context=f"{role} source artifact hash")
        _assert_equal(path.stat().st_size, artifact.get("bytes"), context=f"{role} source bytes")
        referenced: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
        for sidecar in sidecars_by_role[role]:
            _assert_equal(sidecar["source_artifact_sha256"], expected_hash, context=f"{role} sidecar artifact hash")
            line_number = _positive_int(sidecar["source_record_line"], context=f"{role} source_record_line")
            referenced[line_number].append(sidecar)
        total_rows = 0
        dataset_counts: Dict[str, int] = defaultdict(int)
        seen_lines = set()
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise GeminiJudgeValidationError(
                            f"{role} source contains a blank line at {path}:{line_number}"
                        )
                    try:
                        row = json.loads(line, parse_constant=_reject_json_constant)
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise GeminiJudgeValidationError(
                            f"Invalid source JSON at {path}:{line_number}: {exc}"
                        ) from exc
                    if not isinstance(row, dict):
                        raise GeminiJudgeValidationError(f"Source row is not an object: {path}:{line_number}")
                    total_rows += 1
                    dataset_counts[str(row.get("dataset") or "")] += 1
                    if line_number not in referenced:
                        continue
                    seen_lines.add(line_number)
                    row_hash = canonical_json_sha256(row)
                    required = ("dataset", "method", "lang", "lm", "talk_id", "talk_sentence_index", "source", "hypothesis", "reference")
                    missing = [field for field in required if field not in row]
                    if missing:
                        raise GeminiJudgeValidationError(
                            f"Referenced source row {path}:{line_number} misses {missing!r}"
                        )
                    for sidecar in referenced[line_number]:
                        context = f"{role} source row {line_number}/{sidecar['request_key']}"
                        _assert_equal(sidecar["source_record_sha256"], row_hash, context=f"{context} record hash")
                        comparisons = {
                            "dataset": row["dataset"],
                            "method": row["method"],
                            "lang": row["lang"],
                            "lm": str(row["lm"]),
                            "talk_id": row["talk_id"],
                            "talk_sentence_index": _canonical_nonnegative_int(
                                row["talk_sentence_index"], context=f"{context} talk_sentence_index"
                            ),
                            "source": row["source"],
                            "hypothesis": row["hypothesis"],
                        }
                        for field, expected in comparisons.items():
                            _assert_equal(sidecar[field], expected, context=f"{context}.{field}")
                        if not isinstance(row["reference"], str) or not row["reference"].strip():
                            raise GeminiJudgeValidationError(f"{context}.reference must be non-empty")
                        _assert_equal(
                            sidecar["reference_sha256"],
                            sha256_bytes(row["reference"].encode("utf-8")),
                            context=f"{context}.reference_sha256",
                        )
        except (OSError, UnicodeDecodeError) as exc:
            raise GeminiJudgeValidationError(f"Cannot read source artifact {path}: {exc}") from exc
        if seen_lines != set(referenced):
            raise GeminiJudgeValidationError(
                f"{role} source is missing referenced lines: {sorted(set(referenced) - seen_lines)[:10]!r}"
            )
        _assert_equal(total_rows, artifact.get("total_rows"), context=f"{role} total_rows")
        _assert_equal(dict(sorted(dataset_counts.items())), artifact.get("dataset_counts"), context=f"{role} dataset_counts")
        _assert_equal(len(sidecars_by_role[role]), artifact.get("selected_rows"), context=f"{role} selected_rows")


def _validate_collected_segments(
    output_dir: Path,
    expected_by_key: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str, str], List[Dict[str, Any]]]]:
    rows = _read_jsonl(output_dir / "segments.jsonl", label="collected segments")
    _assert_equal(len(rows), EXPECTED_SEGMENTS, context="collected segment count")
    actual_by_key: Dict[str, Dict[str, Any]] = {}
    by_system: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for number, row in enumerate(rows, start=1):
        request_key = _nonempty(row.get("request_key"), context=f"segment line {number}.request_key")
        if request_key in actual_by_key:
            raise GeminiJudgeValidationError(f"Duplicate collected request key: {request_key}")
        expected = expected_by_key.get(request_key)
        if expected is None:
            raise GeminiJudgeValidationError(f"Unknown collected request key: {request_key}")
        for field, expected_value in expected.items():
            if field not in row:
                raise GeminiJudgeValidationError(f"segment {request_key} missing {field}")
            _assert_equal(row[field], expected_value, context=f"segment {request_key}.{field}")
        score = row["judge_score"]
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
            raise GeminiJudgeValidationError(f"segment {request_key}.judge_score must be an integer in [0,100]")
        actual_by_key[request_key] = row
        system_key = tuple(str(row[field]) for field in SYSTEM_KEY_FIELDS)
        by_system[system_key].append(row)
    if set(actual_by_key) != set(expected_by_key):
        raise GeminiJudgeValidationError("Collected/raw response request-key coverage mismatch")
    expected_counts = _expected_system_counts()
    observed_counts = {key: len(values) for key, values in by_system.items()}
    if observed_counts != expected_counts:
        raise GeminiJudgeValidationError(
            f"Collected system matrix mismatch: observed={observed_counts!r}"
        )
    for key, system_rows in by_system.items():
        talks = {str(row["talk_id"]) for row in system_rows}
        _assert_equal(len(talks), EXPECTED_TALKS_PER_SYSTEM, context=f"system {key!r} talk count")
        segment_keys = set()
        for row in system_rows:
            segment_key = (str(row["talk_id"]), int(row["talk_sentence_index"]))
            if segment_key in segment_keys:
                raise GeminiJudgeValidationError(f"Duplicate segment key in system {key!r}: {segment_key!r}")
            segment_keys.add(segment_key)
    acl_count = sum(row["dataset"] == ACL_DATASET for row in rows)
    medicine_count = sum(row["dataset"] == MEDICINE_DATASET for row in rows)
    _assert_equal(acl_count, EXPECTED_ACL_SEGMENTS, context="ACL segment count")
    _assert_equal(medicine_count, EXPECTED_MEDICINE_SEGMENTS, context="medicine segment count")
    return rows, by_system


def _system_metrics(
    by_system: Mapping[Tuple[str, str, str, str], Sequence[Mapping[str, Any]]]
) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    metrics: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for key, rows in by_system.items():
        scores = [float(row["judge_score"]) for row in rows]
        by_talk: Dict[str, List[float]] = defaultdict(list)
        for row in rows:
            by_talk[str(row["talk_id"])].append(float(row["judge_score"]))
        metrics[key] = {
            "talks": len(by_talk),
            "segments": len(rows),
            "llm_judge_mean": float(statistics.fmean(scores)),
            "llm_judge_talk_macro_mean": float(
                statistics.fmean(statistics.fmean(values) for values in by_talk.values())
            ),
        }
    return metrics


def _pair_metrics(
    by_system: Mapping[Tuple[str, str, str, str], Sequence[Mapping[str, Any]]]
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Dict[Tuple[str, int], Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for system_key, rows in by_system.items():
        pair_key = (system_key[0], system_key[2], system_key[3])
        for row in rows:
            segment_key = (str(row["talk_id"]), int(row["talk_sentence_index"]))
            if segment_key in grouped[pair_key][system_key[1]]:
                raise GeminiJudgeValidationError(f"Duplicate paired segment {pair_key!r}/{segment_key!r}")
            grouped[pair_key][system_key[1]][segment_key] = row
    metrics: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for pair_key, by_method in grouped.items():
        if set(by_method) != set(METHODS):
            raise GeminiJudgeValidationError(f"Incomplete pair: {pair_key!r}")
        baseline = by_method["InfiniSST"]
        rasst = by_method["RASST"]
        if set(baseline) != set(rasst):
            raise GeminiJudgeValidationError(f"Unpaired segment keys: {pair_key!r}")
        baseline_scores: List[float] = []
        rasst_scores: List[float] = []
        deltas: List[float] = []
        wins = ties = losses = 0
        for segment_key in sorted(baseline):
            baseline_row = baseline[segment_key]
            rasst_row = rasst[segment_key]
            if (
                baseline_row["source"] != rasst_row["source"]
                or baseline_row["reference_sha256"] != rasst_row["reference_sha256"]
            ):
                raise GeminiJudgeValidationError(
                    f"Paired source/reference mismatch: {pair_key!r}/{segment_key!r}"
                )
            baseline_score = float(baseline_row["judge_score"])
            rasst_score = float(rasst_row["judge_score"])
            delta = rasst_score - baseline_score
            baseline_scores.append(baseline_score)
            rasst_scores.append(rasst_score)
            deltas.append(delta)
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1
            else:
                ties += 1
        metrics[pair_key] = {
            "rasst_method": "RASST",
            "infinisst_method": "InfiniSST",
            "paired_talks": len({key[0] for key in baseline}),
            "paired_segments": len(deltas),
            "rasst_llm_judge_mean": float(statistics.fmean(rasst_scores)),
            "infinisst_llm_judge_mean": float(statistics.fmean(baseline_scores)),
            "delta_rasst_minus_infinisst": float(statistics.fmean(deltas)),
            "paired_delta_stddev": float(statistics.stdev(deltas)) if len(deltas) > 1 else 0.0,
            "rasst_wins": wins,
            "ties": ties,
            "infinisst_wins": losses,
        }
    _assert_equal(len(metrics), EXPECTED_PAIRS, context="recomputed pair count")
    return metrics


def _talk_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (str(row["dataset"]), str(row["lang"]), str(row["lm"]), str(row["talk_id"]))
        grouped[key][str(row["method"])].append(float(row["judge_score"]))
    metrics: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for key, by_method in grouped.items():
        if set(by_method) != set(METHODS):
            raise GeminiJudgeValidationError(f"Incomplete talk pair: {key!r}")
        if len(by_method["RASST"]) != len(by_method["InfiniSST"]):
            raise GeminiJudgeValidationError(f"Talk segment count mismatch: {key!r}")
        rasst_mean = float(statistics.fmean(by_method["RASST"]))
        baseline_mean = float(statistics.fmean(by_method["InfiniSST"]))
        metrics[key] = {
            "paired_segments": len(by_method["RASST"]),
            "rasst_llm_judge_mean": rasst_mean,
            "infinisst_llm_judge_mean": baseline_mean,
            "delta_rasst_minus_infinisst": rasst_mean - baseline_mean,
        }
    _assert_equal(len(metrics), EXPECTED_TALK_PAIRS, context="recomputed talk pair count")
    return metrics


def _group_metrics(
    pair_metrics: Mapping[Tuple[str, str, str], Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    definitions = (
        ("acl_de_4lm_cell_macro", lambda key: key[0] == ACL_DATASET and key[1] == "de"),
        ("acl_ja_4lm_cell_macro", lambda key: key[0] == ACL_DATASET and key[1] == "ja"),
        ("acl_zh_4lm_cell_macro", lambda key: key[0] == ACL_DATASET and key[1] == "zh"),
        ("acl_12cell_macro", lambda key: key[0] == ACL_DATASET),
        ("medicine_de_4lm_cell_macro", lambda key: key[0] == MEDICINE_DATASET),
        ("all_16cell_macro_descriptive", lambda key: True),
    )
    metrics: Dict[str, Dict[str, Any]] = {}
    for name, predicate in definitions:
        selected = [values for key, values in pair_metrics.items() if predicate(key)]
        deltas = [float(row["delta_rasst_minus_infinisst"]) for row in selected]
        metrics[name] = {
            "cells": len(selected),
            "rasst_cell_macro_mean": float(
                statistics.fmean(float(row["rasst_llm_judge_mean"]) for row in selected)
            ),
            "infinisst_cell_macro_mean": float(
                statistics.fmean(float(row["infinisst_llm_judge_mean"]) for row in selected)
            ),
            "delta_cell_macro_rasst_minus_infinisst": float(statistics.fmean(deltas)),
            "positive_cells": sum(value > 0 for value in deltas),
            "zero_cells": sum(value == 0 for value in deltas),
            "negative_cells": sum(value < 0 for value in deltas),
        }
    return metrics


def _compare_metric_table(
    *,
    reported_rows: Sequence[Mapping[str, Any]],
    recomputed: Mapping[Tuple[str, ...], Mapping[str, Any]],
    key_fields: Sequence[str],
    integer_fields: Sequence[str],
    float_fields: Sequence[str],
    string_fields: Sequence[str],
    label: str,
    tolerance: float,
) -> None:
    reported = _index_unique(reported_rows, key_fields, label=label)
    if set(reported) != set(recomputed):
        raise GeminiJudgeValidationError(f"{label}/recomputed key set mismatch")
    for key, expected in recomputed.items():
        row = reported[key]
        for field in string_fields:
            _assert_equal(
                _nonempty(row.get(field), context=f"{label} {key!r}.{field}"),
                expected[field],
                context=f"{label} {key!r}.{field}",
            )
        for field in integer_fields:
            _assert_equal(
                _canonical_nonnegative_int(row.get(field), context=f"{label} {key!r}.{field}"),
                expected[field],
                context=f"{label} {key!r}.{field}",
            )
        for field in float_fields:
            _assert_close(
                row.get(field),
                float(expected[field]),
                context=f"{label} {key!r}.{field}",
                tolerance=tolerance,
            )


def _validate_tables(
    output_dir: Path,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    by_system: Mapping[Tuple[str, str, str, str], Sequence[Mapping[str, Any]]],
    *,
    tolerance: float,
) -> Tuple[
    Dict[Tuple[str, str, str, str], Dict[str, Any]],
    Dict[Tuple[str, str, str], Dict[str, Any]],
    Dict[Tuple[str, str, str, str], Dict[str, Any]],
    Dict[str, Dict[str, Any]],
]:
    system_metrics = _system_metrics(by_system)
    summary_rows = _read_tsv(output_dir / "summary.tsv", label="summary TSV", required_fields=SUMMARY_FIELDS)
    _assert_equal(len(summary_rows), EXPECTED_SYSTEMS, context="summary row count")
    reported_summary = _index_unique(summary_rows, SYSTEM_KEY_FIELDS, label="summary")
    if set(reported_summary) != set(system_metrics):
        raise GeminiJudgeValidationError("summary/recomputed system key set mismatch")
    resolved_versions = {str(row["model_version"]) for row in rows}
    if len(resolved_versions) != 1:
        raise GeminiJudgeValidationError("Collected segments contain multiple model versions")
    resolved_version = next(iter(resolved_versions))
    for key, expected in system_metrics.items():
        row = reported_summary[key]
        for field, value in (
            ("judge_model", manifest["model"]),
            ("resolved_model_version", resolved_version),
            ("prompt_sha256", manifest["methodology"]["prompt_sha256"]),
            ("generation_config_sha256", manifest["methodology"]["generation_config_sha256"]),
            ("run_config_sha256", manifest["run_config_sha256"]),
        ):
            _assert_equal(row[field], value, context=f"summary {key!r}.{field}")
        for field in ("talks", "segments"):
            _assert_equal(
                _canonical_nonnegative_int(row[field], context=f"summary {key!r}.{field}"),
                expected[field],
                context=f"summary {key!r}.{field}",
            )
        for field in ("llm_judge_mean", "llm_judge_talk_macro_mean"):
            _assert_close(row[field], expected[field], context=f"summary {key!r}.{field}", tolerance=tolerance)

    pair_metrics = _pair_metrics(by_system)
    paired_rows = _read_tsv(output_dir / "paired.tsv", label="paired TSV", required_fields=PAIRED_FIELDS)
    _assert_equal(len(paired_rows), EXPECTED_PAIRS, context="paired row count")
    _compare_metric_table(
        reported_rows=paired_rows,
        recomputed=pair_metrics,
        key_fields=PAIR_KEY_FIELDS,
        integer_fields=("paired_talks", "paired_segments", "rasst_wins", "ties", "infinisst_wins"),
        float_fields=("rasst_llm_judge_mean", "infinisst_llm_judge_mean", "delta_rasst_minus_infinisst", "paired_delta_stddev"),
        string_fields=("rasst_method", "infinisst_method"),
        label="paired",
        tolerance=tolerance,
    )

    talk_metrics = _talk_metrics(rows)
    talk_rows = _read_tsv(output_dir / "talk_paired.tsv", label="talk-paired TSV", required_fields=TALK_FIELDS)
    _assert_equal(len(talk_rows), EXPECTED_TALK_PAIRS, context="talk-paired row count")
    _compare_metric_table(
        reported_rows=talk_rows,
        recomputed=talk_metrics,
        key_fields=TALK_KEY_FIELDS,
        integer_fields=("paired_segments",),
        float_fields=("rasst_llm_judge_mean", "infinisst_llm_judge_mean", "delta_rasst_minus_infinisst"),
        string_fields=(),
        label="talk-paired",
        tolerance=tolerance,
    )

    group_metrics = _group_metrics(pair_metrics)
    group_rows = _read_tsv(output_dir / "group_summary.tsv", label="group summary TSV", required_fields=GROUP_FIELDS)
    _assert_equal(len(group_rows), len(group_metrics), context="group row count")
    reported_groups: Dict[str, Mapping[str, Any]] = {}
    for row in group_rows:
        name = _nonempty(row.get("group"), context="group name")
        if name in reported_groups:
            raise GeminiJudgeValidationError(f"Duplicate group row: {name}")
        reported_groups[name] = row
    if set(reported_groups) != set(group_metrics):
        raise GeminiJudgeValidationError("group summary/recomputed key set mismatch")
    for name, expected in group_metrics.items():
        row = reported_groups[name]
        for field in ("cells", "positive_cells", "zero_cells", "negative_cells"):
            _assert_equal(
                _canonical_nonnegative_int(row[field], context=f"group {name}.{field}"),
                expected[field],
                context=f"group {name}.{field}",
            )
        for field in (
            "rasst_cell_macro_mean",
            "infinisst_cell_macro_mean",
            "delta_cell_macro_rasst_minus_infinisst",
        ):
            _assert_close(row[field], expected[field], context=f"group {name}.{field}", tolerance=tolerance)
    return system_metrics, pair_metrics, talk_metrics, group_metrics


def _validate_collection_manifest(
    output_dir: Path,
    run_manifest: Mapping[str, Any],
    *,
    resolved_model_version: str,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    collection = _read_json(output_dir / "collection_manifest.json", label="collection manifest")
    _assert_equal(collection.get("schema_version"), SCHEMA_VERSION, context="collection schema_version")
    for field, expected in (
        ("run_config_sha256", run_manifest["run_config_sha256"]),
        ("segments", EXPECTED_SEGMENTS),
        ("systems", EXPECTED_SYSTEMS),
        ("pairs", EXPECTED_PAIRS),
        ("talk_pairs", EXPECTED_TALK_PAIRS),
        ("resolved_model_version", resolved_model_version),
    ):
        _assert_equal(collection.get(field), expected, context=f"collection manifest {field}")
    usage_totals = collection.get("usage_totals")
    if not isinstance(usage_totals, dict):
        raise GeminiJudgeValidationError("collection manifest usage_totals must be an object")
    expected_usage_totals = {
        field: sum(int(row["usage_tokens"][field]) for row in rows)
        for field in ("prompt_tokens", "candidate_tokens", "thinking_tokens", "total_tokens")
    }
    if set(usage_totals) != set(expected_usage_totals):
        raise GeminiJudgeValidationError("collection manifest usage_totals field set mismatch")
    for field, expected in expected_usage_totals.items():
        _assert_equal(
            _canonical_nonnegative_int(
                usage_totals[field], context=f"collection usage_totals.{field}"
            ),
            expected,
            context=f"collection usage_totals.{field}",
        )
    artifacts = collection.get("artifacts")
    expected_names = {
        "segments.jsonl",
        "summary.tsv",
        "paired.tsv",
        "talk_paired.tsv",
        "group_summary.tsv",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_names:
        raise GeminiJudgeValidationError("collection manifest artifact set mismatch")
    for name in sorted(expected_names):
        evidence = artifacts[name]
        if not isinstance(evidence, dict):
            raise GeminiJudgeValidationError(f"collection artifact {name} evidence is malformed")
        path = _require_file(output_dir / name, label=f"collection artifact {name}")
        _assert_equal(
            sha256_file(path),
            _sha256(evidence.get("sha256"), context=f"collection artifact {name} sha256"),
            context=f"collection artifact {name} hash",
        )
        _assert_equal(path.stat().st_size, evidence.get("bytes"), context=f"collection artifact {name} bytes")
    return collection


def validate_output_dir(
    *, output_dir: Path, absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE
) -> Dict[str, Any]:
    output_dir = lexical_absolute(output_dir)
    if not output_dir.is_dir():
        raise GeminiJudgeValidationError(f"output-dir is not a directory: {output_dir}")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise GeminiJudgeValidationError("absolute_tolerance must be finite and non-negative")
    manifest, shards = _validate_manifest(output_dir)
    expected_segments, sidecars_by_role = _validate_shards(output_dir, manifest, shards)
    _validate_source_artifacts(manifest, sidecars_by_role)
    rows, by_system = _validate_collected_segments(output_dir, expected_segments)
    system_metrics, pair_metrics, talk_metrics, group_metrics = _validate_tables(
        output_dir,
        manifest,
        rows,
        by_system,
        tolerance=absolute_tolerance,
    )
    resolved_model_version = str(rows[0]["model_version"])
    _validate_collection_manifest(
        output_dir,
        manifest,
        resolved_model_version=resolved_model_version,
        rows=rows,
    )
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "ok",
        "output_dir": str(output_dir),
        "validated_counts": {
            "systems": len(system_metrics),
            "pairs": len(pair_metrics),
            "segments": len(rows),
            "talk_pairs": len(talk_metrics),
            "groups": len(group_metrics),
            "shards": len(shards),
        },
        "model": manifest["model"],
        "resolved_model_version": resolved_model_version,
        "run_config_sha256": manifest["run_config_sha256"],
        "sha256": {
            name: sha256_file(output_dir / name)
            for name in (
                "run_manifest.json",
                "collection_manifest.json",
                "segments.jsonl",
                "summary.tsv",
                "paired.tsv",
                "talk_paired.tsv",
                "group_summary.tsv",
            )
        },
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = lexical_absolute(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=DEFAULT_ABSOLUTE_TOLERANCE)
    parser.add_argument("--report-json", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_output_dir(
            output_dir=args.output_dir,
            absolute_tolerance=args.absolute_tolerance,
        )
        if args.report_json is not None:
            _atomic_write_json(args.report_json, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    except GeminiJudgeValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
