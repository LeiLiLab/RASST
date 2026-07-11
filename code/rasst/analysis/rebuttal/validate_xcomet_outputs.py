#!/usr/bin/env python3
"""Independently validate sentence-aligned xCOMET output artifacts.

The validator is deliberately read-only except for an explicitly requested
JSON report.  It parses the scorer manifest and all three output artifacts,
recomputes every reported aggregate from segment scores, and fails on the
first structural or numerical inconsistency.
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


SCHEMA_VERSION = "rasst-xcomet-validation-v1"
DEFAULT_EXPECTED_SYSTEMS = 32
DEFAULT_EXPECTED_PAIRS = 16
DEFAULT_EXPECTED_SEGMENTS = 22_728
DEFAULT_ABSOLUTE_TOLERANCE = 1e-9
TIE_TOLERANCE = 1e-12
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
SYSTEM_KEY_FIELDS = ("dataset", "method", "lang", "lm")
PAIR_KEY_FIELDS = ("dataset", "lang", "lm")
SUMMARY_REQUIRED_FIELDS = (
    *SYSTEM_KEY_FIELDS,
    "talks",
    "segments",
    "xcomet_mean",
    "xcomet_mean_x100",
    "xcomet_talk_macro_mean",
    "model_id",
    "model_revision",
    "scoring_config_sha256",
    "instances_log_sha256",
)
PAIRED_REQUIRED_FIELDS = (
    *PAIR_KEY_FIELDS,
    "rasst_method",
    "infinisst_method",
    "paired_talks",
    "paired_segments",
    "rasst_xcomet_mean",
    "infinisst_xcomet_mean",
    "delta_rasst_minus_infinisst",
    "paired_delta_stddev",
    "rasst_wins",
    "ties",
    "infinisst_wins",
)
SEGMENT_REQUIRED_FIELDS = (
    *SYSTEM_KEY_FIELDS,
    "talk_id",
    "talk_sentence_index",
    "source",
    "reference",
    "xcomet_score",
    "error_spans",
    "model",
    "provenance_hashes",
)


class XCometValidationError(RuntimeError):
    """Raised when an xCOMET artifact is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise XCometValidationError(f"{label} is not a file: {resolved}")
    return resolved


def _non_empty(value: Any, *, context: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise XCometValidationError(f"Empty value for {context}")
    return cleaned


def _row_key(
    row: Mapping[str, Any],
    fields: Sequence[str],
    *,
    context: str,
) -> Tuple[str, ...]:
    return tuple(
        _non_empty(row.get(field), context=f"{context}.{field}") for field in fields
    )


def _read_tsv(
    path: Path,
    *,
    label: str,
    required_fields: Sequence[str],
) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise XCometValidationError(f"{label} has no TSV header: {path}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise XCometValidationError(f"{label} has duplicate TSV columns: {path}")
            missing = [field for field in required_fields if field not in reader.fieldnames]
            if missing:
                raise XCometValidationError(
                    f"{label} is missing required columns: {', '.join(missing)}"
                )
            rows: List[Dict[str, str]] = []
            for line_number, raw_row in enumerate(reader, start=2):
                if None in raw_row:
                    raise XCometValidationError(
                        f"{label} has extra unheaded columns at {path}:{line_number}"
                    )
                row = {str(key): str(value or "") for key, value in raw_row.items()}
                if not any(value.strip() for value in row.values()):
                    raise XCometValidationError(
                        f"{label} contains a blank row at {path}:{line_number}"
                    )
                rows.append(row)
    except UnicodeDecodeError as exc:
        raise XCometValidationError(f"{label} is not valid UTF-8: {path}") from exc
    if not rows:
        raise XCometValidationError(f"{label} contains no data rows: {path}")
    return rows


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard numeric constant {value!r}")


def _read_jsonl(path: Path, *, label: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise XCometValidationError(
                        f"{label} contains a blank line at {path}:{line_number}"
                    )
                try:
                    record = json.loads(line, parse_constant=_reject_json_constant)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise XCometValidationError(
                        f"Invalid JSON at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise XCometValidationError(
                        f"Expected a JSON object at {path}:{line_number}"
                    )
                records.append(record)
    except UnicodeDecodeError as exc:
        raise XCometValidationError(f"{label} is not valid UTF-8: {path}") from exc
    if not records:
        raise XCometValidationError(f"{label} contains no records: {path}")
    return records


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".tsv":
        return list(
            _read_tsv(
                path,
                label="manifest",
                required_fields=SYSTEM_KEY_FIELDS,
            )
        )
    if path.suffix.lower() == ".jsonl":
        rows = _read_jsonl(path, label="manifest")
        for index, row in enumerate(rows, start=1):
            missing = [field for field in SYSTEM_KEY_FIELDS if field not in row]
            if missing:
                raise XCometValidationError(
                    f"manifest row {index} is missing fields: {', '.join(missing)}"
                )
        return rows
    raise XCometValidationError(f"Manifest must end in .tsv or .jsonl: {path}")


def _parse_finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise XCometValidationError(f"Expected a finite float for {context}, got bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise XCometValidationError(
            f"Expected a finite float for {context}, got {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise XCometValidationError(
            f"Expected a finite float for {context}, got {value!r}"
        )
    return parsed


def _parse_non_negative_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        raise XCometValidationError(
            f"Expected a non-negative integer for {context}, got bool"
        )
    text = str(value).strip()
    try:
        parsed = int(text)
    except (TypeError, ValueError) as exc:
        raise XCometValidationError(
            f"Expected a non-negative integer for {context}, got {value!r}"
        ) from exc
    if parsed < 0 or str(parsed) != text:
        raise XCometValidationError(
            f"Expected a canonical non-negative integer for {context}, got {value!r}"
        )
    return parsed


def _validate_sha256(value: Any, *, context: str) -> str:
    cleaned = _non_empty(value, context=context)
    if SHA256_RE.fullmatch(cleaned) is None:
        raise XCometValidationError(
            f"Expected a 64-hex SHA256 for {context}, got {value!r}"
        )
    return cleaned.lower()


def _assert_close(
    reported: Any,
    recomputed: float,
    *,
    context: str,
    tolerance: float,
) -> None:
    reported_float = _parse_finite_float(reported, context=context)
    if not math.isclose(
        reported_float,
        recomputed,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise XCometValidationError(
            f"{context} mismatch: reported={reported_float:.17g}, "
            f"recomputed={recomputed:.17g}, tolerance={tolerance:.3g}"
        )


def _unique_rows_by_key(
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    label: str,
) -> Dict[Tuple[str, ...], Mapping[str, Any]]:
    indexed: Dict[Tuple[str, ...], Mapping[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        key = _row_key(row, fields, context=f"{label} row {index}")
        if key in indexed:
            raise XCometValidationError(f"Duplicate {label} key: {key!r}")
        indexed[key] = row
    return indexed


def _validate_positive_expected_count(value: int, label: str) -> None:
    if value <= 0:
        raise XCometValidationError(f"{label} must be positive, got {value}")


def _system_metrics(
    segments_by_system: Mapping[Tuple[str, ...], Sequence[Mapping[str, Any]]],
) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    metrics: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for key, segments in segments_by_system.items():
        if not segments:
            raise XCometValidationError(f"System has no segments: {key!r}")
        scores = [float(segment["_validated_score"]) for segment in segments]
        scores_by_talk: Dict[str, List[float]] = defaultdict(list)
        segment_keys = set()
        for segment in segments:
            segment_key = (
                str(segment["_validated_talk_id"]),
                int(segment["_validated_talk_sentence_index"]),
            )
            if segment_key in segment_keys:
                raise XCometValidationError(
                    f"Duplicate segment key for system {key!r}: {segment_key!r}"
                )
            segment_keys.add(segment_key)
            scores_by_talk[str(segment["_validated_talk_id"])].append(
                float(segment["_validated_score"])
            )
        mean_score = float(statistics.fmean(scores))
        talk_macro = float(
            statistics.fmean(
                statistics.fmean(talk_scores)
                for talk_scores in scores_by_talk.values()
            )
        )
        metrics[key] = {
            "talks": len(scores_by_talk),
            "segments": len(segments),
            "xcomet_mean": mean_score,
            "xcomet_mean_x100": mean_score * 100.0,
            "xcomet_talk_macro_mean": talk_macro,
        }
    return metrics


def _pair_metrics(
    segments_by_system: Mapping[Tuple[str, ...], Sequence[Mapping[str, Any]]],
    *,
    rasst_method: str,
    infinisst_method: str,
) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    rasst_key = rasst_method.casefold()
    infinisst_key = infinisst_method.casefold()
    if not rasst_key or not infinisst_key or rasst_key == infinisst_key:
        raise XCometValidationError(
            "RASST and InfiniSST method labels must be distinct and non-empty"
        )

    grouped: Dict[
        Tuple[str, ...],
        Dict[str, Tuple[Tuple[str, ...], Sequence[Mapping[str, Any]]]],
    ] = defaultdict(dict)
    for system_key, segments in segments_by_system.items():
        method_key = system_key[1].casefold()
        if method_key not in {rasst_key, infinisst_key}:
            continue
        pair_key = (system_key[0], system_key[2], system_key[3])
        role = "rasst" if method_key == rasst_key else "infinisst"
        if role in grouped[pair_key]:
            raise XCometValidationError(
                f"Multiple {role} systems for pair {pair_key!r}"
            )
        grouped[pair_key][role] = (system_key, segments)

    metrics: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for pair_key, roles in grouped.items():
        if set(roles) != {"rasst", "infinisst"}:
            raise XCometValidationError(
                f"Incomplete RASST/InfiniSST pair {pair_key!r}: found {sorted(roles)}"
            )
        rasst_system_key, rasst_records = roles["rasst"]
        infinisst_system_key, infinisst_records = roles["infinisst"]

        def index_segments(
            records: Sequence[Mapping[str, Any]], role: str
        ) -> Dict[Tuple[str, int], Mapping[str, Any]]:
            indexed: Dict[Tuple[str, int], Mapping[str, Any]] = {}
            for record in records:
                segment_key = (
                    str(record["_validated_talk_id"]),
                    int(record["_validated_talk_sentence_index"]),
                )
                if segment_key in indexed:
                    raise XCometValidationError(
                        f"Duplicate {role} segment key for {pair_key!r}: {segment_key!r}"
                    )
                indexed[segment_key] = record
            return indexed

        rasst_segments = index_segments(rasst_records, "RASST")
        infinisst_segments = index_segments(infinisst_records, "InfiniSST")
        if set(rasst_segments) != set(infinisst_segments):
            missing_rasst = sorted(set(infinisst_segments) - set(rasst_segments))[:10]
            missing_infinisst = sorted(set(rasst_segments) - set(infinisst_segments))[:10]
            raise XCometValidationError(
                f"Unpaired segment keys for {pair_key!r}; "
                f"missing_rasst={missing_rasst!r}, "
                f"missing_infinisst={missing_infinisst!r}"
            )

        rasst_scores: List[float] = []
        infinisst_scores: List[float] = []
        deltas: List[float] = []
        wins = ties = losses = 0
        for segment_key in sorted(rasst_segments):
            rasst_segment = rasst_segments[segment_key]
            infinisst_segment = infinisst_segments[segment_key]
            if (
                rasst_segment["source"] != infinisst_segment["source"]
                or rasst_segment["reference"] != infinisst_segment["reference"]
            ):
                raise XCometValidationError(
                    f"Source/reference mismatch for {pair_key!r}/{segment_key!r}"
                )
            rasst_score = float(rasst_segment["_validated_score"])
            infinisst_score = float(infinisst_segment["_validated_score"])
            delta = rasst_score - infinisst_score
            rasst_scores.append(rasst_score)
            infinisst_scores.append(infinisst_score)
            deltas.append(delta)
            if abs(delta) <= TIE_TOLERANCE:
                ties += 1
            elif delta > 0:
                wins += 1
            else:
                losses += 1

        metrics[pair_key] = {
            "rasst_method": rasst_system_key[1],
            "infinisst_method": infinisst_system_key[1],
            "paired_talks": len({key[0] for key in rasst_segments}),
            "paired_segments": len(deltas),
            "rasst_xcomet_mean": float(statistics.fmean(rasst_scores)),
            "infinisst_xcomet_mean": float(statistics.fmean(infinisst_scores)),
            "delta_rasst_minus_infinisst": float(statistics.fmean(deltas)),
            "paired_delta_stddev": (
                float(statistics.stdev(deltas)) if len(deltas) > 1 else 0.0
            ),
            "rasst_wins": wins,
            "ties": ties,
            "infinisst_wins": losses,
        }
    return metrics


def _report_system_rows(
    metrics: Mapping[Tuple[str, ...], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            **dict(zip(SYSTEM_KEY_FIELDS, key)),
            **dict(values),
        }
        for key, values in sorted(metrics.items())
    ]


def _report_pair_rows(
    metrics: Mapping[Tuple[str, ...], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            **dict(zip(PAIR_KEY_FIELDS, key)),
            **dict(values),
        }
        for key, values in sorted(metrics.items())
    ]


def validate_outputs(
    *,
    manifest: Path,
    summary_tsv: Path,
    paired_tsv: Path,
    segments_jsonl: Path,
    expected_systems: int = DEFAULT_EXPECTED_SYSTEMS,
    expected_pairs: int = DEFAULT_EXPECTED_PAIRS,
    expected_segments: int = DEFAULT_EXPECTED_SEGMENTS,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    rasst_method: str = "RASST",
    infinisst_method: str = "InfiniSST",
) -> Dict[str, Any]:
    for value, label in (
        (expected_systems, "expected_systems"),
        (expected_pairs, "expected_pairs"),
        (expected_segments, "expected_segments"),
    ):
        _validate_positive_expected_count(value, label)
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise XCometValidationError(
            f"absolute_tolerance must be finite and non-negative: {absolute_tolerance!r}"
        )

    manifest = _require_file(manifest, "manifest")
    summary_tsv = _require_file(summary_tsv, "summary TSV")
    paired_tsv = _require_file(paired_tsv, "paired TSV")
    segments_jsonl = _require_file(segments_jsonl, "segments JSONL")
    paths = [manifest, summary_tsv, paired_tsv, segments_jsonl]
    if len(paths) != len(set(paths)):
        raise XCometValidationError("Manifest and output artifact paths must be distinct")

    manifest_rows = _read_manifest(manifest)
    manifest_by_system = _unique_rows_by_key(
        manifest_rows,
        SYSTEM_KEY_FIELDS,
        label="manifest system",
    )
    if len(manifest_by_system) != expected_systems:
        raise XCometValidationError(
            f"Manifest system count mismatch: expected={expected_systems}, "
            f"actual={len(manifest_by_system)}"
        )

    summary_rows = _read_tsv(
        summary_tsv,
        label="summary TSV",
        required_fields=SUMMARY_REQUIRED_FIELDS,
    )
    summary_by_system = _unique_rows_by_key(
        summary_rows,
        SYSTEM_KEY_FIELDS,
        label="summary system",
    )
    if len(summary_by_system) != expected_systems:
        raise XCometValidationError(
            f"Summary system count mismatch: expected={expected_systems}, "
            f"actual={len(summary_by_system)}"
        )
    summary_global_values: Dict[str, str] = {}
    for field in ("model_id", "model_revision", "scoring_config_sha256"):
        values = {
            _non_empty(row[field], context=f"summary {key!r}.{field}")
            for key, row in summary_by_system.items()
        }
        if len(values) != 1:
            raise XCometValidationError(
                f"Summary {field} is not globally consistent: {sorted(values)!r}"
            )
        summary_global_values[field] = next(iter(values))
    summary_global_values["scoring_config_sha256"] = _validate_sha256(
        summary_global_values["scoring_config_sha256"],
        context="summary scoring_config_sha256",
    )
    for system_key, row in summary_by_system.items():
        _validate_sha256(
            row["instances_log_sha256"],
            context=f"summary {system_key!r}.instances_log_sha256",
        )

    paired_rows = _read_tsv(
        paired_tsv,
        label="paired TSV",
        required_fields=PAIRED_REQUIRED_FIELDS,
    )
    paired_by_key = _unique_rows_by_key(
        paired_rows,
        PAIR_KEY_FIELDS,
        label="paired result",
    )
    if len(paired_by_key) != expected_pairs:
        raise XCometValidationError(
            f"Paired row count mismatch: expected={expected_pairs}, "
            f"actual={len(paired_by_key)}"
        )

    segment_rows = _read_jsonl(segments_jsonl, label="segments JSONL")
    if len(segment_rows) != expected_segments:
        raise XCometValidationError(
            f"Segment count mismatch: expected={expected_segments}, "
            f"actual={len(segment_rows)}"
        )
    segments_by_system: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for line_number, segment in enumerate(segment_rows, start=1):
        missing = [field for field in SEGMENT_REQUIRED_FIELDS if field not in segment]
        if missing:
            raise XCometValidationError(
                f"Segment {line_number} is missing fields: {', '.join(missing)}"
            )
        system_key = _row_key(
            segment,
            SYSTEM_KEY_FIELDS,
            context=f"segment line {line_number}",
        )
        score = _parse_finite_float(
            segment["xcomet_score"],
            context=f"segment line {line_number}.xcomet_score",
        )
        talk_id = _non_empty(
            segment["talk_id"],
            context=f"segment line {line_number}.talk_id",
        )
        talk_sentence_index = _parse_non_negative_int(
            segment["talk_sentence_index"],
            context=f"segment line {line_number}.talk_sentence_index",
        )
        if not isinstance(segment["source"], str) or not isinstance(
            segment["reference"], str
        ):
            raise XCometValidationError(
                f"Segment line {line_number} source/reference must be strings"
            )
        error_spans = segment["error_spans"]
        if not isinstance(error_spans, list) or not all(
            isinstance(span, dict) for span in error_spans
        ):
            raise XCometValidationError(
                f"Segment line {line_number}.error_spans must be a list of objects"
            )
        model = segment["model"]
        if not isinstance(model, dict):
            raise XCometValidationError(
                f"Segment line {line_number}.model must be an object"
            )
        segment_model_id = _non_empty(
            model.get("id"), context=f"segment line {line_number}.model.id"
        )
        segment_model_revision = _non_empty(
            model.get("revision"),
            context=f"segment line {line_number}.model.revision",
        )
        if segment_model_id != summary_global_values["model_id"]:
            raise XCometValidationError(
                f"Segment line {line_number}.model.id mismatch: "
                f"segment={segment_model_id!r}, "
                f"summary={summary_global_values['model_id']!r}"
            )
        if segment_model_revision != summary_global_values["model_revision"]:
            raise XCometValidationError(
                f"Segment line {line_number}.model.revision mismatch: "
                f"segment={segment_model_revision!r}, "
                f"summary={summary_global_values['model_revision']!r}"
            )

        provenance_hashes = segment["provenance_hashes"]
        if not isinstance(provenance_hashes, dict):
            raise XCometValidationError(
                f"Segment line {line_number}.provenance_hashes must be an object"
            )
        for required_hash in ("scoring_config_sha256", "instances_log_sha256"):
            if required_hash not in provenance_hashes:
                raise XCometValidationError(
                    f"Segment line {line_number}.provenance_hashes is missing "
                    f"{required_hash}"
                )
        validated_provenance_hashes = {
            key: _validate_sha256(
                value,
                context=f"segment line {line_number}.provenance_hashes.{key}",
            )
            for key, value in provenance_hashes.items()
            if str(key).endswith("_sha256")
        }
        if len(validated_provenance_hashes) != len(provenance_hashes):
            unexpected_keys = sorted(
                str(key)
                for key in provenance_hashes
                if not str(key).endswith("_sha256")
            )
            raise XCometValidationError(
                f"Segment line {line_number}.provenance_hashes has non-hash keys: "
                f"{unexpected_keys!r}"
            )
        if (
            validated_provenance_hashes["scoring_config_sha256"]
            != summary_global_values["scoring_config_sha256"]
        ):
            raise XCometValidationError(
                f"Segment line {line_number} scoring_config_sha256 mismatch: "
                f"segment={validated_provenance_hashes['scoring_config_sha256']!r}, "
                f"summary={summary_global_values['scoring_config_sha256']!r}"
            )
        summary_system = summary_by_system.get(system_key)
        if summary_system is not None:
            summary_instances_hash = _validate_sha256(
                summary_system["instances_log_sha256"],
                context=f"summary {system_key!r}.instances_log_sha256",
            )
            if (
                validated_provenance_hashes["instances_log_sha256"]
                != summary_instances_hash
            ):
                raise XCometValidationError(
                    f"Segment line {line_number} instances_log_sha256 mismatch: "
                    f"segment={validated_provenance_hashes['instances_log_sha256']!r}, "
                    f"summary={summary_instances_hash!r}"
                )
        for field, value in segment.items():
            if str(field).endswith("_sha256"):
                _validate_sha256(
                    value,
                    context=f"segment line {line_number}.{field}",
                )
        segment["_validated_score"] = score
        segment["_validated_talk_id"] = talk_id
        segment["_validated_talk_sentence_index"] = talk_sentence_index
        segments_by_system[system_key].append(segment)

    segment_system_keys = set(segments_by_system)
    manifest_system_keys = set(manifest_by_system)
    summary_system_keys = set(summary_by_system)
    if segment_system_keys != manifest_system_keys:
        raise XCometValidationError(
            "Segment/manifest system keys differ: "
            f"missing_segments={sorted(manifest_system_keys - segment_system_keys)!r}, "
            f"unexpected_segments={sorted(segment_system_keys - manifest_system_keys)!r}"
        )
    if summary_system_keys != manifest_system_keys:
        raise XCometValidationError(
            "Summary/manifest system keys differ: "
            f"missing_summary={sorted(manifest_system_keys - summary_system_keys)!r}, "
            f"unexpected_summary={sorted(summary_system_keys - manifest_system_keys)!r}"
        )

    system_metrics = _system_metrics(segments_by_system)
    for system_key, recomputed in system_metrics.items():
        reported = summary_by_system[system_key]
        context = f"summary {system_key!r}"
        for field in ("talks", "segments"):
            reported_integer = _parse_non_negative_int(
                reported[field], context=f"{context}.{field}"
            )
            if reported_integer != recomputed[field]:
                raise XCometValidationError(
                    f"{context}.{field} mismatch: reported={reported_integer}, "
                    f"recomputed={recomputed[field]}"
                )
        for field in (
            "xcomet_mean",
            "xcomet_mean_x100",
            "xcomet_talk_macro_mean",
        ):
            _assert_close(
                reported[field],
                float(recomputed[field]),
                context=f"{context}.{field}",
                tolerance=absolute_tolerance,
            )

    pair_metrics = _pair_metrics(
        segments_by_system,
        rasst_method=rasst_method,
        infinisst_method=infinisst_method,
    )
    if len(pair_metrics) != expected_pairs:
        raise XCometValidationError(
            f"Recomputed pair count mismatch: expected={expected_pairs}, "
            f"actual={len(pair_metrics)}"
        )
    if set(pair_metrics) != set(paired_by_key):
        raise XCometValidationError(
            "Paired TSV/recomputed keys differ: "
            f"missing_tsv={sorted(set(pair_metrics) - set(paired_by_key))!r}, "
            f"unexpected_tsv={sorted(set(paired_by_key) - set(pair_metrics))!r}"
        )

    for pair_key, recomputed in pair_metrics.items():
        reported = paired_by_key[pair_key]
        context = f"paired {pair_key!r}"
        for field in ("rasst_method", "infinisst_method"):
            value = _non_empty(reported[field], context=f"{context}.{field}")
            if value != recomputed[field]:
                raise XCometValidationError(
                    f"{context}.{field} mismatch: reported={value!r}, "
                    f"recomputed={recomputed[field]!r}"
                )
        for field in (
            "paired_talks",
            "paired_segments",
            "rasst_wins",
            "ties",
            "infinisst_wins",
        ):
            reported_integer = _parse_non_negative_int(
                reported[field], context=f"{context}.{field}"
            )
            if reported_integer != recomputed[field]:
                raise XCometValidationError(
                    f"{context}.{field} mismatch: reported={reported_integer}, "
                    f"recomputed={recomputed[field]}"
                )
        for field in (
            "rasst_xcomet_mean",
            "infinisst_xcomet_mean",
            "delta_rasst_minus_infinisst",
            "paired_delta_stddev",
        ):
            _assert_close(
                reported[field],
                float(recomputed[field]),
                context=f"{context}.{field}",
                tolerance=absolute_tolerance,
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "paths": {
            "manifest": str(manifest),
            "summary_tsv": str(summary_tsv),
            "paired_tsv": str(paired_tsv),
            "segments_jsonl": str(segments_jsonl),
        },
        "sha256": {
            "manifest": sha256_file(manifest),
            "summary_tsv": sha256_file(summary_tsv),
            "paired_tsv": sha256_file(paired_tsv),
            "segments_jsonl": sha256_file(segments_jsonl),
        },
        "expected_counts": {
            "systems": expected_systems,
            "pairs": expected_pairs,
            "segments": expected_segments,
        },
        "validated_counts": {
            "manifest_systems": len(manifest_by_system),
            "summary_systems": len(summary_by_system),
            "pairs": len(pair_metrics),
            "segments": len(segment_rows),
        },
        "tolerances": {
            "tsv_absolute": absolute_tolerance,
            "tie": TIE_TOLERANCE,
        },
        "run_identity": dict(summary_global_values),
        "systems": _report_system_rows(system_metrics),
        "pairs": _report_pair_rows(pair_metrics),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=resolved.parent,
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(resolved)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary-tsv", type=Path, required=True)
    parser.add_argument("--paired-tsv", type=Path, required=True)
    parser.add_argument("--segments-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument(
        "--expected-systems", type=int, default=DEFAULT_EXPECTED_SYSTEMS
    )
    parser.add_argument("--expected-pairs", type=int, default=DEFAULT_EXPECTED_PAIRS)
    parser.add_argument(
        "--expected-segments", type=int, default=DEFAULT_EXPECTED_SEGMENTS
    )
    parser.add_argument(
        "--absolute-tolerance", type=float, default=DEFAULT_ABSOLUTE_TOLERANCE
    )
    parser.add_argument("--rasst-method", default="RASST")
    parser.add_argument("--infinisst-method", default="InfiniSST")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    input_paths = {
        args.manifest.resolve(),
        args.summary_tsv.resolve(),
        args.paired_tsv.resolve(),
        args.segments_jsonl.resolve(),
    }
    if args.report_json is not None and args.report_json.resolve() in input_paths:
        print("[ERROR] Refusing to overwrite an input artifact with the report", file=sys.stderr)
        return 2
    try:
        report = validate_outputs(
            manifest=args.manifest,
            summary_tsv=args.summary_tsv,
            paired_tsv=args.paired_tsv,
            segments_jsonl=args.segments_jsonl,
            expected_systems=args.expected_systems,
            expected_pairs=args.expected_pairs,
            expected_segments=args.expected_segments,
            absolute_tolerance=args.absolute_tolerance,
            rasst_method=args.rasst_method,
            infinisst_method=args.infinisst_method,
        )
        if args.report_json is not None:
            _atomic_write_json(args.report_json, report)
    except (OSError, XCometValidationError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report["validated_counts"], sort_keys=True), flush=True)
    if args.report_json is not None:
        print(
            f"[REPORT] {args.report_json.resolve()} "
            f"sha256={sha256_file(args.report_json.resolve())}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
