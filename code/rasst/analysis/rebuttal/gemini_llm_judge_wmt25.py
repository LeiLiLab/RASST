#!/usr/bin/env python3
"""Run the WMT25 reference-free LLM-judge prompt with Gemini Batch.

The rebuttal matrix is deliberately strict: ACL En-{Zh,De,Ja} is selected
from the release-cache xCOMET segment artifact, while Medicine/ESO En-De is
selected from the submitted-paper-exact replacement artifact.  The script
prepares opaque-keyed Gemini Batch JSONL shards, submits one shard at a time,
polls and downloads completed jobs, and aggregates only a complete result set.

No environment variable is read for configuration or credentials.  Gemini
credentials must be supplied through an owner-only regular file.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import platform
import re
import socket
import stat
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "rasst-gemini-llm-judge-wmt25-v1"
WMT_PAPER_URL = "https://aclanthology.org/2025.wmt-1.24.pdf"
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
TERMINAL_STATES = {
    "BATCH_STATE_SUCCEEDED",
    "BATCH_STATE_FAILED",
    "BATCH_STATE_CANCELLED",
    "BATCH_STATE_EXPIRED",
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}
SUCCESS_STATES = {"BATCH_STATE_SUCCEEDED", "JOB_STATE_SUCCEEDED"}
SCORE_RE = re.compile(r"^(?:100|[0-9]{1,2})$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

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


class LLMJudgeError(RuntimeError):
    """Raised when an LLM-judge artifact cannot be used without guessing."""


@dataclass(frozen=True)
class SourceRecord:
    dataset: str
    method: str
    lang: str
    lm: str
    talk_id: str
    talk_sentence_index: int
    source: str
    hypothesis: str
    reference_sha256: str
    source_artifact_role: str
    source_artifact_sha256: str
    source_record_line: int
    source_record_sha256: str

    @property
    def system_key(self) -> Tuple[str, str, str, str]:
        return (self.dataset, self.method, self.lang, self.lm)

    @property
    def pair_key(self) -> Tuple[str, str, str]:
        return (self.dataset, self.lang, self.lm)

    @property
    def paired_segment_key(self) -> Tuple[str, str, int]:
        return (self.dataset, self.talk_id, self.talk_sentence_index)

    @property
    def identity(self) -> Tuple[str, str, str, str, str, int]:
        return (
            self.dataset,
            self.method,
            self.lang,
            self.lm,
            self.talk_id,
            self.talk_sentence_index,
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def json_line(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    atomic_write_text(path, payload)


def read_json(path: Path, *, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise LLMJudgeError(f"Cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMJudgeError(f"{label} must be a JSON object: {path}")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def iter_jsonl(path: Path, *, label: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise LLMJudgeError(f"{label} contains a blank line at {path}:{line_number}")
                try:
                    row = json.loads(line, parse_constant=_reject_constant)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise LLMJudgeError(
                        f"Invalid JSON in {label} at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise LLMJudgeError(
                        f"Expected a JSON object in {label} at {path}:{line_number}"
                    )
                yield line_number, row
    except (OSError, UnicodeDecodeError) as exc:
        raise LLMJudgeError(f"Cannot read {label} {path}: {exc}") from exc


def require_sha256(value: str, *, label: str) -> str:
    cleaned = str(value or "").strip().lower()
    if SHA256_RE.fullmatch(cleaned) is None:
        raise LLMJudgeError(f"{label} must be a 64-character lowercase SHA-256")
    return cleaned


def require_text(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise LLMJudgeError(f"{label} must be a string")
    if not allow_empty and not value.strip():
        raise LLMJudgeError(f"{label} must be non-empty")
    return value


def require_canonical_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise LLMJudgeError(f"{label} must be a non-negative integer, got bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LLMJudgeError(f"{label} must be a non-negative integer, got {value!r}") from exc
    if parsed < 0 or (isinstance(value, str) and str(parsed) != value):
        raise LLMJudgeError(f"{label} must be a canonical non-negative integer, got {value!r}")
    return parsed


def format_prompt(lang: str, source: str, hypothesis: str) -> str:
    if lang not in TARGET_LANGUAGE_NAMES:
        raise LLMJudgeError(f"Unsupported target language: {lang!r}")
    require_text(source, label="source")
    require_text(hypothesis, label="hypothesis", allow_empty=True)
    return PROMPT_TEMPLATE.format(
        source_lang=SOURCE_LANGUAGE_NAME,
        target_lang=TARGET_LANGUAGE_NAMES[lang],
        source_seg=source,
        target_seg=hypothesis,
    )


def generation_config(mode: str) -> Dict[str, Any]:
    if mode == "api-default":
        return {}
    if mode == "temperature-zero":
        return {
            "temperature": 0.0,
            "candidateCount": 1,
            "responseMimeType": "text/plain",
        }
    raise LLMJudgeError(f"Unsupported generation-config-mode: {mode!r}")


def expected_system_counts() -> Dict[Tuple[str, str, str, str], int]:
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


def _source_record_from_json(
    row: Mapping[str, Any],
    *,
    line_number: int,
    role: str,
    artifact_sha256: str,
) -> SourceRecord:
    required = (
        "dataset",
        "method",
        "lang",
        "lm",
        "talk_id",
        "talk_sentence_index",
        "source",
        "hypothesis",
        "reference",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise LLMJudgeError(f"{role} row {line_number} is missing fields: {', '.join(missing)}")
    dataset = require_text(row["dataset"], label=f"{role} row {line_number} dataset")
    method = require_text(row["method"], label=f"{role} row {line_number} method")
    lang = require_text(row["lang"], label=f"{role} row {line_number} lang")
    lm = require_text(str(row["lm"]), label=f"{role} row {line_number} lm")
    talk_id = require_text(row["talk_id"], label=f"{role} row {line_number} talk_id")
    talk_sentence_index = require_canonical_nonnegative_int(
        row["talk_sentence_index"],
        label=f"{role} row {line_number} talk_sentence_index",
    )
    source = require_text(row["source"], label=f"{role} row {line_number} source")
    hypothesis = require_text(
        row["hypothesis"],
        label=f"{role} row {line_number} hypothesis",
        allow_empty=True,
    )
    reference = require_text(row["reference"], label=f"{role} row {line_number} reference")
    return SourceRecord(
        dataset=dataset,
        method=method,
        lang=lang,
        lm=lm,
        talk_id=talk_id,
        talk_sentence_index=talk_sentence_index,
        source=source,
        hypothesis=hypothesis,
        reference_sha256=sha256_bytes(reference.encode("utf-8")),
        source_artifact_role=role,
        source_artifact_sha256=artifact_sha256,
        source_record_line=line_number,
        source_record_sha256=canonical_json_sha256(row),
    )


def load_selected_source_records(
    *,
    acl_segments: Path,
    acl_expected_sha256: str,
    medicine_segments: Path,
    medicine_expected_sha256: str,
) -> Tuple[List[SourceRecord], List[Dict[str, Any]]]:
    specifications = (
        ("acl_release_cache", lexical_absolute(acl_segments), acl_expected_sha256, ACL_DATASET),
        (
            "medicine_paper_exact",
            lexical_absolute(medicine_segments),
            medicine_expected_sha256,
            MEDICINE_DATASET,
        ),
    )
    selected: List[SourceRecord] = []
    artifacts: List[Dict[str, Any]] = []
    for role, path, expected_hash, selected_dataset in specifications:
        if not path.is_file():
            raise LLMJudgeError(f"Input artifact is not a file: {path}")
        expected_hash = require_sha256(expected_hash, label=f"{role} expected SHA-256")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise LLMJudgeError(
                f"{role} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
            )
        total_rows = 0
        selected_rows = 0
        dataset_counts: Dict[str, int] = defaultdict(int)
        for line_number, row in iter_jsonl(path, label=role):
            total_rows += 1
            dataset = str(row.get("dataset") or "")
            dataset_counts[dataset] += 1
            if dataset != selected_dataset:
                continue
            selected.append(
                _source_record_from_json(
                    row,
                    line_number=line_number,
                    role=role,
                    artifact_sha256=actual_hash,
                )
            )
            selected_rows += 1
        artifacts.append(
            {
                "role": role,
                "path": str(path),
                "sha256": actual_hash,
                "bytes": path.stat().st_size,
                "total_rows": total_rows,
                "dataset_counts": dict(sorted(dataset_counts.items())),
                "selection": {"dataset_equals": selected_dataset},
                "selected_rows": selected_rows,
            }
        )
    validate_rebuttal_matrix(selected)
    return selected, artifacts


def validate_rebuttal_matrix(records: Sequence[SourceRecord]) -> None:
    expected_counts = expected_system_counts()
    observed_counts: Dict[Tuple[str, str, str, str], int] = defaultdict(int)
    identities = set()
    talks: Dict[Tuple[str, str, str, str], set[str]] = defaultdict(set)
    pairs: Dict[Tuple[str, str, str, str, int], Dict[str, SourceRecord]] = defaultdict(dict)
    for record in records:
        if record.identity in identities:
            raise LLMJudgeError(f"Duplicate segment identity: {record.identity!r}")
        identities.add(record.identity)
        observed_counts[record.system_key] += 1
        talks[record.system_key].add(record.talk_id)
        pair_identity = (
            record.dataset,
            record.lang,
            record.lm,
            record.talk_id,
            record.talk_sentence_index,
        )
        if record.method in pairs[pair_identity]:
            raise LLMJudgeError(f"Duplicate method in paired segment: {pair_identity!r}")
        pairs[pair_identity][record.method] = record
    if observed_counts != expected_counts:
        missing = sorted(set(expected_counts) - set(observed_counts))
        unexpected = sorted(set(observed_counts) - set(expected_counts))
        wrong = sorted(
            (key, expected_counts[key], observed_counts.get(key))
            for key in set(expected_counts) & set(observed_counts)
            if expected_counts[key] != observed_counts[key]
        )
        raise LLMJudgeError(
            "Rebuttal system matrix mismatch; "
            f"missing={missing!r}, unexpected={unexpected!r}, wrong_counts={wrong!r}"
        )
    for key, talk_ids in talks.items():
        if len(talk_ids) != EXPECTED_TALKS_PER_SYSTEM:
            raise LLMJudgeError(
                f"Expected {EXPECTED_TALKS_PER_SYSTEM} talks for {key!r}, got {len(talk_ids)}"
            )
    for pair_identity, by_method in pairs.items():
        if set(by_method) != set(METHODS):
            raise LLMJudgeError(
                f"Incomplete paired segment {pair_identity!r}: found {sorted(by_method)!r}"
            )
        first, second = (by_method[method] for method in METHODS)
        if first.source != second.source or first.reference_sha256 != second.reference_sha256:
            raise LLMJudgeError(f"Source/reference mismatch in paired segment {pair_identity!r}")
    if len(records) != EXPECTED_SEGMENTS:
        raise LLMJudgeError(f"Expected {EXPECTED_SEGMENTS} segments, got {len(records)}")
    acl_count = sum(record.dataset == ACL_DATASET for record in records)
    medicine_count = sum(record.dataset == MEDICINE_DATASET for record in records)
    if acl_count != EXPECTED_ACL_SEGMENTS or medicine_count != EXPECTED_MEDICINE_SEGMENTS:
        raise LLMJudgeError(
            f"Dataset counts mismatch: ACL={acl_count}, medicine={medicine_count}"
        )


def shard_id_for(record: SourceRecord) -> str:
    return f"{record.dataset}__{record.lang}__lm{record.lm}"


def _request_key(
    record: SourceRecord,
    *,
    prompt_sha256: str,
    model: str,
    generation_config_sha256: str,
) -> str:
    payload = {
        "identity": record.identity,
        "source": record.source,
        "hypothesis": record.hypothesis,
        "prompt_sha256": prompt_sha256,
        "model": model,
        "generation_config_sha256": generation_config_sha256,
    }
    return "judge-" + canonical_json_sha256(payload)


def _git_commit(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _repo_root_from_script() -> Path:
    return lexical_absolute(Path(__file__)).parents[4]


def prepare_run(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = lexical_absolute(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise LLMJudgeError(f"Output directory must be absent or empty: {output_dir}")
    if RUN_ID_RE.fullmatch(args.run_id) is None:
        raise LLMJudgeError(
            "run-id must start with an alphanumeric character and contain only "
            "alphanumerics, '.', '_', or '-' (maximum 96 characters)"
        )
    model = require_text(args.model, label="model")
    config = generation_config(args.generation_config_mode)
    config_sha256 = canonical_json_sha256(config)
    prompt_sha256 = sha256_bytes(PROMPT_TEMPLATE.encode("utf-8"))
    records, source_artifacts = load_selected_source_records(
        acl_segments=args.acl_segments,
        acl_expected_sha256=args.acl_segments_sha256,
        medicine_segments=args.medicine_segments,
        medicine_expected_sha256=args.medicine_segments_sha256,
    )
    records = sorted(
        records,
        key=lambda record: (
            record.dataset,
            record.lang,
            int(record.lm),
            record.talk_id,
            record.talk_sentence_index,
            record.method,
        ),
    )
    grouped: Dict[str, List[SourceRecord]] = defaultdict(list)
    for record in records:
        grouped[shard_id_for(record)].append(record)
    if len(grouped) != EXPECTED_PAIRS:
        raise LLMJudgeError(f"Expected {EXPECTED_PAIRS} pair shards, got {len(grouped)}")

    requests_dir = output_dir / "requests"
    sidecars_dir = output_dir / "sidecars"
    states_dir = output_dir / "states"
    requests_dir.mkdir(parents=True, exist_ok=True)
    sidecars_dir.mkdir(parents=True, exist_ok=True)
    states_dir.mkdir(parents=True, exist_ok=True)

    shard_entries: List[Dict[str, Any]] = []
    all_request_keys = set()
    total_prompt_chars = 0
    for shard_id in sorted(grouped):
        shard_records = grouped[shard_id]
        request_lines: List[str] = []
        sidecar_lines: List[str] = []
        shard_prompt_chars = 0
        for record in shard_records:
            prompt = format_prompt(record.lang, record.source, record.hypothesis)
            request_key = _request_key(
                record,
                prompt_sha256=prompt_sha256,
                model=model,
                generation_config_sha256=config_sha256,
            )
            if request_key in all_request_keys:
                raise LLMJudgeError(f"Duplicate opaque request key: {request_key}")
            all_request_keys.add(request_key)
            api_request = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ]
            }
            if config:
                api_request["generation_config"] = config
            request_lines.append(json_line({"key": request_key, "request": api_request}))
            judge_input_sha256 = canonical_json_sha256(
                {
                    "source_lang": SOURCE_LANGUAGE_NAME,
                    "target_lang": TARGET_LANGUAGE_NAMES[record.lang],
                    "source": record.source,
                    "translation": record.hypothesis,
                    "prompt_template_sha256": prompt_sha256,
                }
            )
            sidecar_lines.append(
                json_line(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "request_key": request_key,
                        "shard_id": shard_id,
                        "dataset": record.dataset,
                        "method": record.method,
                        "lang": record.lang,
                        "lm": record.lm,
                        "talk_id": record.talk_id,
                        "talk_sentence_index": record.talk_sentence_index,
                        "source": record.source,
                        "hypothesis": record.hypothesis,
                        "source_sha256": sha256_bytes(record.source.encode("utf-8")),
                        "hypothesis_sha256": sha256_bytes(record.hypothesis.encode("utf-8")),
                        "reference_sha256": record.reference_sha256,
                        "prompt_sha256": prompt_sha256,
                        "judge_input_sha256": judge_input_sha256,
                        "api_request_sha256": canonical_json_sha256(api_request),
                        "source_artifact_role": record.source_artifact_role,
                        "source_artifact_sha256": record.source_artifact_sha256,
                        "source_record_line": record.source_record_line,
                        "source_record_sha256": record.source_record_sha256,
                    }
                )
            )
            shard_prompt_chars += len(prompt)
        request_path = requests_dir / f"{shard_id}.jsonl"
        sidecar_path = sidecars_dir / f"{shard_id}.jsonl"
        atomic_write_text(request_path, "".join(request_lines))
        atomic_write_text(sidecar_path, "".join(sidecar_lines))
        expected_per_method = EXPECTED_SEGMENTS_PER_SYSTEM[shard_records[0].dataset]
        if len(shard_records) != 2 * expected_per_method:
            raise LLMJudgeError(
                f"Shard {shard_id} has {len(shard_records)} requests, expected {2 * expected_per_method}"
            )
        entry = {
            "shard_id": shard_id,
            "dataset": shard_records[0].dataset,
            "lang": shard_records[0].lang,
            "lm": shard_records[0].lm,
            "methods": list(METHODS),
            "request_count": len(shard_records),
            "prompt_characters": shard_prompt_chars,
            "heuristic_input_tokens_chars_div_4": math.ceil(shard_prompt_chars / 4),
            "request_path": str(request_path.relative_to(output_dir)),
            "request_bytes": request_path.stat().st_size,
            "request_sha256": sha256_file(request_path),
            "sidecar_path": str(sidecar_path.relative_to(output_dir)),
            "sidecar_bytes": sidecar_path.stat().st_size,
            "sidecar_sha256": sha256_file(sidecar_path),
        }
        state = {
            "schema_version": SCHEMA_VERSION,
            "run_id": args.run_id,
            "shard_id": shard_id,
            "status": "PREPARED",
            "model": model,
            "generation_config_sha256": config_sha256,
            "request_path": entry["request_path"],
            "request_sha256": entry["request_sha256"],
            "request_bytes": entry["request_bytes"],
            "request_count": entry["request_count"],
            "history": [{"status": "PREPARED", "at_utc": utc_now()}],
        }
        atomic_write_json(states_dir / f"{shard_id}.json", state)
        shard_entries.append(entry)
        total_prompt_chars += shard_prompt_chars

    runner = lexical_absolute(Path(__file__))
    run_config = {
        "run_id": args.run_id,
        "model": model,
        "prompt_sha256": prompt_sha256,
        "generation_config": config,
        "source_artifact_hashes": [artifact["sha256"] for artifact in source_artifacts],
        "selected_datasets": [ACL_DATASET, MEDICINE_DATASET],
        "shard_request_hashes": [entry["request_sha256"] for entry in shard_entries],
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "run_id": args.run_id,
        "model": model,
        "methodology": {
            "metric": "reference-free LLM as a judge",
            "source_language": SOURCE_LANGUAGE_NAME,
            "target_language_names": TARGET_LANGUAGE_NAMES,
            "prompt_template": PROMPT_TEMPLATE,
            "prompt_sha256": prompt_sha256,
            "prompt_source": WMT_PAPER_URL,
            "prompt_source_location": "Appendix A, LLM Prompt for Task 1",
            "generation_config": config,
            "generation_config_sha256": config_sha256,
            "generation_config_mode": args.generation_config_mode,
            "thinking_config": "model default dynamic thinking; not overridden",
            "reference_passed_to_model": False,
            "request_deduplication": False,
        },
        "run_config_sha256": canonical_json_sha256(run_config),
        "source_artifacts": source_artifacts,
        "matrix": {
            "systems": EXPECTED_SYSTEMS,
            "pairs": EXPECTED_PAIRS,
            "segments": EXPECTED_SEGMENTS,
            "acl_segments": EXPECTED_ACL_SEGMENTS,
            "medicine_segments": EXPECTED_MEDICINE_SEGMENTS,
            "shards": EXPECTED_PAIRS,
            "empty_hypotheses": sum(not record.hypothesis for record in records),
        },
        "size_estimate": {
            "prompt_characters": total_prompt_chars,
            "heuristic_input_tokens_chars_div_4": math.ceil(total_prompt_chars / 4),
            "warning": "character/4 is a rough planning heuristic, not Gemini token accounting",
        },
        "software": {
            "runner": str(runner),
            "runner_sha256": sha256_file(runner),
            "git_commit": _git_commit(_repo_root_from_script()),
            "python": platform.python_version(),
            "host": socket.gethostname(),
        },
        "shards": shard_entries,
    }
    manifest_path = output_dir / "run_manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "run_config_sha256": manifest["run_config_sha256"],
                "systems": EXPECTED_SYSTEMS,
                "pairs": EXPECTED_PAIRS,
                "segments": EXPECTED_SEGMENTS,
                "shards": EXPECTED_PAIRS,
            },
            sort_keys=True,
        )
    )
    return manifest


def load_run_manifest(output_dir: Path) -> Dict[str, Any]:
    manifest_path = lexical_absolute(output_dir) / "run_manifest.json"
    manifest = read_json(manifest_path, label="run manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise LLMJudgeError(f"Unsupported run manifest schema: {manifest.get('schema_version')!r}")
    if len(manifest.get("shards") or []) != EXPECTED_PAIRS:
        raise LLMJudgeError("Run manifest does not describe the complete 16-shard matrix")
    return manifest


def _manifest_shard(manifest: Mapping[str, Any], shard_id: str) -> Dict[str, Any]:
    matches = [entry for entry in manifest.get("shards", []) if entry.get("shard_id") == shard_id]
    if len(matches) != 1:
        raise LLMJudgeError(f"Run manifest does not contain exactly one shard {shard_id!r}")
    return dict(matches[0])


def read_private_api_key(path: Path) -> str:
    if path.is_symlink():
        raise LLMJudgeError(f"API key path must not be a symlink: {path}")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise LLMJudgeError(f"Cannot stat API key file {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise LLMJudgeError(f"API key path must be a regular file: {path}")
    if metadata.st_uid != os.getuid():
        raise LLMJudgeError(f"API key file must be owned by uid {os.getuid()}: {path}")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise LLMJudgeError(f"API key file must not grant group/other permissions: {path}")
    if metadata.st_size <= 0 or metadata.st_size > 4096:
        raise LLMJudgeError(f"API key file size is invalid: {path}")
    try:
        key = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise LLMJudgeError(f"Cannot read API key file {path}: {exc}") from exc
    if not key or any(character.isspace() for character in key):
        raise LLMJudgeError(f"API key file must contain exactly one non-whitespace token: {path}")
    return key


@contextmanager
def locked_state(output_dir: Path, shard_id: str) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    states_dir = output_dir.resolve() / "states"
    state_path = states_dir / f"{shard_id}.json"
    lock_path = states_dir / f"{shard_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        state = read_json(state_path, label=f"state for {shard_id}")
        try:
            yield state_path, state
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def update_state(
    state_path: Path,
    state: Dict[str, Any],
    *,
    status: str,
    details: Optional[Mapping[str, Any]] = None,
) -> None:
    state["status"] = status
    event = {"status": status, "at_utc": utc_now()}
    if details:
        event.update(details)
        state.update(details)
    history = state.setdefault("history", [])
    if not isinstance(history, list):
        raise LLMJudgeError(f"State history is malformed: {state_path}")
    history.append(event)
    atomic_write_json(state_path, state)


def _load_google_genai() -> Tuple[Any, Any, str]:
    try:
        from google import genai
        from google.genai import types
        import google.genai as google_genai
    except ImportError as exc:
        raise LLMJudgeError(
            "google-genai is required for upload/submission; use the pinned Taurus venv"
        ) from exc
    return genai, types, getattr(google_genai, "__version__", "unknown")


def _safe_exception(exc: BaseException) -> Dict[str, str]:
    message = str(exc)
    if len(message) > 2_000:
        message = message[:2_000] + "...[truncated]"
    return {"exception_type": type(exc).__name__, "exception_message": message}


def submit_shard(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    manifest = load_run_manifest(output_dir)
    if args.confirm_run_config_sha256 != manifest.get("run_config_sha256"):
        raise LLMJudgeError(
            "--confirm-run-config-sha256 must exactly match the prepared run manifest"
        )
    shard = _manifest_shard(manifest, args.shard_id)
    request_path = output_dir / shard["request_path"]
    if sha256_file(request_path) != shard["request_sha256"]:
        raise LLMJudgeError(f"Prepared request file hash mismatch: {request_path}")
    api_key = read_private_api_key(args.api_key_file)
    genai, types, sdk_version = _load_google_genai()
    client = genai.Client(vertexai=False, api_key=api_key)
    with locked_state(output_dir, args.shard_id) as (state_path, state):
        current_status = state.get("status")
        if current_status not in {"PREPARED", "UPLOADED"}:
            raise LLMJudgeError(
                f"Shard {args.shard_id} is {current_status!r}, not PREPARED/UPLOADED; "
                "refusing a duplicate or ambiguous submission"
            )
        display_stem = re.sub(r"[^A-Za-z0-9_-]+", "-", manifest["run_id"]).strip("-")
        display_name = (
            f"{display_stem[:42]}-{args.shard_id[:42]}-{shard['request_sha256'][:10]}"
        )
        upload_display_name = f"input-{display_name}"
        if current_status == "PREPARED":
            attempt_uuid = str(uuid.uuid4())
            update_state(
                state_path,
                state,
                status="UPLOAD_INTENT",
                details={
                    "attempt_uuid": attempt_uuid,
                    "upload_display_name": upload_display_name,
                    "sdk": "google-genai",
                    "sdk_version": sdk_version,
                },
            )
            try:
                uploaded = client.files.upload(
                    file=str(request_path),
                    config=types.UploadFileConfig(
                        display_name=upload_display_name,
                        mime_type="jsonl",
                    ),
                )
            except Exception as exc:
                update_state(
                    state_path,
                    state,
                    status="UPLOAD_UNCERTAIN",
                    details=_safe_exception(exc),
                )
                raise LLMJudgeError(
                    "Upload outcome is uncertain; state was frozen and must be audited before retry"
                ) from exc
            uploaded_name = require_text(getattr(uploaded, "name", None), label="uploaded file name")
            update_state(
                state_path,
                state,
                status="UPLOADED",
                details={"uploaded_file_name": uploaded_name},
            )
        else:
            uploaded_name = require_text(
                state.get("uploaded_file_name"),
                label=f"{args.shard_id} uploaded file name",
            )
        update_state(
            state_path,
            state,
            status="SUBMIT_INTENT",
            details={"batch_display_name": display_name},
        )
        try:
            job = client.batches.create(
                model=manifest["model"],
                src=uploaded_name,
                config=types.CreateBatchJobConfig(
                    display_name=display_name,
                    http_options=types.HttpOptions(
                        timeout=300_000,
                        retry_options=types.HttpRetryOptions(attempts=1),
                    ),
                ),
            )
        except Exception as exc:
            update_state(
                state_path,
                state,
                status="SUBMISSION_UNCERTAIN",
                details=_safe_exception(exc),
            )
            raise LLMJudgeError(
                "Batch create outcome is uncertain; create must not be retried automatically"
            ) from exc
        job_name = require_text(getattr(job, "name", None), label="batch job name")
        job_state = getattr(getattr(job, "state", None), "name", None)
        update_state(
            state_path,
            state,
            status="SUBMITTED",
            details={
                "job_name": job_name,
                "sdk_reported_state": job_state,
                "submitted_at_utc": utc_now(),
            },
        )
    print(json.dumps({"shard_id": args.shard_id, "job_name": job_name}, sort_keys=True))


def _raw_api_json(api_key: str, url: str, *, timeout_seconds: float) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers={"x-goog-api-key": api_key})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read(16_384).decode("utf-8", errors="replace")
        raise LLMJudgeError(f"Gemini REST HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMJudgeError(f"Gemini REST request failed: {type(exc).__name__}: {exc}") from exc
    try:
        value = json.loads(payload, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise LLMJudgeError("Gemini REST response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise LLMJudgeError("Gemini REST response is not a JSON object")
    return value


def _raw_job_status(api_key: str, job_name: str, *, timeout_seconds: float) -> Dict[str, Any]:
    quoted_name = "/".join(urllib.parse.quote(part, safe="") for part in job_name.split("/"))
    return _raw_api_json(
        api_key,
        f"https://generativelanguage.googleapis.com/v1beta/{quoted_name}",
        timeout_seconds=timeout_seconds,
    )


def _job_state_name(raw_status: Mapping[str, Any]) -> str:
    candidates = (
        raw_status.get("state"),
        (raw_status.get("metadata") or {}).get("state")
        if isinstance(raw_status.get("metadata"), dict)
        else None,
    )
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    raise LLMJudgeError("Raw Batch status contains no state")


def _response_file_name(raw_status: Mapping[str, Any]) -> Optional[str]:
    paths = (
        ("dest", "fileName"),
        ("metadata", "output", "responsesFile"),
        ("response", "responsesFile"),
        ("outputConfig", "fileName"),
    )
    for path in paths:
        value: Any = raw_status
        for component in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(component)
        if isinstance(value, str) and value:
            return value
    return None


def poll_shard(
    *,
    output_dir: Path,
    shard_id: str,
    api_key: str,
    timeout_seconds: float,
) -> str:
    manifest = load_run_manifest(output_dir)
    _manifest_shard(manifest, shard_id)
    with locked_state(output_dir, shard_id) as (state_path, state):
        current = str(state.get("status") or "")
        if current in {"PREPARED", "UPLOAD_INTENT", "UPLOAD_UNCERTAIN", "UPLOADED", "SUBMIT_INTENT", "SUBMISSION_UNCERTAIN"}:
            raise LLMJudgeError(f"Shard {shard_id} cannot be polled from state {current!r}")
        if current == "DOWNLOADED":
            return current
        job_name = require_text(state.get("job_name"), label=f"{shard_id} job name")
        raw_status = _raw_job_status(api_key, job_name, timeout_seconds=timeout_seconds)
        raw_status_path = output_dir / "status" / f"{shard_id}.json"
        atomic_write_json(raw_status_path, raw_status)
        raw_state = _job_state_name(raw_status)
        details: Dict[str, Any] = {
            "raw_api_state": raw_state,
            "raw_status_path": str(raw_status_path.relative_to(output_dir)),
            "raw_status_sha256": sha256_file(raw_status_path),
            "last_polled_at_utc": utc_now(),
        }
        response_file_name = _response_file_name(raw_status)
        if response_file_name:
            details["response_file_name"] = response_file_name
        new_status = "SUCCEEDED" if raw_state in SUCCESS_STATES else raw_state
        update_state(state_path, state, status=new_status, details=details)
        return new_status


def _download_response_file(
    *, api_key: str, file_name: str, timeout_seconds: float
) -> bytes:
    quoted_name = "/".join(urllib.parse.quote(part, safe="") for part in file_name.split("/"))
    url = (
        "https://generativelanguage.googleapis.com/download/v1beta/"
        f"{quoted_name}:download?alt=media"
    )
    request = urllib.request.Request(url, headers={"x-goog-api-key": api_key})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read(16_384).decode("utf-8", errors="replace")
        raise LLMJudgeError(f"Gemini result download HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMJudgeError(f"Gemini result download failed: {type(exc).__name__}: {exc}") from exc


def download_shard(
    *,
    output_dir: Path,
    shard_id: str,
    api_key: str,
    timeout_seconds: float,
) -> Path:
    manifest = load_run_manifest(output_dir)
    shard = _manifest_shard(manifest, shard_id)
    with locked_state(output_dir, shard_id) as (state_path, state):
        if state.get("status") == "DOWNLOADED":
            existing = output_dir / str(state.get("response_path"))
            if existing.is_file() and sha256_file(existing) == state.get("response_sha256"):
                return existing
            raise LLMJudgeError(f"Downloaded response evidence is missing or changed: {existing}")
        raw_state = str(state.get("raw_api_state") or state.get("status") or "")
        if raw_state not in SUCCESS_STATES and state.get("status") != "SUCCEEDED":
            raise LLMJudgeError(f"Shard {shard_id} is not succeeded: {raw_state!r}")
        file_name = require_text(
            state.get("response_file_name"), label=f"{shard_id} response file name"
        )
        payload = _download_response_file(
            api_key=api_key,
            file_name=file_name,
            timeout_seconds=timeout_seconds,
        )
        response_path = output_dir / "responses" / f"{shard_id}.jsonl"
        atomic_write_bytes(response_path, payload)
        response_rows = sum(1 for _ in iter_jsonl(response_path, label=f"{shard_id} response"))
        if response_rows != int(shard["request_count"]):
            raise LLMJudgeError(
                f"Downloaded {shard_id} response has {response_rows} rows, "
                f"expected {shard['request_count']}"
            )
        update_state(
            state_path,
            state,
            status="DOWNLOADED",
            details={
                "response_path": str(response_path.relative_to(output_dir)),
                "response_sha256": sha256_file(response_path),
                "response_bytes": response_path.stat().st_size,
                "response_rows": response_rows,
                "downloaded_at_utc": utc_now(),
            },
        )
        return response_path


def status_command(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    manifest = load_run_manifest(output_dir)
    api_key = read_private_api_key(args.api_key_file)
    shard_ids = [args.shard_id] if args.shard_id else [entry["shard_id"] for entry in manifest["shards"]]
    results = []
    for shard_id in shard_ids:
        try:
            status = poll_shard(
                output_dir=output_dir,
                shard_id=shard_id,
                api_key=api_key,
                timeout_seconds=args.timeout_seconds,
            )
            if args.download_succeeded and status == "SUCCEEDED":
                download_shard(
                    output_dir=output_dir,
                    shard_id=shard_id,
                    api_key=api_key,
                    timeout_seconds=args.timeout_seconds,
                )
                status = "DOWNLOADED"
            results.append({"shard_id": shard_id, "status": status})
        except LLMJudgeError as exc:
            results.append({"shard_id": shard_id, "status": "ERROR", "error": str(exc)})
            if args.fail_fast:
                raise
    print(json.dumps(results, ensure_ascii=False, sort_keys=True))


def monitor_command(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    manifest = load_run_manifest(output_dir)
    api_key = read_private_api_key(args.api_key_file)
    deadline = time.monotonic() + args.max_hours * 3600.0
    shard_ids = [entry["shard_id"] for entry in manifest["shards"]]
    while True:
        statuses: Dict[str, str] = {}
        for shard_id in shard_ids:
            state = read_json(output_dir / "states" / f"{shard_id}.json", label="shard state")
            current = str(state.get("status") or "")
            if current == "DOWNLOADED":
                statuses[shard_id] = current
                continue
            if current in {"PREPARED", "UPLOAD_INTENT", "UPLOAD_UNCERTAIN", "UPLOADED", "SUBMIT_INTENT", "SUBMISSION_UNCERTAIN"}:
                statuses[shard_id] = current
                continue
            polled = poll_shard(
                output_dir=output_dir,
                shard_id=shard_id,
                api_key=api_key,
                timeout_seconds=args.timeout_seconds,
            )
            if polled == "SUCCEEDED":
                download_shard(
                    output_dir=output_dir,
                    shard_id=shard_id,
                    api_key=api_key,
                    timeout_seconds=args.timeout_seconds,
                )
                polled = "DOWNLOADED"
            statuses[shard_id] = polled
        print(json.dumps({"at_utc": utc_now(), "statuses": statuses}, sort_keys=True), flush=True)
        if all(status == "DOWNLOADED" for status in statuses.values()):
            return
        failed = {key: value for key, value in statuses.items() if value in TERMINAL_STATES and value not in SUCCESS_STATES}
        if failed:
            raise LLMJudgeError(f"Terminal Batch failures: {failed!r}")
        if time.monotonic() >= deadline:
            raise LLMJudgeError(f"Monitor exceeded max-hours={args.max_hours}")
        time.sleep(args.poll_seconds)


def parse_score_text(text: Any) -> int:
    if not isinstance(text, str):
        raise LLMJudgeError("Visible candidate output must be a string")
    cleaned = text.strip(" \t\r\n")
    if SCORE_RE.fullmatch(cleaned) is None:
        raise LLMJudgeError(f"Judge output is not a single integer in [0,100]: {text!r}")
    return int(cleaned)


def _parse_success_response(response: Mapping[str, Any]) -> Dict[str, Any]:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise LLMJudgeError("Response must contain exactly one candidate")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise LLMJudgeError("Response candidate must be an object")
    if candidate.get("finishReason") != "STOP":
        raise LLMJudgeError(f"Candidate finishReason is not STOP: {candidate.get('finishReason')!r}")
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list) or not parts:
        raise LLMJudgeError("Candidate content has no parts")
    visible_texts: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            raise LLMJudgeError("Candidate content part must be an object")
        if part.get("thought") is True:
            continue
        if "text" in part:
            visible_texts.append(require_text(part["text"], label="candidate text", allow_empty=True))
    if len(visible_texts) != 1:
        raise LLMJudgeError(
            f"Response must contain exactly one non-thinking text part, got {len(visible_texts)}"
        )
    raw_text = visible_texts[0]
    score = parse_score_text(raw_text)
    model_version = require_text(response.get("modelVersion"), label="response modelVersion")
    usage = response.get("usageMetadata")
    if not isinstance(usage, dict):
        raise LLMJudgeError("usageMetadata must be an object")
    usage_tokens: Dict[str, int] = {}
    for api_field, output_field, required in (
        ("promptTokenCount", "prompt_tokens", True),
        ("candidatesTokenCount", "candidate_tokens", True),
        ("thoughtsTokenCount", "thinking_tokens", False),
        ("totalTokenCount", "total_tokens", True),
    ):
        if api_field not in usage:
            if required:
                raise LLMJudgeError(f"usageMetadata is missing {api_field}")
            usage_tokens[output_field] = 0
            continue
        usage_tokens[output_field] = require_canonical_nonnegative_int(
            usage[api_field], label=f"usageMetadata.{api_field}"
        )
    return {
        "judge_score": score,
        "judge_raw_text": raw_text,
        "model_version": model_version,
        "finish_reason": candidate["finishReason"],
        "usage_metadata": usage,
        "usage_tokens": usage_tokens,
        "response_id": response.get("responseId"),
    }


def _load_sidecar(path: Path, expected_sha256: str) -> Dict[str, Dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise LLMJudgeError(f"Sidecar SHA-256 mismatch: {path}")
    indexed: Dict[str, Dict[str, Any]] = {}
    for line_number, row in iter_jsonl(path, label="judge sidecar"):
        key = require_text(row.get("request_key"), label=f"sidecar row {line_number} request_key")
        if key in indexed:
            raise LLMJudgeError(f"Duplicate sidecar request key: {key}")
        indexed[key] = row
    return indexed


def _load_response(path: Path, expected_sha256: str) -> Dict[str, Dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise LLMJudgeError(f"Response SHA-256 mismatch: {path}")
    indexed: Dict[str, Dict[str, Any]] = {}
    for line_number, row in iter_jsonl(path, label="Gemini Batch response"):
        key = require_text(row.get("key"), label=f"response row {line_number} key")
        if key in indexed:
            raise LLMJudgeError(f"Duplicate response key: {key}")
        indexed[key] = row
    return indexed


def collect_segments(output_dir: Path, manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    seen_keys = set()
    model_versions = set()
    for shard in manifest["shards"]:
        shard_id = shard["shard_id"]
        state_path = output_dir / "states" / f"{shard_id}.json"
        state = read_json(state_path, label=f"state for {shard_id}")
        if state.get("status") != "DOWNLOADED":
            raise LLMJudgeError(f"Shard {shard_id} is not DOWNLOADED: {state.get('status')!r}")
        sidecars = _load_sidecar(
            output_dir / shard["sidecar_path"],
            shard["sidecar_sha256"],
        )
        responses = _load_response(
            output_dir / state["response_path"],
            require_sha256(state["response_sha256"], label=f"{shard_id} response SHA-256"),
        )
        if set(sidecars) != set(responses):
            raise LLMJudgeError(
                f"Response key coverage mismatch for {shard_id}; "
                f"missing={sorted(set(sidecars) - set(responses))[:10]!r}, "
                f"unknown={sorted(set(responses) - set(sidecars))[:10]!r}"
            )
        for request_key in sorted(sidecars):
            if request_key in seen_keys:
                raise LLMJudgeError(f"Request key occurs across multiple shards: {request_key}")
            seen_keys.add(request_key)
            sidecar = sidecars[request_key]
            batch_row = responses[request_key]
            has_response = "response" in batch_row
            has_error = "error" in batch_row
            if has_response == has_error:
                errors.append(
                    {
                        "shard_id": shard_id,
                        "request_key": request_key,
                        "error": "Batch row must contain exactly one of response or error",
                        "raw_batch_row": batch_row,
                    }
                )
                continue
            if has_error:
                errors.append(
                    {
                        "shard_id": shard_id,
                        "request_key": request_key,
                        "error": "Gemini per-request error",
                        "raw_batch_row": batch_row,
                    }
                )
                continue
            response = batch_row["response"]
            if not isinstance(response, dict):
                errors.append(
                    {
                        "shard_id": shard_id,
                        "request_key": request_key,
                        "error": "response field is not an object",
                        "raw_batch_row": batch_row,
                    }
                )
                continue
            try:
                parsed = _parse_success_response(response)
            except LLMJudgeError as exc:
                errors.append(
                    {
                        "shard_id": shard_id,
                        "request_key": request_key,
                        "error": str(exc),
                        "raw_batch_row": batch_row,
                    }
                )
                continue
            model_versions.add(parsed["model_version"])
            collected.append(
                {
                    **sidecar,
                    **parsed,
                    "judge_model": manifest["model"],
                    "generation_config_sha256": manifest["methodology"][
                        "generation_config_sha256"
                    ],
                    "response_artifact_sha256": state["response_sha256"],
                }
            )
    errors_path = output_dir / "collection_errors.jsonl"
    if errors:
        atomic_write_text(errors_path, "".join(json_line(row) for row in errors))
        raise LLMJudgeError(
            f"Collection found {len(errors)} invalid/error responses; see {errors_path}"
        )
    errors_path.unlink(missing_ok=True)
    if len(collected) != EXPECTED_SEGMENTS:
        raise LLMJudgeError(f"Expected {EXPECTED_SEGMENTS} collected segments, got {len(collected)}")
    if len(model_versions) != 1:
        raise LLMJudgeError(f"Expected one resolved modelVersion, got {sorted(model_versions)!r}")
    return sorted(
        collected,
        key=lambda row: (
            row["dataset"],
            row["lang"],
            int(row["lm"]),
            row["talk_id"],
            int(row["talk_sentence_index"]),
            row["method"],
        ),
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise LLMJudgeError("Cannot compute a mean over zero values")
    return float(statistics.fmean(values))


def build_summary_rows(
    segments: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for segment in segments:
        grouped[(segment["dataset"], segment["method"], segment["lang"], segment["lm"])].append(
            segment
        )
    rows = []
    for key in sorted(grouped):
        system_segments = grouped[key]
        scores = [float(segment["judge_score"]) for segment in system_segments]
        by_talk: Dict[str, List[float]] = defaultdict(list)
        for segment in system_segments:
            by_talk[str(segment["talk_id"])].append(float(segment["judge_score"]))
        rows.append(
            {
                "dataset": key[0],
                "method": key[1],
                "lang": key[2],
                "lm": key[3],
                "talks": len(by_talk),
                "segments": len(system_segments),
                "llm_judge_mean": _mean(scores),
                "llm_judge_talk_macro_mean": _mean([_mean(values) for values in by_talk.values()]),
                "judge_model": manifest["model"],
                "resolved_model_version": system_segments[0]["model_version"],
                "prompt_sha256": manifest["methodology"]["prompt_sha256"],
                "generation_config_sha256": manifest["methodology"][
                    "generation_config_sha256"
                ],
                "run_config_sha256": manifest["run_config_sha256"],
            }
        )
    if len(rows) != EXPECTED_SYSTEMS:
        raise LLMJudgeError(f"Expected {EXPECTED_SYSTEMS} summary rows, got {len(rows)}")
    return rows


def build_paired_rows(segments: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Dict[Tuple[str, int], Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for segment in segments:
        group_key = (segment["dataset"], segment["lang"], segment["lm"])
        segment_key = (str(segment["talk_id"]), int(segment["talk_sentence_index"]))
        method_map = grouped[group_key][segment["method"]]
        if segment_key in method_map:
            raise LLMJudgeError(f"Duplicate paired segment key: {group_key!r}/{segment_key!r}")
        method_map[segment_key] = segment
    rows = []
    for group_key in sorted(grouped):
        by_method = grouped[group_key]
        if set(by_method) != set(METHODS):
            raise LLMJudgeError(f"Incomplete method pair: {group_key!r}")
        baseline = by_method["InfiniSST"]
        rasst = by_method["RASST"]
        if set(baseline) != set(rasst):
            raise LLMJudgeError(f"Unpaired segment keys: {group_key!r}")
        baseline_scores: List[float] = []
        rasst_scores: List[float] = []
        deltas: List[float] = []
        wins = ties = losses = 0
        talk_ids = set()
        for key in sorted(baseline):
            baseline_segment = baseline[key]
            rasst_segment = rasst[key]
            if baseline_segment["source"] != rasst_segment["source"]:
                raise LLMJudgeError(f"Paired source mismatch: {group_key!r}/{key!r}")
            baseline_score = float(baseline_segment["judge_score"])
            rasst_score = float(rasst_segment["judge_score"])
            delta = rasst_score - baseline_score
            baseline_scores.append(baseline_score)
            rasst_scores.append(rasst_score)
            deltas.append(delta)
            talk_ids.add(key[0])
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1
            else:
                ties += 1
        rows.append(
            {
                "dataset": group_key[0],
                "lang": group_key[1],
                "lm": group_key[2],
                "rasst_method": "RASST",
                "infinisst_method": "InfiniSST",
                "paired_talks": len(talk_ids),
                "paired_segments": len(deltas),
                "rasst_llm_judge_mean": _mean(rasst_scores),
                "infinisst_llm_judge_mean": _mean(baseline_scores),
                "delta_rasst_minus_infinisst": _mean(deltas),
                "paired_delta_stddev": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "rasst_wins": wins,
                "ties": ties,
                "infinisst_wins": losses,
            }
        )
    if len(rows) != EXPECTED_PAIRS:
        raise LLMJudgeError(f"Expected {EXPECTED_PAIRS} paired rows, got {len(rows)}")
    return rows


def build_talk_paired_rows(segments: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for segment in segments:
        key = (segment["dataset"], segment["lang"], segment["lm"], segment["talk_id"])
        grouped[key][segment["method"]].append(float(segment["judge_score"]))
    rows = []
    for key in sorted(grouped):
        by_method = grouped[key]
        if set(by_method) != set(METHODS):
            raise LLMJudgeError(f"Incomplete method pair in talk aggregate: {key!r}")
        baseline_mean = _mean(by_method["InfiniSST"])
        rasst_mean = _mean(by_method["RASST"])
        if len(by_method["InfiniSST"]) != len(by_method["RASST"]):
            raise LLMJudgeError(f"Talk segment count mismatch: {key!r}")
        rows.append(
            {
                "dataset": key[0],
                "lang": key[1],
                "lm": key[2],
                "talk_id": key[3],
                "paired_segments": len(by_method["RASST"]),
                "rasst_llm_judge_mean": rasst_mean,
                "infinisst_llm_judge_mean": baseline_mean,
                "delta_rasst_minus_infinisst": rasst_mean - baseline_mean,
            }
        )
    if len(rows) != EXPECTED_PAIRS * EXPECTED_TALKS_PER_SYSTEM:
        raise LLMJudgeError(
            f"Expected {EXPECTED_PAIRS * EXPECTED_TALKS_PER_SYSTEM} talk-paired rows, got {len(rows)}"
        )
    return rows


def build_group_rows(paired_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    definitions: List[Tuple[str, Any]] = [
        ("acl_de_4lm_cell_macro", lambda row: row["dataset"] == ACL_DATASET and row["lang"] == "de"),
        ("acl_ja_4lm_cell_macro", lambda row: row["dataset"] == ACL_DATASET and row["lang"] == "ja"),
        ("acl_zh_4lm_cell_macro", lambda row: row["dataset"] == ACL_DATASET and row["lang"] == "zh"),
        ("acl_12cell_macro", lambda row: row["dataset"] == ACL_DATASET),
        ("medicine_de_4lm_cell_macro", lambda row: row["dataset"] == MEDICINE_DATASET),
        ("all_16cell_macro_descriptive", lambda row: True),
    ]
    rows = []
    for group_name, predicate in definitions:
        selected = [row for row in paired_rows if predicate(row)]
        rows.append(
            {
                "group": group_name,
                "cells": len(selected),
                "rasst_cell_macro_mean": _mean(
                    [float(row["rasst_llm_judge_mean"]) for row in selected]
                ),
                "infinisst_cell_macro_mean": _mean(
                    [float(row["infinisst_llm_judge_mean"]) for row in selected]
                ),
                "delta_cell_macro_rasst_minus_infinisst": _mean(
                    [float(row["delta_rasst_minus_infinisst"]) for row in selected]
                ),
                "positive_cells": sum(float(row["delta_rasst_minus_infinisst"]) > 0 for row in selected),
                "zero_cells": sum(float(row["delta_rasst_minus_infinisst"]) == 0 for row in selected),
                "negative_cells": sum(float(row["delta_rasst_minus_infinisst"]) < 0 for row in selected),
            }
        )
    return rows


def _format_tsv_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.10f}"
    return str(value)


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _format_tsv_value(row[field]) for field in fields})
    atomic_write_text(path, output.getvalue())


def collect_command(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    manifest = load_run_manifest(output_dir)
    segments = collect_segments(output_dir, manifest)
    summary_rows = build_summary_rows(segments, manifest)
    paired_rows = build_paired_rows(segments)
    talk_rows = build_talk_paired_rows(segments)
    group_rows = build_group_rows(paired_rows)
    atomic_write_text(
        output_dir / "segments.jsonl",
        "".join(json_line(row) for row in segments),
    )
    write_tsv(
        output_dir / "summary.tsv",
        summary_rows,
        (
            "dataset",
            "method",
            "lang",
            "lm",
            "talks",
            "segments",
            "llm_judge_mean",
            "llm_judge_talk_macro_mean",
            "judge_model",
            "resolved_model_version",
            "prompt_sha256",
            "generation_config_sha256",
            "run_config_sha256",
        ),
    )
    write_tsv(
        output_dir / "paired.tsv",
        paired_rows,
        (
            "dataset",
            "lang",
            "lm",
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
        ),
    )
    write_tsv(
        output_dir / "talk_paired.tsv",
        talk_rows,
        (
            "dataset",
            "lang",
            "lm",
            "talk_id",
            "paired_segments",
            "rasst_llm_judge_mean",
            "infinisst_llm_judge_mean",
            "delta_rasst_minus_infinisst",
        ),
    )
    write_tsv(
        output_dir / "group_summary.tsv",
        group_rows,
        (
            "group",
            "cells",
            "rasst_cell_macro_mean",
            "infinisst_cell_macro_mean",
            "delta_cell_macro_rasst_minus_infinisst",
            "positive_cells",
            "zero_cells",
            "negative_cells",
        ),
    )
    artifacts = {}
    for name in ("segments.jsonl", "summary.tsv", "paired.tsv", "talk_paired.tsv", "group_summary.tsv"):
        path = output_dir / name
        artifacts[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    usage_totals = {
        field: sum(int(segment["usage_tokens"][field]) for segment in segments)
        for field in ("prompt_tokens", "candidate_tokens", "thinking_tokens", "total_tokens")
    }
    collection_manifest = {
        "schema_version": SCHEMA_VERSION,
        "collected_at_utc": utc_now(),
        "run_config_sha256": manifest["run_config_sha256"],
        "segments": len(segments),
        "systems": len(summary_rows),
        "pairs": len(paired_rows),
        "talk_pairs": len(talk_rows),
        "resolved_model_version": segments[0]["model_version"],
        "usage_totals": usage_totals,
        "artifacts": artifacts,
    }
    atomic_write_json(output_dir / "collection_manifest.json", collection_manifest)
    print(json.dumps(collection_manifest, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare the strict 16-shard rebuttal matrix.")
    prepare.add_argument("--acl-segments", required=True, type=Path)
    prepare.add_argument("--acl-segments-sha256", required=True)
    prepare.add_argument("--medicine-segments", required=True, type=Path)
    prepare.add_argument("--medicine-segments-sha256", required=True)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--model", required=True)
    prepare.add_argument(
        "--generation-config-mode",
        choices=("api-default", "temperature-zero"),
        required=True,
        help=(
            "Use model/API defaults to mirror the WMT-style pilot, or explicitly set "
            "temperature=0 with one text candidate."
        ),
    )
    prepare.set_defaults(handler=prepare_run)

    submit = subparsers.add_parser("submit", help="Upload and submit exactly one prepared shard.")
    submit.add_argument("--output-dir", required=True, type=Path)
    submit.add_argument("--shard-id", required=True)
    submit.add_argument("--api-key-file", required=True, type=Path)
    submit.add_argument("--confirm-run-config-sha256", required=True)
    submit.set_defaults(handler=submit_shard)

    status = subparsers.add_parser("status", help="Poll one or all submitted shards.")
    status.add_argument("--output-dir", required=True, type=Path)
    status.add_argument("--api-key-file", required=True, type=Path)
    status.add_argument("--shard-id")
    status.add_argument("--timeout-seconds", type=float, default=120.0)
    status.add_argument("--download-succeeded", action="store_true")
    status.add_argument("--fail-fast", action="store_true")
    status.set_defaults(handler=status_command)

    monitor = subparsers.add_parser("monitor", help="Poll and download submitted shards until terminal.")
    monitor.add_argument("--output-dir", required=True, type=Path)
    monitor.add_argument("--api-key-file", required=True, type=Path)
    monitor.add_argument("--timeout-seconds", type=float, default=120.0)
    monitor.add_argument("--poll-seconds", type=float, default=60.0)
    monitor.add_argument("--max-hours", type=float, default=30.0)
    monitor.set_defaults(handler=monitor_command)

    collect = subparsers.add_parser("collect", help="Strictly parse and aggregate all 16 shards.")
    collect.add_argument("--output-dir", required=True, type=Path)
    collect.set_defaults(handler=collect_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except LLMJudgeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
