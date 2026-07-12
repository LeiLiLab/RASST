#!/usr/bin/env python3
"""Score talk-level SimulEval outputs with sentence-aligned xCOMET.

The input manifest is either TSV or JSONL.  Each row describes one system and
must contain these fields::

    dataset method lang lm instances_log source_text reference audio_yaml latency_unit

Paths may be absolute or relative to the manifest.  ``source_text`` and
``reference`` are sentence-aligned text files; ``audio_yaml`` assigns every
sentence to a talk through its ``wav`` field.  Every talk in ``instances_log``
is mapped to those sentences and resegmented with an explicitly supplied
``mwerSegmenter`` executable before xCOMET sees it.

The model checkpoint is always local.  The model id and immutable revision are
still required so the resulting artifact records what that checkpoint claims
to contain.  No environment variable is read or modified for configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "rasst-xcomet-segments-v1"
MANIFEST_FIELDS = (
    "dataset",
    "method",
    "lang",
    "lm",
    "instances_log",
    "source_text",
    "reference",
    "audio_yaml",
    "latency_unit",
)
PATH_FIELDS = ("instances_log", "source_text", "reference", "audio_yaml")
TAG_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
COMET_GATHER_FILE_RE = re.compile(r"^(?:pred|batch_indices)_[0-9]+[.]pt$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class XCometScoringError(RuntimeError):
    """Raised when inputs cannot be scored without silently changing the data."""


class _RestrictedCometPredictionTorch:
    """Delegate torch operations while restricting unsafe COMET gather loads."""

    def __init__(self, torch_module: Any, output_dir: Any) -> None:
        self._torch = torch_module
        try:
            self._output_dir = Path(output_dir).resolve(strict=True)
        except (OSError, TypeError) as exc:
            raise XCometScoringError(
                f"COMET prediction output directory is invalid: {output_dir!r}"
            ) from exc
        if not self._output_dir.is_dir():
            raise XCometScoringError(
                f"COMET prediction output path is not a directory: {self._output_dir}"
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._torch, name)

    def load(self, path: Any, *args: Any, **kwargs: Any) -> Any:
        if "weights_only" in kwargs:
            raise XCometScoringError(
                "COMET prediction gather unexpectedly supplied weights_only; "
                "review the installed COMET implementation before scoring"
            )
        try:
            resolved = Path(os.fsdecode(os.fspath(path))).resolve(strict=True)
        except (OSError, TypeError) as exc:
            raise XCometScoringError(
                f"COMET prediction gather requested an invalid file: {path!r}"
            ) from exc
        if (
            resolved.parent != self._output_dir
            or not COMET_GATHER_FILE_RE.fullmatch(resolved.name)
            or not resolved.is_file()
        ):
            raise XCometScoringError(
                "Refusing non-COMET temporary prediction load outside the restricted "
                f"gather set: {resolved}"
            )
        return self._torch.load(resolved, *args, weights_only=False, **kwargs)


@dataclass(frozen=True)
class ManifestRow:
    dataset: str
    method: str
    lang: str
    lm: str
    instances_log: Path
    source_text: Path
    reference: Path
    audio_yaml: Path
    latency_unit: str
    manifest_row: int

    @property
    def system_key(self) -> Tuple[str, str, str, str]:
        return (self.dataset, self.method, self.lang, self.lm)


@dataclass
class PreparedSystem:
    manifest: ManifestRow
    talk_count: int
    segments: List[Dict[str, Any]]
    provenance_hashes: Dict[str, str]


@dataclass(frozen=True)
class RecoveryConfig:
    prediction_gather_dir: Path
    inference_runner_sha256: str
    gather_files: Tuple[Path, ...]


class FileHasher:
    """Hash immutable run inputs once even when manifest rows share them."""

    def __init__(self) -> None:
        self._cache: Dict[Path, str] = {}

    def sha256(self, path: Path) -> str:
        resolved = path.resolve()
        cached = self._cache.get(resolved)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self._cache[resolved] = value
        return value


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise XCometScoringError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise XCometScoringError(
                    f"Expected a JSON object at {path}:{line_number}"
                )
            yield line_number, row


def _read_manifest_records(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return list(_iter_jsonl(path))
    if suffix != ".tsv":
        raise XCometScoringError(
            f"Manifest must end in .tsv or .jsonl, got: {path}"
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise XCometScoringError(f"TSV manifest has no header: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise XCometScoringError(f"TSV manifest has duplicate columns: {path}")
        records: List[Tuple[int, Dict[str, Any]]] = []
        for line_number, row in enumerate(reader, start=2):
            if not any(str(value or "").strip() for value in row.values()):
                continue
            records.append((line_number, dict(row)))
        return records


def _resolve_manifest_path(raw: Any, manifest_path: Path, field: str, row_number: int) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise XCometScoringError(
            f"Empty manifest field {field!r} at {manifest_path}:{row_number}"
        )
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise XCometScoringError(
            f"Manifest field {field!r} is not a file at {manifest_path}:{row_number}: {path}"
        )
    return path


def load_manifest(path: Path) -> List[ManifestRow]:
    path = path.resolve()
    if not path.is_file():
        raise XCometScoringError(f"Manifest is not a file: {path}")
    records = _read_manifest_records(path)
    if not records:
        raise XCometScoringError(f"Manifest contains no system rows: {path}")

    rows: List[ManifestRow] = []
    seen: Dict[Tuple[str, str, str, str], int] = {}
    for row_number, record in records:
        missing = [name for name in MANIFEST_FIELDS if name not in record]
        if missing:
            raise XCometScoringError(
                f"Missing manifest fields at {path}:{row_number}: {', '.join(missing)}"
            )
        scalar_values: Dict[str, str] = {}
        for name in ("dataset", "method", "lang", "lm"):
            raw_value = record.get(name)
            value = "" if raw_value is None else str(raw_value).strip()
            if not value:
                raise XCometScoringError(
                    f"Empty manifest field {name!r} at {path}:{row_number}"
                )
            scalar_values[name] = value
        latency_unit = str(record.get("latency_unit") or "").strip().lower()
        if latency_unit not in {"char", "word"}:
            raise XCometScoringError(
                f"latency_unit must be 'char' or 'word' at {path}:{row_number}, "
                f"got {latency_unit!r}"
            )
        paths = {
            name: _resolve_manifest_path(record.get(name), path, name, row_number)
            for name in PATH_FIELDS
        }
        row = ManifestRow(
            dataset=scalar_values["dataset"],
            method=scalar_values["method"],
            lang=scalar_values["lang"],
            lm=scalar_values["lm"],
            instances_log=paths["instances_log"],
            source_text=paths["source_text"],
            reference=paths["reference"],
            audio_yaml=paths["audio_yaml"],
            latency_unit=latency_unit,
            manifest_row=row_number,
        )
        duplicate_at = seen.get(row.system_key)
        if duplicate_at is not None:
            raise XCometScoringError(
                f"Duplicate system key {row.system_key!r} at manifest rows "
                f"{duplicate_at} and {row_number}"
            )
        seen[row.system_key] = row_number
        rows.append(row)
    return rows


def validate_executable(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise XCometScoringError(f"mwerSegmenter is not a file: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise XCometScoringError(f"mwerSegmenter is not executable: {resolved}")
    return resolved


def checkpoint_hparams_path(checkpoint: Path) -> Path:
    if len(checkpoint.parents) < 2:
        raise XCometScoringError(
            f"Checkpoint path has no COMET snapshot parent directory: {checkpoint}"
        )
    return checkpoint.parents[1] / "hparams.yaml"


def validate_checkpoint(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise XCometScoringError(f"xCOMET checkpoint is not a file: {resolved}")
    if resolved.stat().st_size <= 0:
        raise XCometScoringError(f"xCOMET checkpoint is empty: {resolved}")
    hparams_path = checkpoint_hparams_path(resolved)
    if not hparams_path.is_file():
        raise XCometScoringError(
            f"COMET hparams.yaml is missing beside the checkpoint snapshot: {hparams_path}"
        )
    return resolved


def _normalise_sha256(value: str, label: str) -> str:
    cleaned = str(value or "").strip()
    if not SHA256_RE.fullmatch(cleaned):
        raise XCometScoringError(
            f"{label} must be exactly 64 hexadecimal characters"
        )
    return cleaned.lower()


def validate_prediction_gather_dir(
    path: Path,
    *,
    device_count: int,
) -> Tuple[Path, Tuple[Path, ...]]:
    if device_count <= 0:
        raise XCometScoringError(
            "Cannot validate a COMET prediction gather directory without devices"
        )
    raw_path = Path(path)
    if raw_path.is_symlink():
        raise XCometScoringError(
            f"COMET prediction gather directory must not be a symlink: {raw_path}"
        )
    try:
        resolved = raw_path.resolve(strict=True)
        directory_stat = os.lstat(resolved)
    except (OSError, TypeError) as exc:
        raise XCometScoringError(
            f"COMET prediction gather directory is invalid: {raw_path}"
        ) from exc
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise XCometScoringError(
            f"COMET prediction gather path is not a directory: {resolved}"
        )

    expected_names = {
        f"{prefix}_{rank}.pt"
        for rank in range(device_count)
        for prefix in ("pred", "batch_indices")
    }
    try:
        entries = list(resolved.iterdir())
    except OSError as exc:
        raise XCometScoringError(
            f"Cannot list COMET prediction gather directory: {resolved}"
        ) from exc
    actual_names = {entry.name for entry in entries}
    if len(entries) != len(expected_names) or actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise XCometScoringError(
            "COMET prediction gather directory must contain exactly one pred_<rank>.pt "
            "and batch_indices_<rank>.pt per device; "
            f"missing={missing!r} unexpected={unexpected!r}: {resolved}"
        )

    gather_files: List[Path] = []
    for name in sorted(expected_names):
        entry = resolved / name
        try:
            entry_stat = os.lstat(entry)
        except OSError as exc:
            raise XCometScoringError(
                f"Cannot stat COMET prediction gather file: {entry}"
            ) from exc
        if not stat.S_ISREG(entry_stat.st_mode):
            raise XCometScoringError(
                f"COMET prediction gather entry must be a regular file: {entry}"
            )
        gather_files.append(entry)
    return resolved, tuple(gather_files)


def resolve_recovery_config(
    *,
    prediction_gather_dir: Optional[Path],
    inference_runner_sha256: Optional[str],
    devices: Sequence[int],
) -> Optional[RecoveryConfig]:
    has_gather_dir = prediction_gather_dir is not None
    has_runner_hash = inference_runner_sha256 is not None
    if has_gather_dir != has_runner_hash:
        raise XCometScoringError(
            "--prediction-gather-dir and --inference-runner-sha256 must be supplied together"
        )
    if not has_gather_dir:
        return None
    assert prediction_gather_dir is not None
    assert inference_runner_sha256 is not None
    resolved_dir, gather_files = validate_prediction_gather_dir(
        prediction_gather_dir,
        device_count=len(devices),
    )
    return RecoveryConfig(
        prediction_gather_dir=resolved_dir,
        inference_runner_sha256=_normalise_sha256(
            inference_runner_sha256,
            "inference_runner_sha256",
        ),
        gather_files=gather_files,
    )


def normalise_output_tag_names(names: Sequence[str]) -> Tuple[str, ...]:
    unique: List[str] = []
    seen = set()
    for raw_name in names:
        name = str(raw_name or "").strip()
        if not TAG_NAME_RE.fullmatch(name):
            raise XCometScoringError(
                f"Invalid --strip-output-tag value {raw_name!r}; expected an XML-like tag name"
            )
        key = name.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(name)
    return tuple(unique)


def strip_explicit_output_tags(text: str, tag_names: Sequence[str]) -> str:
    """Remove only explicitly named open/close markers and preserve their content."""
    if not tag_names:
        return str(text or "").strip()
    names = normalise_output_tag_names(tag_names)
    alternation = "|".join(re.escape(name) for name in names)
    pattern = re.compile(rf"</?\s*(?:{alternation})\s*>", flags=re.IGNORECASE)
    return pattern.sub("", str(text or "")).strip()


def _read_aligned_lines(path: Path, label: str) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise XCometScoringError(f"{label} file is empty: {path}")
    empty_indices = [str(index) for index, line in enumerate(lines) if not line.strip()]
    if empty_indices:
        preview = ", ".join(empty_indices[:10])
        raise XCometScoringError(
            f"{label} contains empty sentence rows at zero-based indices {preview}: {path}"
        )
    return lines


def _talk_id_from_wav(raw: Any, *, context: str) -> str:
    value = str(raw or "").strip()
    if not value:
        raise XCometScoringError(f"Missing wav path for {context}")
    path = Path(value)
    if path.suffix.lower() != ".wav":
        raise XCometScoringError(f"Expected a .wav path for {context}, got {value!r}")
    talk_id = path.stem.strip()
    if not talk_id:
        raise XCometScoringError(f"Could not derive talk id for {context}: {value!r}")
    return talk_id


def _load_audio_talks(path: Path, expected_sentences: int) -> Tuple[List[str], Dict[str, List[int]]]:
    raw_text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise XCometScoringError(
                "PyYAML is required to read non-JSON audio_yaml files"
            ) from exc
        try:
            data = yaml.safe_load(raw_text)
        except Exception as exc:
            raise XCometScoringError(f"Failed to parse audio YAML {path}: {exc}") from exc
    except Exception as exc:
        raise XCometScoringError(f"Failed to parse audio YAML {path}: {exc}") from exc
    if not isinstance(data, list):
        raise XCometScoringError(f"audio_yaml must contain a list: {path}")
    if len(data) != expected_sentences:
        raise XCometScoringError(
            f"audio_yaml rows ({len(data)}) != sentence rows ({expected_sentences}): {path}"
        )

    talk_ids: List[str] = []
    indices_by_talk: Dict[str, List[int]] = defaultdict(list)
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise XCometScoringError(f"audio_yaml row {index} is not an object: {path}")
        talk_id = _talk_id_from_wav(item.get("wav"), context=f"{path} row {index}")
        talk_ids.append(talk_id)
        indices_by_talk[talk_id].append(index)
    return talk_ids, dict(indices_by_talk)


def _instance_talk_id(instance: Mapping[str, Any], path: Path, line_number: int) -> str:
    raw_source = instance.get("source")
    candidates: List[str] = []
    if isinstance(raw_source, list):
        candidates.extend(str(value) for value in raw_source)
    elif isinstance(raw_source, str):
        candidates.append(raw_source)
    source_path = instance.get("source_path")
    if isinstance(source_path, str):
        candidates.append(source_path)

    talk_ids = {
        Path(value.strip()).stem
        for value in candidates
        if value.strip() and Path(value.strip()).suffix.lower() == ".wav"
    }
    talk_ids.discard("")
    if len(talk_ids) != 1:
        raise XCometScoringError(
            f"Expected exactly one .wav talk in {path}:{line_number}, found {sorted(talk_ids)!r}"
        )
    return next(iter(talk_ids))


def _load_instances(path: Path) -> List[Tuple[int, Dict[str, Any], str]]:
    instances: List[Tuple[int, Dict[str, Any], str]] = []
    seen_talks: Dict[str, int] = {}
    for line_number, instance in _iter_jsonl(path):
        if "prediction" not in instance:
            raise XCometScoringError(f"Missing prediction field at {path}:{line_number}")
        talk_id = _instance_talk_id(instance, path, line_number)
        previous_line = seen_talks.get(talk_id)
        if previous_line is not None:
            raise XCometScoringError(
                f"Duplicate talk {talk_id!r} in talk-level instances log {path} at lines "
                f"{previous_line} and {line_number}"
            )
        seen_talks[talk_id] = line_number
        instances.append((line_number, instance, talk_id))
    if not instances:
        raise XCometScoringError(f"instances_log contains no talks: {path}")
    return instances


def segment_prediction_by_references(
    *,
    executable: Path,
    prediction: str,
    reference_sentences: Sequence[str],
    latency_unit: str,
    timeout_seconds: float,
) -> List[str]:
    """Invoke mwerSegmenter and return one hypothesis for each reference sentence."""
    if latency_unit not in {"char", "word"}:
        raise XCometScoringError(f"Unsupported latency unit: {latency_unit!r}")
    if not reference_sentences:
        raise XCometScoringError("Cannot resegment a talk with zero reference sentences")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise XCometScoringError("mwerSegmenter timeout must be a finite positive number")

    character_level = latency_unit == "char"
    hypothesis_text = str(prediction or "")
    references = [str(value or "") for value in reference_sentences]
    if character_level:
        hypothesis_text = " ".join(hypothesis_text)
        references = [" ".join(value) for value in references]

    with tempfile.TemporaryDirectory(prefix="rasst_xcomet_mwer_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        hypothesis_path = temp_dir / "hypothesis.txt"
        reference_path = temp_dir / "references.txt"
        segments_path = temp_dir / "__segments"
        hypothesis_path.write_text(hypothesis_text, encoding="utf-8")
        reference_path.write_text(
            "".join(reference + "\n" for reference in references),
            encoding="utf-8",
        )
        command = [
            str(executable),
            "-mref",
            str(reference_path),
            "-hypfile",
            str(hypothesis_path),
            "-usecase",
            "1",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=temp_dir,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise XCometScoringError(
                f"mwerSegmenter timed out after {timeout_seconds:g}s"
            ) from exc
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise XCometScoringError(
                f"mwerSegmenter exited with code {completed.returncode}: {stderr[-2000:]}"
            )
        if not segments_path.is_file():
            raise XCometScoringError(
                f"mwerSegmenter succeeded but did not create {segments_path.name}"
            )
        segments = segments_path.read_text(encoding="utf-8").splitlines()

    if character_level:
        segments = [re.sub(r"(.)\s", r"\1", segment).strip() for segment in segments]
    else:
        segments = [segment.strip() for segment in segments]
    if len(segments) != len(reference_sentences):
        raise XCometScoringError(
            f"mwerSegmenter returned {len(segments)} segments for "
            f"{len(reference_sentences)} references"
        )
    return segments


def prepare_system(
    manifest: ManifestRow,
    *,
    manifest_sha256: str,
    runner_sha256: str,
    checkpoint_sha256: str,
    checkpoint_hparams_sha256: str,
    segmenter: Path,
    segmenter_sha256: str,
    output_tags: Sequence[str],
    timeout_seconds: float,
    file_hasher: FileHasher,
) -> PreparedSystem:
    sources = _read_aligned_lines(manifest.source_text, "source_text")
    references = _read_aligned_lines(manifest.reference, "reference")
    if len(sources) != len(references):
        raise XCometScoringError(
            f"source/reference row mismatch for {manifest.system_key!r}: "
            f"{len(sources)} != {len(references)}"
        )
    _, sentence_indices_by_talk = _load_audio_talks(
        manifest.audio_yaml,
        expected_sentences=len(sources),
    )
    instances = _load_instances(manifest.instances_log)

    provenance_hashes = {
        "manifest_sha256": manifest_sha256,
        "runner_sha256": runner_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_hparams_sha256": checkpoint_hparams_sha256,
        "mwersegmenter_sha256": segmenter_sha256,
        "instances_log_sha256": file_hasher.sha256(manifest.instances_log),
        "source_text_sha256": file_hasher.sha256(manifest.source_text),
        "reference_sha256": file_hasher.sha256(manifest.reference),
        "audio_yaml_sha256": file_hasher.sha256(manifest.audio_yaml),
    }
    segments: List[Dict[str, Any]] = []
    for instance_order, (line_number, instance, talk_id) in enumerate(instances):
        sentence_indices = sentence_indices_by_talk.get(talk_id)
        if sentence_indices is None:
            raise XCometScoringError(
                f"Talk {talk_id!r} from {manifest.instances_log}:{line_number} "
                f"does not occur in {manifest.audio_yaml}"
            )
        raw_prediction = str(instance.get("prediction") or "")
        clean_prediction = strip_explicit_output_tags(raw_prediction, output_tags)
        talk_references = [references[index] for index in sentence_indices]
        talk_hypotheses = segment_prediction_by_references(
            executable=segmenter,
            prediction=clean_prediction,
            reference_sentences=talk_references,
            latency_unit=manifest.latency_unit,
            timeout_seconds=timeout_seconds,
        )
        raw_instance_index = instance.get("index", instance_order)
        for talk_sentence_index, (sentence_index, hypothesis) in enumerate(
            zip(sentence_indices, talk_hypotheses)
        ):
            source = sources[sentence_index]
            reference = references[sentence_index]
            segments.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset": manifest.dataset,
                    "method": manifest.method,
                    "lang": manifest.lang,
                    "lm": manifest.lm,
                    "manifest_row": manifest.manifest_row,
                    "instances_line": line_number,
                    "instance_index": raw_instance_index,
                    "talk_id": talk_id,
                    "talk_sentence_index": talk_sentence_index,
                    "sentence_index": sentence_index,
                    "source": source,
                    "hypothesis": hypothesis,
                    "reference": reference,
                    "talk_prediction_sha256": hashlib.sha256(
                        raw_prediction.encode("utf-8")
                    ).hexdigest(),
                    "scoring_input_sha256": _canonical_json_sha256(
                        {"src": source, "mt": hypothesis, "ref": reference}
                    ),
                }
            )
    if not segments:
        raise XCometScoringError(f"No sentence segments produced for {manifest.system_key!r}")
    return PreparedSystem(
        manifest=manifest,
        talk_count=len(instances),
        segments=segments,
        provenance_hashes=provenance_hashes,
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        return _json_safe(item_method())
    raise XCometScoringError(
        f"xCOMET returned a non-JSON-serializable value of type {type(value).__name__}"
    )


def _prediction_field(output: Any, name: str) -> Any:
    if hasattr(output, name):
        return getattr(output, name)
    if isinstance(output, Mapping) and name in output:
        return output[name]
    raise XCometScoringError(f"xCOMET output is missing {name!r}")


def extract_xcomet_output(output: Any, expected_segments: int) -> Tuple[List[float], List[List[Dict[str, Any]]]]:
    raw_scores = _prediction_field(output, "scores")
    try:
        scores = [float(score) for score in raw_scores]
    except (TypeError, ValueError) as exc:
        raise XCometScoringError("xCOMET scores are not a numeric sequence") from exc
    if len(scores) != expected_segments:
        raise XCometScoringError(
            f"xCOMET returned {len(scores)} scores for {expected_segments} segments"
        )
    if not all(math.isfinite(score) for score in scores):
        raise XCometScoringError("xCOMET returned a non-finite segment score")

    metadata = _prediction_field(output, "metadata")
    raw_error_spans = _prediction_field(metadata, "error_spans")
    safe_error_spans = _json_safe(raw_error_spans)
    if not isinstance(safe_error_spans, list) or len(safe_error_spans) != expected_segments:
        actual = len(safe_error_spans) if isinstance(safe_error_spans, list) else "non-list"
        raise XCometScoringError(
            f"xCOMET returned {actual} error-span rows for {expected_segments} segments"
        )
    error_spans: List[List[Dict[str, Any]]] = []
    for segment_index, spans in enumerate(safe_error_spans):
        if not isinstance(spans, list) or not all(isinstance(span, dict) for span in spans):
            raise XCometScoringError(
                f"xCOMET error spans for segment {segment_index} are not a list of objects"
            )
        error_spans.append(spans)
    return scores, error_spans


def load_xcomet(checkpoint: Path) -> Any:
    try:
        from comet import load_from_checkpoint
    except ImportError as exc:
        raise XCometScoringError(
            "unbabel-comet is required to run xCOMET scoring"
        ) from exc

    return load_from_checkpoint(str(checkpoint), local_files_only=True)


@contextmanager
def restricted_comet_prediction_gather_loads() -> Iterator[None]:
    """Permit unsafe pickle only for COMET's private prediction gather files."""

    try:
        from comet.models import predict_writer
    except ImportError as exc:
        raise XCometScoringError(
            "unbabel-comet is required to run xCOMET scoring"
        ) from exc

    writer_class = predict_writer.CustomWriter
    original_gather = writer_class.gather_all_predictions

    # note (luojiaxuan): COMET 2.2.7 pickles its Prediction OrderedDict subclass
    # during DDP gather. PyTorch 2.6+ cannot weights-only load that subclass even
    # when allowlisted, so limit weights_only=False to COMET's private temp files.
    def restricted_gather(writer: Any, *args: Any, **kwargs: Any) -> Any:
        original_torch = predict_writer.torch
        predict_writer.torch = _RestrictedCometPredictionTorch(
            original_torch,
            writer.output_dir,
        )
        try:
            return original_gather(writer, *args, **kwargs)
        finally:
            predict_writer.torch = original_torch

    writer_class.gather_all_predictions = restricted_gather
    try:
        yield
    finally:
        writer_class.gather_all_predictions = original_gather


def _assign_xcomet_output(
    systems: Sequence[PreparedSystem],
    output: Any,
) -> None:
    flat_segments = [segment for system in systems for segment in system.segments]
    scores, error_spans = extract_xcomet_output(output, len(flat_segments))
    for segment, score, spans in zip(flat_segments, scores, error_spans):
        segment["xcomet_score"] = score
        segment["error_spans"] = spans


def _hash_gather_files(paths: Sequence[Path]) -> Dict[str, str]:
    hasher = FileHasher()
    return {
        f"{path.stem}_sha256": hasher.sha256(path)
        for path in paths
    }


def recover_prepared_systems(
    systems: Sequence[PreparedSystem],
    *,
    recovery: RecoveryConfig,
    devices: Sequence[int],
    recovery_runner_sha256: str,
) -> None:
    if not systems:
        raise XCometScoringError("No prepared systems to recover")
    if not devices:
        raise XCometScoringError("At least one GPU device index is required")

    gather_dir, gather_files = validate_prediction_gather_dir(
        recovery.prediction_gather_dir,
        device_count=len(devices),
    )
    if gather_dir != recovery.prediction_gather_dir or gather_files != recovery.gather_files:
        raise XCometScoringError(
            "COMET prediction gather directory entries changed after initial validation"
        )
    before_hashes = _hash_gather_files(gather_files)
    total_segments = sum(len(system.segments) for system in systems)
    print(
        f"[RECOVER] gathering predictions from {gather_dir} "
        f"files={len(gather_files)} expected_segments={total_segments}",
        flush=True,
    )

    try:
        from comet.models import predict_writer

        writer = predict_writer.CustomWriter()
        writer.output_dir = str(gather_dir)
        with restricted_comet_prediction_gather_loads():
            output = writer.gather_all_predictions()
    except XCometScoringError:
        raise
    except Exception as exc:
        raise XCometScoringError(
            f"Failed to recover COMET predictions from {gather_dir}: {exc}"
        ) from exc

    after_dir, after_files = validate_prediction_gather_dir(
        gather_dir,
        device_count=len(devices),
    )
    after_hashes = _hash_gather_files(after_files)
    if after_dir != gather_dir or after_files != gather_files or after_hashes != before_hashes:
        raise XCometScoringError(
            "COMET prediction gather files changed while recovery was reading them"
        )

    _assign_xcomet_output(systems, output)
    recovery_hashes = {
        "recovery_runner_sha256": _normalise_sha256(
            recovery_runner_sha256,
            "recovery_runner_sha256",
        ),
        **after_hashes,
    }
    for system in systems:
        system.provenance_hashes.update(recovery_hashes)
    print(
        f"[RECOVER] restored scores_and_metadata={total_segments}; input gather retained",
        flush=True,
    )


def score_prepared_systems(
    systems: Sequence[PreparedSystem],
    *,
    model: Any,
    devices: Sequence[int],
    batch_size: int,
    num_workers: int,
    progress_bar: bool,
) -> None:
    if not systems:
        raise XCometScoringError("No prepared systems to score")
    if not devices:
        raise XCometScoringError("At least one GPU device index is required")
    if len(devices) != len(set(devices)) or any(device < 0 for device in devices):
        raise XCometScoringError(f"GPU devices must be distinct non-negative indices: {devices!r}")
    if batch_size <= 0:
        raise XCometScoringError("batch_size must be positive")
    if num_workers < 0:
        raise XCometScoringError("num_workers must be non-negative")

    flat_segments = [segment for system in systems for segment in system.segments]
    model_inputs = [
        {
            "src": segment["source"],
            "mt": segment["hypothesis"],
            "ref": segment["reference"],
        }
        for segment in flat_segments
    ]
    with restricted_comet_prediction_gather_loads():
        output = model.predict(
            model_inputs,
            batch_size=batch_size,
            gpus=len(devices),
            devices=list(devices),
            accelerator="gpu",
            num_workers=num_workers,
            progress_bar=progress_bar,
            length_batching=True,
        )
    _assign_xcomet_output(systems, output)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise XCometScoringError("Cannot compute a mean over zero scores")
    return float(statistics.fmean(values))


def build_summary_rows(
    systems: Sequence[PreparedSystem],
    *,
    model_id: str,
    model_revision: str,
    scoring_config_sha256: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for system in systems:
        scores = [float(segment["xcomet_score"]) for segment in system.segments]
        scores_by_talk: Dict[str, List[float]] = defaultdict(list)
        for segment in system.segments:
            scores_by_talk[str(segment["talk_id"])].append(float(segment["xcomet_score"]))
        mean_score = _mean(scores)
        talk_macro = _mean([_mean(values) for values in scores_by_talk.values()])
        rows.append(
            {
                "dataset": system.manifest.dataset,
                "method": system.manifest.method,
                "lang": system.manifest.lang,
                "lm": system.manifest.lm,
                "talks": system.talk_count,
                "segments": len(system.segments),
                "xcomet_mean": mean_score,
                "xcomet_mean_x100": mean_score * 100.0,
                "xcomet_talk_macro_mean": talk_macro,
                "model_id": model_id,
                "model_revision": model_revision,
                "scoring_config_sha256": scoring_config_sha256,
                "instances_log_sha256": system.provenance_hashes["instances_log_sha256"],
            }
        )
    return rows


def _pair_segment_key(segment: Mapping[str, Any]) -> Tuple[str, int]:
    return (str(segment["talk_id"]), int(segment["talk_sentence_index"]))


def build_paired_rows(
    systems: Sequence[PreparedSystem],
    *,
    rasst_method: str,
    baseline_method: str,
) -> List[Dict[str, Any]]:
    rasst_key = rasst_method.casefold()
    baseline_key = baseline_method.casefold()
    if rasst_key == baseline_key:
        raise XCometScoringError("RASST and baseline method labels must differ")

    grouped: Dict[Tuple[str, str, str], Dict[str, PreparedSystem]] = defaultdict(dict)
    for system in systems:
        method_key = system.manifest.method.casefold()
        if method_key not in {rasst_key, baseline_key}:
            continue
        group_key = (system.manifest.dataset, system.manifest.lang, system.manifest.lm)
        role = "rasst" if method_key == rasst_key else "baseline"
        if role in grouped[group_key]:
            raise XCometScoringError(
                f"Multiple {role} systems found for paired group {group_key!r}"
            )
        grouped[group_key][role] = system
    if not grouped:
        raise XCometScoringError(
            f"Manifest contains no systems named {rasst_method!r} or {baseline_method!r}"
        )

    rows: List[Dict[str, Any]] = []
    for group_key in sorted(grouped):
        pair = grouped[group_key]
        if set(pair) != {"rasst", "baseline"}:
            raise XCometScoringError(
                f"Incomplete RASST/InfiniSST pair for {group_key!r}: found {sorted(pair)}"
            )
        rasst_system = pair["rasst"]
        baseline_system = pair["baseline"]
        rasst_segments = {_pair_segment_key(segment): segment for segment in rasst_system.segments}
        baseline_segments = {
            _pair_segment_key(segment): segment for segment in baseline_system.segments
        }
        if len(rasst_segments) != len(rasst_system.segments):
            raise XCometScoringError(f"Duplicate RASST segment keys for {group_key!r}")
        if len(baseline_segments) != len(baseline_system.segments):
            raise XCometScoringError(f"Duplicate InfiniSST segment keys for {group_key!r}")
        if set(rasst_segments) != set(baseline_segments):
            missing_rasst = sorted(set(baseline_segments) - set(rasst_segments))[:10]
            missing_baseline = sorted(set(rasst_segments) - set(baseline_segments))[:10]
            raise XCometScoringError(
                f"Unpaired segment keys for {group_key!r}; missing_rasst={missing_rasst!r} "
                f"missing_infinisst={missing_baseline!r}"
            )

        rasst_scores: List[float] = []
        baseline_scores: List[float] = []
        deltas: List[float] = []
        wins = ties = losses = 0
        for key in sorted(rasst_segments):
            rasst_segment = rasst_segments[key]
            baseline_segment = baseline_segments[key]
            if (
                rasst_segment["source"] != baseline_segment["source"]
                or rasst_segment["reference"] != baseline_segment["reference"]
            ):
                raise XCometScoringError(
                    f"Source/reference mismatch in paired segment {group_key!r}/{key!r}"
                )
            rasst_score = float(rasst_segment["xcomet_score"])
            baseline_score = float(baseline_segment["xcomet_score"])
            delta = rasst_score - baseline_score
            rasst_scores.append(rasst_score)
            baseline_scores.append(baseline_score)
            deltas.append(delta)
            if abs(delta) <= 1e-12:
                ties += 1
            elif delta > 0:
                wins += 1
            else:
                losses += 1

        rows.append(
            {
                "dataset": group_key[0],
                "lang": group_key[1],
                "lm": group_key[2],
                "rasst_method": rasst_system.manifest.method,
                "infinisst_method": baseline_system.manifest.method,
                "paired_talks": rasst_system.talk_count,
                "paired_segments": len(deltas),
                "rasst_xcomet_mean": _mean(rasst_scores),
                "infinisst_xcomet_mean": _mean(baseline_scores),
                "delta_rasst_minus_infinisst": _mean(deltas),
                "paired_delta_stddev": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "rasst_wins": wins,
                "ties": ties,
                "infinisst_wins": losses,
            }
        )
    return rows


def _format_tsv_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.10f}"
    return str(value)


def _atomic_write(path: Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _format_tsv_value(row[name]) for name in fieldnames})
    _atomic_write(path, output.getvalue())


def write_segment_jsonl(
    path: Path,
    systems: Sequence[PreparedSystem],
    *,
    model_id: str,
    model_revision: str,
    checkpoint: Path,
    segmenter: Path,
    devices: Sequence[int],
    output_tags: Sequence[str],
    scoring_config_sha256: str,
) -> None:
    lines: List[str] = []
    for system in systems:
        for segment in system.segments:
            record = dict(segment)
            record["model"] = {
                "id": model_id,
                "revision": model_revision,
                "local_checkpoint": str(checkpoint),
                "gpu_devices": list(devices),
            }
            record["resegmentation"] = {
                "mwersegmenter": str(segmenter),
                "latency_unit": system.manifest.latency_unit,
                "stripped_output_tags": list(output_tags),
            }
            record["provenance_hashes"] = {
                **system.provenance_hashes,
                "scoring_config_sha256": scoring_config_sha256,
            }
            lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    _atomic_write(path, "\n".join(lines) + "\n")


def _non_empty(value: str, label: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise XCometScoringError(f"{label} must be non-empty")
    return cleaned


def validate_output_paths(paths: Sequence[Path], protected_inputs: Sequence[Path]) -> None:
    resolved_outputs = [path.resolve() for path in paths]
    if len(resolved_outputs) != len(set(resolved_outputs)):
        raise XCometScoringError(f"Output paths must be distinct: {resolved_outputs!r}")
    protected = {path.resolve() for path in protected_inputs}
    collisions = sorted(str(path) for path in resolved_outputs if path in protected)
    if collisions:
        raise XCometScoringError(
            f"Refusing to overwrite an input with an output: {', '.join(collisions)}"
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resegment every talk and score sentence-aligned RASST systems with local xCOMET."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="System manifest (.tsv or .jsonl).")
    parser.add_argument("--mwer-segmenter", type=Path, required=True, help="Executable mwerSegmenter path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local xCOMET .ckpt file.")
    parser.add_argument("--model-id", required=True, help="Hugging Face model id recorded as provenance.")
    parser.add_argument("--model-revision", required=True, help="Immutable model revision/commit.")
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        required=True,
        help="Physical GPU indices passed directly to COMET predict, e.g. --devices 2 3.",
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument(
        "--prediction-gather-dir",
        type=Path,
        help=(
            "Recover from a complete COMET DDP prediction directory instead of "
            "loading the model or running inference. Must be paired with "
            "--inference-runner-sha256."
        ),
    )
    parser.add_argument(
        "--inference-runner-sha256",
        help=(
            "SHA-256 of the scoring runner that produced --prediction-gather-dir; "
            "required only in recovery mode."
        ),
    )
    parser.add_argument(
        "--segmenter-timeout-seconds",
        type=float,
        default=300.0,
        help="Per-talk mwerSegmenter timeout.",
    )
    parser.add_argument(
        "--strip-output-tag",
        action="append",
        default=[],
        metavar="TAG",
        help="Repeat for each exact output marker to remove, e.g. term and t.",
    )
    parser.add_argument("--rasst-method", default="RASST")
    parser.add_argument("--infinisst-method", default="InfiniSST")
    parser.add_argument("--summary-tsv", type=Path, required=True)
    parser.add_argument("--paired-tsv", type=Path, required=True)
    parser.add_argument("--segments-jsonl", type=Path, required=True)
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    rows = load_manifest(manifest_path)
    checkpoint = validate_checkpoint(args.checkpoint)
    segmenter = validate_executable(args.mwer_segmenter)
    model_id = _non_empty(args.model_id, "model_id")
    model_revision = _non_empty(args.model_revision, "model_revision")
    output_tags = normalise_output_tag_names(args.strip_output_tag)
    devices = list(args.devices)
    if len(devices) != len(set(devices)) or any(device < 0 for device in devices):
        raise XCometScoringError(f"GPU devices must be distinct non-negative indices: {devices!r}")
    if args.batch_size <= 0:
        raise XCometScoringError("batch_size must be positive")
    if args.num_workers < 0:
        raise XCometScoringError("num_workers must be non-negative")
    recovery = resolve_recovery_config(
        prediction_gather_dir=args.prediction_gather_dir,
        inference_runner_sha256=args.inference_runner_sha256,
        devices=devices,
    )
    validate_output_paths(
        [args.summary_tsv, args.paired_tsv, args.segments_jsonl],
        [
            manifest_path,
            checkpoint,
            checkpoint_hparams_path(checkpoint),
            segmenter,
            *(recovery.gather_files if recovery is not None else ()),
            *(
                path
                for row in rows
                for path in (row.instances_log, row.source_text, row.reference, row.audio_yaml)
            ),
        ],
    )

    file_hasher = FileHasher()
    manifest_sha256 = file_hasher.sha256(manifest_path)
    runner_path = Path(__file__).resolve()
    runner_sha256 = file_hasher.sha256(runner_path)
    preparation_runner_sha256 = (
        recovery.inference_runner_sha256 if recovery is not None else runner_sha256
    )
    checkpoint_sha256 = file_hasher.sha256(checkpoint)
    checkpoint_hparams = checkpoint_hparams_path(checkpoint)
    checkpoint_hparams_sha256 = file_hasher.sha256(checkpoint_hparams)
    segmenter_sha256 = file_hasher.sha256(segmenter)
    scoring_config = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "model_revision": model_revision,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_hparams_sha256": checkpoint_hparams_sha256,
        "mwersegmenter_sha256": segmenter_sha256,
        "gpu_devices": devices,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "output_tags": list(output_tags),
    }
    scoring_config_sha256 = _canonical_json_sha256(scoring_config)

    systems: List[PreparedSystem] = []
    for row in rows:
        print(f"[PREPARE] {row.dataset}/{row.method}/{row.lang}/lm={row.lm}", flush=True)
        systems.append(
            prepare_system(
                row,
                manifest_sha256=manifest_sha256,
                runner_sha256=preparation_runner_sha256,
                checkpoint_sha256=checkpoint_sha256,
                checkpoint_hparams_sha256=checkpoint_hparams_sha256,
                segmenter=segmenter,
                segmenter_sha256=segmenter_sha256,
                output_tags=output_tags,
                timeout_seconds=args.segmenter_timeout_seconds,
                file_hasher=file_hasher,
            )
        )

    total_segments = sum(len(system.segments) for system in systems)
    if recovery is None:
        print(
            f"[MODEL] loading {model_id}@{model_revision} once from {checkpoint}",
            flush=True,
        )
        model = load_xcomet(checkpoint)
        print(
            f"[SCORE] systems={len(systems)} segments={total_segments} devices={devices}",
            flush=True,
        )
        score_prepared_systems(
            systems,
            model=model,
            devices=devices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            progress_bar=not args.no_progress_bar,
        )
    else:
        recover_prepared_systems(
            systems,
            recovery=recovery,
            devices=devices,
            recovery_runner_sha256=runner_sha256,
        )

    summary_rows = build_summary_rows(
        systems,
        model_id=model_id,
        model_revision=model_revision,
        scoring_config_sha256=scoring_config_sha256,
    )
    paired_rows = build_paired_rows(
        systems,
        rasst_method=args.rasst_method,
        baseline_method=args.infinisst_method,
    )
    summary_fields = (
        "dataset",
        "method",
        "lang",
        "lm",
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
    paired_fields = (
        "dataset",
        "lang",
        "lm",
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
    write_tsv(args.summary_tsv, summary_rows, summary_fields)
    write_tsv(args.paired_tsv, paired_rows, paired_fields)
    write_segment_jsonl(
        args.segments_jsonl,
        systems,
        model_id=model_id,
        model_revision=model_revision,
        checkpoint=checkpoint,
        segmenter=segmenter,
        devices=devices,
        output_tags=output_tags,
        scoring_config_sha256=scoring_config_sha256,
    )
    print(f"[DONE] summary={args.summary_tsv}", flush=True)
    print(f"[DONE] paired={args.paired_tsv}", flush=True)
    print(f"[DONE] segments={args.segments_jsonl}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except XCometScoringError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
