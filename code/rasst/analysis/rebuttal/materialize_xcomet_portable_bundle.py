#!/usr/bin/env python3
"""Materialize an xCOMET input manifest as a portable, verified bundle.

The input is a TSV or JSONL xCOMET manifest.  Every file referenced by the
four path fields is copied below ``files/<sha256>/<basename>`` and the portable
manifest uses paths relative to its own location.  ``provenance.jsonl`` maps
each resolved source file to its content address and manifest references.

The output directory must not already exist.  Work is staged in a sibling
directory and renamed into place only after every copied file has passed a
size and SHA-256 check.  Configuration is provided only through arguments;
environment variables are not read or modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "rasst-xcomet-portable-bundle-v1"
PATH_FIELDS = ("instances_log", "source_text", "reference", "audio_yaml")
RESERVED_OUTPUT_NAMES = {"bundle.json", "provenance.jsonl"}
COPY_CHUNK_BYTES = 8 * 1024 * 1024


class PortableBundleError(RuntimeError):
    """Raised when a bundle cannot be produced without changing its meaning."""


@dataclass(frozen=True)
class ManifestDocument:
    format: str
    fieldnames: Tuple[str, ...]
    rows: Tuple[Mapping[str, Any], ...]
    source_lines: Tuple[int, ...]


@dataclass(frozen=True)
class FileIdentity:
    path: Path
    sha256: str
    bytes: int
    bundle_path: str


@dataclass
class ProvenanceRecord:
    identity: FileIdentity
    references: List[Dict[str, Any]] = field(default_factory=list)


def _strict_json_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PortableBundleError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_tsv_manifest(path: Path) -> ManifestDocument:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise PortableBundleError(f"TSV manifest is empty: {path}") from exc

        if not header or any(not name for name in header):
            raise PortableBundleError(f"TSV manifest has an empty column name: {path}")
        if len(header) != len(set(header)):
            raise PortableBundleError(f"TSV manifest has duplicate columns: {path}")

        rows: List[Mapping[str, Any]] = []
        source_lines: List[int] = []
        for line_number, values in enumerate(reader, start=2):
            if not values or all(value == "" for value in values):
                continue
            if len(values) != len(header):
                raise PortableBundleError(
                    f"TSV row has {len(values)} columns, expected {len(header)} "
                    f"at {path}:{line_number}"
                )
            rows.append(dict(zip(header, values)))
            source_lines.append(line_number)

    return ManifestDocument("tsv", tuple(header), tuple(rows), tuple(source_lines))


def _read_jsonl_manifest(path: Path) -> ManifestDocument:
    rows: List[Mapping[str, Any]] = []
    source_lines: List[int] = []
    fieldnames: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_strict_json_object)
            except (json.JSONDecodeError, PortableBundleError) as exc:
                raise PortableBundleError(
                    f"Invalid JSON object at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise PortableBundleError(
                    f"Expected a JSON object at {path}:{line_number}"
                )
            rows.append(row)
            source_lines.append(line_number)
            fieldnames.update(row)

    return ManifestDocument(
        "jsonl",
        tuple(sorted(fieldnames)),
        tuple(rows),
        tuple(source_lines),
    )


def read_manifest(path: Path) -> ManifestDocument:
    resolved = path.resolve()
    if not resolved.is_file():
        raise PortableBundleError(f"Input manifest is not a file: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix == ".tsv":
        document = _read_tsv_manifest(resolved)
    elif suffix == ".jsonl":
        document = _read_jsonl_manifest(resolved)
    else:
        raise PortableBundleError(
            f"Input manifest must end in .tsv or .jsonl, got: {resolved}"
        )
    if not document.rows:
        raise PortableBundleError(f"Input manifest contains no rows: {resolved}")

    missing_columns = [name for name in PATH_FIELDS if name not in document.fieldnames]
    if missing_columns:
        raise PortableBundleError(
            f"Input manifest is missing path fields: {', '.join(missing_columns)}"
        )
    for row_number, (row, source_line) in enumerate(
        zip(document.rows, document.source_lines), start=1
    ):
        missing = [name for name in PATH_FIELDS if name not in row]
        if missing:
            raise PortableBundleError(
                f"Manifest row {row_number} (source line {source_line}) is missing "
                f"path fields: {', '.join(missing)}"
            )
    return document


def _hash_file(path: Path) -> Tuple[str, int]:
    try:
        before = path.stat()
    except OSError as exc:
        raise PortableBundleError(f"Cannot stat referenced file {path}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise PortableBundleError(f"Referenced path is not a regular file: {path}")

    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(COPY_CHUNK_BYTES), b""):
                digest.update(chunk)
                byte_count += len(chunk)
        after = path.stat()
    except OSError as exc:
        raise PortableBundleError(f"Cannot hash referenced file {path}: {exc}") from exc

    stable_fields = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if stable_fields[:4] != stable_fields[4:] or byte_count != after.st_size:
        raise PortableBundleError(f"Referenced file changed while hashing: {path}")
    return digest.hexdigest(), byte_count


def _resolve_reference(raw: Any, manifest_path: Path, field_name: str, line: int) -> Path:
    if not isinstance(raw, str):
        raise PortableBundleError(
            f"Path field {field_name!r} must be a string at {manifest_path}:{line}"
        )
    value = raw.strip()
    if not value:
        raise PortableBundleError(
            f"Path field {field_name!r} is empty at {manifest_path}:{line}"
        )
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise PortableBundleError(
            f"Cannot resolve path field {field_name!r} at {manifest_path}:{line}: "
            f"{candidate} ({exc})"
        ) from exc
    if not resolved.is_file():
        raise PortableBundleError(
            f"Path field {field_name!r} is not a file at {manifest_path}:{line}: "
            f"{resolved}"
        )
    return resolved


def _validate_output_manifest_name(raw: Optional[str], input_format: str) -> str:
    name = raw or f"manifest.{input_format}"
    path = Path(name)
    if path.is_absolute() or len(path.parts) != 1 or name in {"", ".", ".."}:
        raise PortableBundleError(
            "--output-manifest must be a single filename at the bundle root"
        )
    if name in RESERVED_OUTPUT_NAMES or name == "files":
        raise PortableBundleError(f"Reserved output manifest name: {name}")
    expected_suffix = f".{input_format}"
    if path.suffix.lower() != expected_suffix:
        raise PortableBundleError(
            f"Output manifest must keep input format {expected_suffix}: {name}"
        )
    return name


def _plan_bundle(
    manifest_path: Path,
    document: ManifestDocument,
) -> Tuple[
    List[MutableMapping[str, Any]],
    Dict[Path, ProvenanceRecord],
    Dict[str, FileIdentity],
]:
    rewritten_rows: List[MutableMapping[str, Any]] = []
    provenance: Dict[Path, ProvenanceRecord] = {}
    payloads: Dict[str, FileIdentity] = {}

    for row_number, (source_row, source_line) in enumerate(
        zip(document.rows, document.source_lines), start=1
    ):
        rewritten = dict(source_row)
        for field_name in PATH_FIELDS:
            raw_value = source_row[field_name]
            source_path = _resolve_reference(
                raw_value, manifest_path, field_name, source_line
            )
            record = provenance.get(source_path)
            if record is None:
                sha256, byte_count = _hash_file(source_path)
                bundle_path = f"files/{sha256}/{source_path.name}"
                identity = FileIdentity(
                    source_path,
                    sha256,
                    byte_count,
                    bundle_path,
                )
                record = ProvenanceRecord(identity)
                provenance[source_path] = record
                payloads.setdefault(bundle_path, identity)
            record.references.append(
                {
                    "field": field_name,
                    "manifest_row": row_number,
                    "source_line": source_line,
                    "manifest_value": raw_value,
                }
            )
            rewritten[field_name] = record.identity.bundle_path
        rewritten_rows.append(rewritten)
    return rewritten_rows, provenance, payloads


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise PortableBundleError(f"Cannot write bundle file {path}: {exc}") from exc


def _render_manifest(
    document: ManifestDocument,
    rewritten_rows: Sequence[Mapping[str, Any]],
) -> bytes:
    output = io.StringIO(newline="")
    if document.format == "tsv":
        writer = csv.DictWriter(
            output,
            fieldnames=document.fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rewritten_rows)
    else:
        for row in rewritten_rows:
            output.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            output.write("\n")
    return output.getvalue().encode("utf-8")


def _render_provenance(provenance: Mapping[Path, ProvenanceRecord]) -> bytes:
    lines: List[str] = []
    for source_path in sorted(provenance, key=lambda value: str(value)):
        record = provenance[source_path]
        value = {
            "bundle_path": record.identity.bundle_path,
            "bytes": record.identity.bytes,
            "original_path": str(record.identity.path),
            "references": sorted(
                record.references,
                key=lambda item: (
                    item["manifest_row"],
                    item["field"],
                    item["manifest_value"],
                ),
            ),
            "sha256": record.identity.sha256,
        }
        lines.append(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _copy_verified(source: FileIdentity, destination: Path) -> None:
    _verify_identity(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(source.path, destination)
    except OSError as exc:
        raise PortableBundleError(
            f"Cannot copy referenced file {source.path} to {destination}: {exc}"
        ) from exc
    copied_sha256, copied_bytes = _hash_file(destination)
    if (copied_sha256, copied_bytes) != (source.sha256, source.bytes):
        raise PortableBundleError(
            f"Copied payload failed verification: {source.path} -> {destination}"
        )
    try:
        with destination.open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError as exc:
        raise PortableBundleError(f"Cannot sync copied payload {destination}: {exc}") from exc


def _verify_identity(identity: FileIdentity) -> None:
    current_sha256, current_bytes = _hash_file(identity.path)
    if (current_sha256, current_bytes) != (identity.sha256, identity.bytes):
        raise PortableBundleError(
            f"Referenced file changed after bundle planning: {identity.path}"
        )


def _file_metadata(path: Path) -> Dict[str, Any]:
    sha256, byte_count = _hash_file(path)
    return {"bytes": byte_count, "sha256": sha256}


def materialize_bundle(
    input_manifest: Path,
    output_dir: Path,
    output_manifest: Optional[str] = None,
) -> Dict[str, Any]:
    manifest_path = input_manifest.resolve()
    initial_manifest_metadata = _file_metadata(manifest_path)
    document = read_manifest(manifest_path)
    output_manifest_name = _validate_output_manifest_name(
        output_manifest, document.format
    )
    output_path = output_dir.resolve()
    if output_path.exists():
        raise PortableBundleError(f"Output directory already exists: {output_path}")
    output_parent = output_path.parent
    if not output_parent.is_dir():
        raise PortableBundleError(
            f"Output parent directory does not exist: {output_parent}"
        )

    rewritten_rows, provenance, payloads = _plan_bundle(manifest_path, document)
    source_manifest_metadata = _file_metadata(manifest_path)
    if source_manifest_metadata != initial_manifest_metadata:
        raise PortableBundleError(
            f"Input manifest changed during bundle planning: {manifest_path}"
        )
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_parent)
    )
    try:
        for source_path in sorted(provenance, key=lambda value: str(value)):
            _verify_identity(provenance[source_path].identity)
        for bundle_path in sorted(payloads):
            _copy_verified(payloads[bundle_path], stage / bundle_path)

        manifest_bytes = _render_manifest(document, rewritten_rows)
        provenance_bytes = _render_provenance(provenance)
        portable_manifest_path = stage / output_manifest_name
        provenance_path = stage / "provenance.jsonl"
        _write_bytes_exclusive(portable_manifest_path, manifest_bytes)
        _write_bytes_exclusive(provenance_path, provenance_bytes)

        bundle_metadata = {
            "payload": {
                "content_address_count": len(payloads),
                "referenced_source_file_count": len(provenance),
                "total_content_addressed_bytes": sum(
                    identity.bytes for identity in payloads.values()
                ),
            },
            "portable_manifest": {
                "bytes": len(manifest_bytes),
                "path": output_manifest_name,
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            },
            "provenance": {
                "bytes": len(provenance_bytes),
                "path": "provenance.jsonl",
                "sha256": hashlib.sha256(provenance_bytes).hexdigest(),
            },
            "schema_version": SCHEMA_VERSION,
            "source_manifest": {
                "bytes": source_manifest_metadata["bytes"],
                "format": document.format,
                "original_path": str(manifest_path),
                "sha256": source_manifest_metadata["sha256"],
            },
            "system_row_count": len(rewritten_rows),
        }
        metadata_bytes = (
            json.dumps(
                bundle_metadata,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        _write_bytes_exclusive(stage / "bundle.json", metadata_bytes)

        if output_path.exists():
            raise PortableBundleError(
                f"Output directory appeared during materialization: {output_path}"
            )
        try:
            os.rename(stage, output_path)
        except OSError as exc:
            raise PortableBundleError(
                f"Cannot publish bundle directory {output_path}: {exc}"
            ) from exc
        stage = Path()
    except Exception:
        if stage != Path() and stage.exists():
            shutil.rmtree(stage)
        raise

    return bundle_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--output-manifest",
        help="Root-level output manifest filename; defaults to manifest.tsv/jsonl",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        metadata = materialize_bundle(
            arguments.input_manifest,
            arguments.output_dir,
            arguments.output_manifest,
        )
    except PortableBundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
