#!/usr/bin/env python3
"""Append verified latency-focused SFT variants to a base Speech LLM dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


SCHEMA_VERSION = "rasst-slm-latency-curriculum-v1"


class CurriculumError(RuntimeError):
    """Raised when a curriculum dataset cannot be assembled safely."""


def absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CurriculumError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise CurriculumError(f"Expected object at {path}:{line_number}")
            yield line_number, row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_row(row: Mapping[str, Any], path: Path, line_number: int) -> None:
    messages = row.get("messages")
    audios = row.get("audios")
    gt_terms = row.get("gt_terms_by_chunk")
    if not isinstance(messages, list) or not isinstance(audios, list) or not isinstance(gt_terms, list):
        raise CurriculumError(f"Malformed SFT row at {path}:{line_number}")
    user_count = sum(
        message.get("role") == "user"
        and str(message.get("content") or "").startswith("<audio>")
        for message in messages
        if isinstance(message, Mapping)
    )
    if user_count != len(audios) or len(gt_terms) != len(audios):
        raise CurriculumError(
            f"user/audio/gt mismatch at {path}:{line_number}: "
            f"{user_count}/{len(audios)}/{len(gt_terms)}"
        )


def atomic_write(path: Path, text: str) -> None:
    path = absolute_without_symlink_resolution(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
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


def source_signature(row: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {
            "audios": row.get("audios"),
            "assistant": [
                message.get("content")
                for message in row.get("messages", [])
                if isinstance(message, Mapping) and message.get("role") == "assistant"
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    base_path = absolute_without_symlink_resolution(args.base_jsonl)
    supplement_paths = [
        absolute_without_symlink_resolution(path) for path in args.supplement_jsonl
    ]
    output_path = absolute_without_symlink_resolution(args.output_jsonl)
    stats_path = absolute_without_symlink_resolution(args.stats_json)
    protected = {base_path, *supplement_paths}
    if not base_path.is_file() or any(not path.is_file() for path in supplement_paths):
        raise CurriculumError("Every base/supplement input must be a file")
    if output_path in protected or stats_path in protected or output_path == stats_path:
        raise CurriculumError("Input and output paths must be distinct")
    if args.focus_multiplier <= 0:
        raise CurriculumError("focus_multiplier must be positive")

    lines: list[str] = []
    source_signatures: set[str] = set()
    base_rows = supplement_rows = repeated_sources = 0
    for line_number, row in iter_jsonl(base_path):
        validate_row(row, base_path, line_number)
        signature = source_signature(row)
        source_signatures.add(signature)
        row["latency_curriculum"] = {
            "schema_version": SCHEMA_VERSION,
            "role": "base",
            "source_jsonl": str(base_path),
            "source_line_number": line_number,
            "focus_multiplier": args.focus_multiplier,
        }
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
        base_rows += 1

    for variant_index, path in enumerate(supplement_paths, start=1):
        for line_number, row in iter_jsonl(path):
            validate_row(row, path, line_number)
            selection = row.get("latency_multiplier_selection")
            if not isinstance(selection, Mapping):
                raise CurriculumError(
                    f"Supplement lacks latency selection metadata at {path}:{line_number}"
                )
            if int(selection.get("focus_multiplier", -1)) != args.focus_multiplier:
                raise CurriculumError(
                    f"Supplement focus mismatch at {path}:{line_number}"
                )
            signature = source_signature(row)
            repeated_sources += int(signature in source_signatures)
            source_signatures.add(signature)
            row["latency_curriculum"] = {
                "schema_version": SCHEMA_VERSION,
                "role": "supplement",
                "variant_index": variant_index,
                "source_jsonl": str(path),
                "source_line_number": line_number,
                "focus_multiplier": args.focus_multiplier,
                "selection": dict(selection),
            }
            lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
            supplement_rows += 1

    if args.expected_base_rows is not None and base_rows != args.expected_base_rows:
        raise CurriculumError(
            f"Base row count mismatch: expected={args.expected_base_rows} actual={base_rows}"
        )
    if args.expected_supplement_rows is not None and supplement_rows != args.expected_supplement_rows:
        raise CurriculumError(
            "Supplement row count mismatch: "
            f"expected={args.expected_supplement_rows} actual={supplement_rows}"
        )
    atomic_write(output_path, "\n".join(lines) + "\n")
    total_rows = base_rows + supplement_rows
    focus_rows = args.base_focus_rows + supplement_rows
    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "base_jsonl": str(base_path),
        "base_jsonl_sha256": sha256_file(base_path),
        "supplement_jsonl": [str(path) for path in supplement_paths],
        "supplement_jsonl_sha256": [sha256_file(path) for path in supplement_paths],
        "output_jsonl": str(output_path),
        "output_jsonl_sha256": sha256_file(output_path),
        "focus_multiplier": args.focus_multiplier,
        "base_rows": base_rows,
        "base_focus_rows": args.base_focus_rows,
        "supplement_rows": supplement_rows,
        "total_rows": total_rows,
        "focus_rows_after": focus_rows,
        "focus_row_rate_before": args.base_focus_rows / base_rows,
        "focus_row_rate_after": focus_rows / total_rows,
        "repeated_source_signatures": repeated_sources,
        "note": "Repeated source signatures are intentional; term-map denoise seeds differ.",
    }
    atomic_write(stats_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--supplement-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--focus-multiplier", type=int, required=True)
    parser.add_argument("--base-focus-rows", type=int, required=True)
    parser.add_argument("--expected-base-rows", type=int)
    parser.add_argument("--expected-supplement-rows", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run(args)
    except CurriculumError as exc:
        print(f"[ERROR] {exc}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
