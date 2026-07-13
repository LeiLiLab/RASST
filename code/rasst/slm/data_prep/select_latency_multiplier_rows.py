#!/usr/bin/env python3
"""Select Speech LLM rows by recorded audio-chunk latency multiplier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


SCHEMA_VERSION = "rasst-slm-latency-selection-v1"


class LatencySelectionError(RuntimeError):
    """Raised when latency rows cannot be selected without guessing."""


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LatencySelectionError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise LatencySelectionError(
                    f"Expected JSON object at {path}:{line_number}"
                )
            yield line_number, row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_multipliers(
    row: Mapping[str, Any],
    *,
    path: Path,
    line_number: int,
) -> list[int]:
    metadata = row.get("chunk_metadata")
    audios = row.get("audios")
    if not isinstance(metadata, list) or not metadata:
        raise LatencySelectionError(
            f"Missing non-empty chunk_metadata at {path}:{line_number}"
        )
    if not isinstance(audios, list) or len(audios) != len(metadata):
        raise LatencySelectionError(
            f"audio/chunk_metadata mismatch at {path}:{line_number}"
        )
    values: list[int] = []
    for chunk_index, item in enumerate(metadata):
        if not isinstance(item, Mapping):
            raise LatencySelectionError(
                f"Invalid chunk_metadata[{chunk_index}] at {path}:{line_number}"
            )
        raw_value = item.get("multiplier")
        if isinstance(raw_value, bool):
            raise LatencySelectionError(
                f"Boolean multiplier at {path}:{line_number}:{chunk_index}"
            )
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise LatencySelectionError(
                f"Invalid multiplier at {path}:{line_number}:{chunk_index}: {raw_value!r}"
            ) from exc
        if value <= 0:
            raise LatencySelectionError(
                f"Non-positive multiplier at {path}:{line_number}:{chunk_index}: {value}"
            )
        values.append(value)
    return values


def matches(values: Sequence[int], focus: int, policy: str) -> bool:
    if policy == "all":
        return all(value == focus for value in values)
    if policy == "any":
        return any(value == focus for value in values)
    if policy == "first":
        return values[0] == focus
    if policy == "dominant":
        counts = Counter(values)
        dominant = min(
            (value for value, count in counts.items() if count == max(counts.values())),
        )
        return dominant == focus
    raise LatencySelectionError(f"Unsupported match policy: {policy!r}")


def atomic_write(path: Path, text: str) -> None:
    path = path.resolve()
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


def run(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = args.input_jsonl.resolve()
    output_path = args.output_jsonl.resolve()
    stats_path = args.stats_json.resolve()
    if not input_path.is_file():
        raise LatencySelectionError(f"Input JSONL is not a file: {input_path}")
    if output_path in {input_path, stats_path} or stats_path == input_path:
        raise LatencySelectionError("Input and output paths must be distinct")
    if args.focus_multiplier <= 0:
        raise LatencySelectionError("focus_multiplier must be positive")

    input_sha256 = sha256_file(input_path)
    selected_lines: list[str] = []
    total_rows = selected_rows = total_chunks = selected_chunks = 0
    row_first_hist: Counter[int] = Counter()
    chunk_hist: Counter[int] = Counter()
    for line_number, row in iter_jsonl(input_path):
        values = row_multipliers(
            row,
            path=input_path,
            line_number=line_number,
        )
        total_rows += 1
        total_chunks += len(values)
        row_first_hist[values[0]] += 1
        chunk_hist.update(values)
        if not matches(values, args.focus_multiplier, args.match_policy):
            continue
        selected_rows += 1
        selected_chunks += len(values)
        row["latency_multiplier_selection"] = {
            "schema_version": SCHEMA_VERSION,
            "source_jsonl": str(input_path),
            "source_jsonl_sha256": input_sha256,
            "source_line_number": line_number,
            "focus_multiplier": args.focus_multiplier,
            "match_policy": args.match_policy,
            "row_multipliers": values,
        }
        selected_lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))

    if selected_rows == 0:
        raise LatencySelectionError("Selection produced zero rows")
    if args.expected_rows is not None and selected_rows != args.expected_rows:
        raise LatencySelectionError(
            f"Selected row count mismatch: expected={args.expected_rows} actual={selected_rows}"
        )
    atomic_write(output_path, "\n".join(selected_lines) + "\n")
    output_sha256 = sha256_file(output_path)
    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": input_sha256,
        "output_jsonl": str(output_path),
        "output_jsonl_sha256": output_sha256,
        "focus_multiplier": args.focus_multiplier,
        "match_policy": args.match_policy,
        "total_rows": total_rows,
        "selected_rows": selected_rows,
        "selected_row_rate": selected_rows / total_rows,
        "total_chunks": total_chunks,
        "selected_chunks": selected_chunks,
        "row_first_multiplier_hist": {
            str(key): value for key, value in sorted(row_first_hist.items())
        },
        "chunk_multiplier_hist": {
            str(key): value for key, value in sorted(chunk_hist.items())
        },
    }
    atomic_write(stats_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--focus-multiplier", type=int, required=True)
    parser.add_argument(
        "--match-policy",
        choices=("all", "any", "first", "dominant"),
        default="all",
    )
    parser.add_argument("--expected-rows", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run(args)
    except LatencySelectionError as exc:
        print(f"[ERROR] {exc}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
