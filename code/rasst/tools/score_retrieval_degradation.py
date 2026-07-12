#!/usr/bin/env python3
"""Aggregate achieved retrieval degradation from RASST runtime JSONL logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


COUNT_FIELDS = (
    "hint_count_original",
    "hint_count_final",
    "relevant_gold_count",
    "relevant_hint_count_original",
    "relevant_hint_count_final",
    "replaced_relevant_hint_count",
)


def _records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if isinstance(row, dict):
                yield row


def score_runtime_log(path: Path) -> Dict[str, Any]:
    totals = {field: 0 for field in COUNT_FIELDS}
    event_count = 0
    configs = set()
    for row in _records(path):
        if row.get("type") != "llm_input":
            continue
        audit = row.get("retrieval_degradation")
        if not isinstance(audit, dict):
            continue
        event_count += 1
        configs.add((float(audit["configured_rate"]), int(audit["seed"])))
        for field in COUNT_FIELDS:
            totals[field] += int(audit.get(field) or 0)
    if not event_count:
        raise ValueError(f"No degraded llm_input records in {path}")
    if len(configs) != 1:
        raise ValueError(f"Mixed degradation configs in one runtime log: {sorted(configs)}")

    rate, seed = next(iter(configs))

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator) / float(denominator) if denominator else 0.0

    result = {
        "runtime_log": str(path.resolve()),
        "configured_rate": rate,
        "seed": seed,
        "retrieval_events": event_count,
        **totals,
        "hint_count_preserved": totals["hint_count_original"]
        == totals["hint_count_final"],
        "achieved_replacement_rate": ratio(
            totals["replaced_relevant_hint_count"],
            totals["relevant_hint_count_original"],
        ),
        "retrieval_precision_original": ratio(
            totals["relevant_hint_count_original"], totals["hint_count_original"]
        ),
        "retrieval_precision_final": ratio(
            totals["relevant_hint_count_final"], totals["hint_count_final"]
        ),
        "retrieval_recall_original": ratio(
            totals["relevant_hint_count_original"], totals["relevant_gold_count"]
        ),
        "retrieval_recall_final": ratio(
            totals["relevant_hint_count_final"], totals["relevant_gold_count"]
        ),
    }
    if not result["hint_count_preserved"]:
        raise ValueError("Hint count changed during retrieval degradation")
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-log", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = score_runtime_log(Path(args.runtime_log))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
