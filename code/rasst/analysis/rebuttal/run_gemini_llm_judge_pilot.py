#!/usr/bin/env python3
"""Prepare and run a paired Gemini LLM-as-a-judge cost pilot."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import random
import re
import stat
import statistics
import tempfile
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple


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
PROMPT_SOURCE = "https://aclanthology.org/2025.wmt-1.24.pdf#page=32"
LANGUAGE_NAMES = {"de": "German", "ja": "Japanese", "zh": "Chinese"}
METHODS = ("InfiniSST", "RASST")
EXPECTED_DATASETS = ("acl_tagged_raw", "medicine_hardraw")
SCORE_RE = re.compile(r"\A[ \t\r\n]*(0|[1-9][0-9]?|100)[ \t\r\n]*\Z")
SCHEMA_VERSION = "rasst-gemini-llm-judge-pilot-v1"
PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"
PRICING_DATE = "2026-07-12"
PRICES_PER_MILLION = {
    "gemini-2.5-pro": {
        "standard_input": 1.25,
        "standard_output": 10.0,
        "batch_input": 0.625,
        "batch_output": 5.0,
    },
    "gemini-2.5-flash": {
        "standard_input": 0.30,
        "standard_output": 2.50,
        "batch_input": 0.15,
        "batch_output": 1.25,
    },
    "gemini-3.1-pro-preview": {
        "standard_input": 2.0,
        "standard_output": 12.0,
        "batch_input": 1.0,
        "batch_output": 6.0,
    },
}


class PilotError(RuntimeError):
    """Raised when the pilot cannot proceed without changing its protocol."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise PilotError(f"Blank JSONL line at {path}:{line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PilotError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise PilotError(f"Expected an object at {path}:{line_number}")
                yield line_number, row
    except UnicodeDecodeError as exc:
        raise PilotError(f"Input is not valid UTF-8: {path}") from exc


def format_prompt(lang: str, source: str, hypothesis: str) -> str:
    if lang not in LANGUAGE_NAMES:
        raise PilotError(f"Unsupported target language: {lang!r}")
    if not isinstance(source, str) or not source:
        raise PilotError("Source text must be a non-empty string")
    if not isinstance(hypothesis, str):
        raise PilotError("Hypothesis must be a string; an empty string is allowed")
    return PROMPT_TEMPLATE.format(
        source_lang="English",
        target_lang=LANGUAGE_NAMES[lang],
        source_seg=source,
        target_seg=hypothesis,
    )


def parse_score(text: str) -> int:
    match = SCORE_RE.fullmatch(text)
    if match is None:
        raise PilotError(f"Gemini response is not one integer from 0 to 100: {text!r}")
    return int(match.group(1))


def _non_empty(row: Mapping[str, Any], field: str, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise PilotError(f"Missing {field!r} for {context}")
    return value.strip()


def _segment_index(row: Mapping[str, Any], context: str) -> int:
    value = row.get("talk_sentence_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PilotError(f"Invalid talk_sentence_index for {context}: {value!r}")
    return value


def load_matrix(
    acl_path: Path,
    medicine_path: Path,
    *,
    expected_acl_sha256: str,
    expected_medicine_sha256: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    inputs = [
        (lexical_absolute(acl_path), "acl_tagged_raw", expected_acl_sha256.lower()),
        (lexical_absolute(medicine_path), "medicine_hardraw", expected_medicine_sha256.lower()),
    ]
    rows: List[Dict[str, Any]] = []
    provenance: List[Dict[str, Any]] = []
    seen = set()
    for path, wanted_dataset, expected_sha256 in inputs:
        if not path.is_file():
            raise PilotError(f"Input is not a file: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise PilotError(
                f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
            )
        selected = 0
        for line_number, raw in iter_jsonl(path):
            if raw.get("dataset") != wanted_dataset:
                continue
            context = f"{path}:{line_number}"
            dataset = _non_empty(raw, "dataset", context)
            method = _non_empty(raw, "method", context)
            lang = _non_empty(raw, "lang", context)
            lm = _non_empty(raw, "lm", context)
            talk_id = _non_empty(raw, "talk_id", context)
            index = _segment_index(raw, context)
            source = raw.get("source")
            hypothesis = raw.get("hypothesis")
            reference = raw.get("reference")
            format_prompt(lang, source, hypothesis)
            if not isinstance(reference, str):
                raise PilotError(f"Reference alignment text must be a string at {context}")
            if method not in METHODS:
                raise PilotError(f"Unexpected method at {context}: {method!r}")
            identity = (dataset, method, lang, lm, talk_id, index)
            if identity in seen:
                raise PilotError(f"Duplicate system-segment identity: {identity!r}")
            seen.add(identity)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "lang": lang,
                    "lm": lm,
                    "talk_id": talk_id,
                    "talk_sentence_index": index,
                    "source": source,
                    "hypothesis": hypothesis,
                    "reference_sha256": sha256_bytes(reference.encode("utf-8")),
                    "source_artifact_sha256": actual_sha256,
                    "source_record_line": line_number,
                }
            )
            selected += 1
        provenance.append(
            {
                "path": str(path),
                "sha256": actual_sha256,
                "selection_dataset": wanted_dataset,
                "selected_rows": selected,
            }
        )

    expected_systems = {
        ("acl_tagged_raw", method, lang, str(lm))
        for method in METHODS
        for lang in ("de", "ja", "zh")
        for lm in range(1, 5)
    } | {
        ("medicine_hardraw", method, "de", str(lm))
        for method in METHODS
        for lm in range(1, 5)
    }
    counts = Counter((row["dataset"], row["method"], row["lang"], row["lm"]) for row in rows)
    if set(counts) != expected_systems:
        raise PilotError(
            f"Unexpected system matrix; missing={sorted(expected_systems - set(counts))}, "
            f"extra={sorted(set(counts) - expected_systems)}"
        )
    for system, count in counts.items():
        expected = 468 if system[0] == "acl_tagged_raw" else 1437
        if count != expected:
            raise PilotError(f"Unexpected segment count for {system!r}: {count} != {expected}")
    if len(rows) != 22_728:
        raise PilotError(f"Expected 22,728 matrix rows, got {len(rows)}")
    return rows, provenance


def _allocate_samples(cell_pair_counts: Mapping[Tuple[str, str, str], int], total: int) -> Dict[Tuple[str, str, str], int]:
    population = sum(cell_pair_counts.values())
    raw = {cell: total * count / population for cell, count in cell_pair_counts.items()}
    allocation = {cell: math.floor(value) for cell, value in raw.items()}
    remaining = total - sum(allocation.values())
    ranked = sorted(raw, key=lambda cell: (-(raw[cell] - allocation[cell]), cell))
    for cell in ranked[:remaining]:
        allocation[cell] += 1
    return allocation


def prepare_sample(args: argparse.Namespace) -> None:
    rows, provenance = load_matrix(
        args.acl_segments,
        args.medicine_segments,
        expected_acl_sha256=args.expected_acl_sha256,
        expected_medicine_sha256=args.expected_medicine_sha256,
    )
    if args.system_requests <= 0 or args.system_requests % 2:
        raise PilotError("--system-requests must be a positive even number")

    paired: Dict[Tuple[str, str, str], Dict[Tuple[str, int], Dict[str, Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        cell = (row["dataset"], row["lang"], row["lm"])
        pair_key = (row["talk_id"], row["talk_sentence_index"])
        paired[cell][pair_key][row["method"]] = row

    for cell, pairs in paired.items():
        for pair_key, methods in pairs.items():
            if set(methods) != set(METHODS):
                raise PilotError(f"Incomplete method pair for {cell!r}/{pair_key!r}")
            left, right = (methods[method] for method in METHODS)
            if left["source"] != right["source"] or left["reference_sha256"] != right["reference_sha256"]:
                raise PilotError(f"Source/reference mismatch for {cell!r}/{pair_key!r}")

    pair_counts = {cell: len(pairs) for cell, pairs in paired.items()}
    allocation = _allocate_samples(pair_counts, args.system_requests // 2)
    prompt_sha256 = sha256_bytes(PROMPT_TEMPLATE.encode("utf-8"))
    sampled: List[Dict[str, Any]] = []
    for cell in sorted(paired):
        ranked_pairs = sorted(
            paired[cell],
            key=lambda pair_key: canonical_json_sha256([*cell, *pair_key]),
        )
        for sample_rank, pair_key in enumerate(ranked_pairs[: allocation[cell]], start=1):
            methods = paired[cell][pair_key]
            for method in METHODS:
                row = methods[method]
                prompt = format_prompt(row["lang"], row["source"], row["hypothesis"])
                request_identity = {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "lang": row["lang"],
                    "lm": row["lm"],
                    "talk_id": row["talk_id"],
                    "talk_sentence_index": row["talk_sentence_index"],
                    "source": row["source"],
                    "hypothesis": row["hypothesis"],
                    "prompt_sha256": prompt_sha256,
                }
                sampled.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "request_key": canonical_json_sha256(request_identity),
                        "sample_rank_within_cell": sample_rank,
                        **row,
                        "prompt": prompt,
                        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                        "judge_input_sha256": canonical_json_sha256(
                            {
                                "source_lang": "English",
                                "target_lang": LANGUAGE_NAMES[row["lang"]],
                                "source": row["source"],
                                "translation": row["hypothesis"],
                                "prompt_template_sha256": prompt_sha256,
                            }
                        ),
                    }
                )

    sampled.sort(
        key=lambda row: (
            row["dataset"],
            row["lang"],
            int(row["lm"]),
            row["sample_rank_within_cell"],
            row["talk_id"],
            row["talk_sentence_index"],
            row["method"],
        )
    )
    if len(sampled) != args.system_requests:
        raise PilotError(f"Expected {args.system_requests} sampled requests, got {len(sampled)}")
    if len({row["request_key"] for row in sampled}) != len(sampled):
        raise PilotError("Sample contains duplicate request keys")

    request_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sampled
    )
    atomic_write_text(args.output_requests, request_text)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "protocol": {
            "prompt_template": PROMPT_TEMPLATE,
            "prompt_template_sha256": prompt_sha256,
            "prompt_source": PROMPT_SOURCE,
            "source_language": "English",
            "target_language_names": LANGUAGE_NAMES,
            "generation_config": {},
            "thinking": "model default",
            "sampling": "deterministic proportional paired stratification by dataset/language/lm",
        },
        "inputs": provenance,
        "full_matrix": {
            "system_requests": len(rows),
            "paired_segments": len(rows) // 2,
            "cell_system_request_counts": {
                "/".join(cell): pair_counts[cell] * 2 for cell in sorted(pair_counts)
            },
        },
        "sample": {
            "system_requests": len(sampled),
            "paired_segments": len(sampled) // 2,
            "cell_pair_allocation": {
                "/".join(cell): allocation[cell] for cell in sorted(allocation)
            },
            "requests_path": str(lexical_absolute(args.output_requests)),
            "requests_sha256": sha256_file(args.output_requests),
        },
    }
    atomic_write_text(
        args.output_manifest,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "requests": len(sampled),
                "paired_segments": len(sampled) // 2,
                "output": str(lexical_absolute(args.output_requests)),
                "sha256": sha256_file(args.output_requests),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def read_api_key(path: Path) -> str:
    if path.is_symlink():
        raise PilotError(f"API key path must not be a symlink: {path}")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise PilotError(f"API key path must be a regular file: {path}")
    if info.st_uid != os.getuid():
        raise PilotError(f"API key file must be owned by uid {os.getuid()}: {path}")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise PilotError(f"API key file must be owner-only (0600 or stricter): {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise PilotError(f"API key file is empty: {path}")
    return value


def load_requests(path: Path) -> List[Dict[str, Any]]:
    rows = [row for _, row in iter_jsonl(path)]
    if not rows:
        raise PilotError(f"No requests found: {path}")
    keys = [str(row.get("request_key") or "") for row in rows]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise PilotError("Request keys must be non-empty and unique")
    for row in rows:
        expected_prompt = format_prompt(row["lang"], row["source"], row["hypothesis"])
        if row.get("prompt") != expected_prompt:
            raise PilotError(f"Prompt mismatch for request {row['request_key']}")
    return rows


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value or ""))


def _usage_dict(usage: Any) -> Dict[str, int]:
    fields = (
        "prompt_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "cached_content_token_count",
        "tool_use_prompt_token_count",
        "total_token_count",
    )
    return {field: int(getattr(usage, field, 0) or 0) for field in fields}


def _response_dict(response: Any) -> Dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json", by_alias=True, exclude_none=True)
    return {"repr": repr(response)}


def _score_one(client: Any, model: str, request: Mapping[str, Any]) -> Dict[str, Any]:
    response = client.models.generate_content(model=model, contents=request["prompt"])
    candidates = list(getattr(response, "candidates", None) or [])
    if len(candidates) != 1:
        raise PilotError(f"Expected one candidate, got {len(candidates)}")
    candidate = candidates[0]
    parts = list(getattr(getattr(candidate, "content", None), "parts", None) or [])
    visible_parts = [
        str(getattr(part, "text", "") or "")
        for part in parts
        if not bool(getattr(part, "thought", False)) and getattr(part, "text", None) is not None
    ]
    if len(visible_parts) != 1:
        raise PilotError(f"Expected one visible text part, got {len(visible_parts)}")
    raw_text = visible_parts[0]
    score = parse_score(raw_text)
    finish_reason = _enum_name(getattr(candidate, "finish_reason", ""))
    if finish_reason not in {"STOP", "FinishReason.STOP"}:
        raise PilotError(f"Unexpected finish reason: {finish_reason!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "model": model,
        "request_key": request["request_key"],
        "success": True,
        "score": score,
        "raw_text": raw_text,
        "finish_reason": finish_reason,
        "model_version": str(getattr(response, "model_version", "") or ""),
        "response_id": str(getattr(response, "response_id", "") or ""),
        "usage": _usage_dict(getattr(response, "usage_metadata", None)),
        "response": _response_dict(response),
    }


def run_model(args: argparse.Namespace) -> None:
    if args.model not in PRICES_PER_MILLION:
        raise PilotError(f"Pilot model must be one of {sorted(PRICES_PER_MILLION)}")
    requests = load_requests(args.requests)
    if args.limit is not None:
        if args.limit <= 0:
            raise PilotError("--limit must be positive")
        requests = requests[: args.limit]
    api_key = read_api_key(args.api_key_file)
    existing: Dict[str, Dict[str, Any]] = {}
    if args.output.exists():
        for _, row in iter_jsonl(args.output):
            if row.get("model") != args.model:
                raise PilotError(f"Output contains a different model: {row.get('model')!r}")
            key = str(row.get("request_key") or "")
            if not key or key in existing:
                raise PilotError(f"Duplicate or empty result key: {key!r}")
            existing[key] = row
    pending = [request for request in requests if request["request_key"] not in existing]
    if not pending:
        print(json.dumps({"model": args.model, "pending": 0, "complete": len(existing)}))
        return

    try:
        from google import genai
        import google.genai as google_genai
    except ImportError as exc:
        raise PilotError("google-genai is required for scoring") from exc

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_lock = threading.Lock()
    thread_local = threading.local()

    def score(request: Mapping[str, Any]) -> Dict[str, Any]:
        if not hasattr(thread_local, "client"):
            thread_local.client = genai.Client(api_key=api_key)
        try:
            return _score_one(thread_local.client, args.model, request)
        except Exception as exc:
            return {
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": utc_now(),
                "model": args.model,
                "request_key": request["request_key"],
                "success": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

    mode = "a" if output_path.exists() else "w"
    completed = len(existing)
    with output_path.open(mode, encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(score, request): request for request in pending}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with output_lock:
                    handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    completed += 1
                    if completed % 10 == 0 or completed == len(requests):
                        print(
                            json.dumps(
                                {
                                    "model": args.model,
                                    "completed": completed,
                                    "total": len(requests),
                                    "latest_success": result["success"],
                                    "sdk_version": getattr(google_genai, "__version__", "unknown"),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
    failures = sum(1 for _, row in iter_jsonl(output_path) if not row.get("success"))
    if failures:
        raise PilotError(f"{args.model} pilot contains {failures} failed requests")


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise PilotError("Cannot average an empty sequence")
    return float(statistics.fmean(values))


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise PilotError("Pearson correlation requires two aligned non-trivial sequences")
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else float("nan")


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values or not 0.0 <= probability <= 1.0:
        raise PilotError("Invalid percentile input")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def build_report(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requests = load_requests(args.requests)
    request_by_key = {row["request_key"]: row for row in requests}
    full_counts = {
        tuple(key.split("/")): int(value)
        for key, value in manifest["full_matrix"]["cell_system_request_counts"].items()
    }
    results_by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    model_files = {
        args.pro_model: args.pro_results,
        "gemini-2.5-flash": args.flash_results,
    }
    for model, path in model_files.items():
        indexed: Dict[str, Dict[str, Any]] = {}
        for _, row in iter_jsonl(path):
            key = str(row.get("request_key") or "")
            if key not in request_by_key or key in indexed:
                raise PilotError(f"Unknown or duplicate {model} result key: {key!r}")
            if not row.get("success"):
                raise PilotError(f"Failed {model} result for key {key}")
            if isinstance(row.get("score"), bool) or not isinstance(row.get("score"), int):
                raise PilotError(f"Non-integer {model} score for key {key}")
            indexed[key] = row
        if set(indexed) != set(request_by_key):
            raise PilotError(f"Incomplete {model} result set: {len(indexed)}/{len(request_by_key)}")
        results_by_model[model] = indexed

    model_summaries: Dict[str, Dict[str, Any]] = {}
    for model, indexed in results_by_model.items():
        prices = PRICES_PER_MILLION[model]
        usage_fields = (
            "prompt_token_count",
            "candidates_token_count",
            "thoughts_token_count",
            "total_token_count",
        )
        totals = {
            field: sum(int(row["usage"].get(field, 0)) for row in indexed.values())
            for field in usage_fields
        }
        per_cell: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for key, result in indexed.items():
            request = request_by_key[key]
            per_cell[(request["dataset"], request["lang"], request["lm"])].append(result)
        projected_prompt = projected_candidate = projected_thought = 0.0
        cell_projection: Dict[str, Any] = {}
        for cell in sorted(full_counts):
            cell_results = per_cell[cell]
            prompt_mean = _mean([row["usage"]["prompt_token_count"] for row in cell_results])
            candidate_mean = _mean([row["usage"]["candidates_token_count"] for row in cell_results])
            thought_mean = _mean([row["usage"]["thoughts_token_count"] for row in cell_results])
            count = full_counts[cell]
            projected_prompt += prompt_mean * count
            projected_candidate += candidate_mean * count
            projected_thought += thought_mean * count
            cell_projection["/".join(cell)] = {
                "sample_requests": len(cell_results),
                "full_requests": count,
                "mean_prompt_tokens": prompt_mean,
                "mean_candidate_tokens": candidate_mean,
                "mean_thought_tokens": thought_mean,
            }
        projected_output = projected_candidate + projected_thought
        bootstrap_rng = random.Random(canonical_json_sha256([model, "cost-bootstrap-20260712"]))
        bootstrap_costs: List[float] = []
        for _ in range(5_000):
            bootstrap_prompt = 0.0
            bootstrap_output = 0.0
            for cell, full_count in full_counts.items():
                cell_results = per_cell[cell]
                resampled = [
                    cell_results[bootstrap_rng.randrange(len(cell_results))]
                    for _ in cell_results
                ]
                bootstrap_prompt += _mean(
                    [row["usage"]["prompt_token_count"] for row in resampled]
                ) * full_count
                bootstrap_output += _mean(
                    [
                        row["usage"]["candidates_token_count"]
                        + row["usage"]["thoughts_token_count"]
                        for row in resampled
                    ]
                ) * full_count
            bootstrap_costs.append(
                (
                    bootstrap_prompt * prices["batch_input"]
                    + bootstrap_output * prices["batch_output"]
                )
                / 1_000_000
            )
        bootstrap_costs.sort()
        model_summaries[model] = {
            "sample": {
                "requests": len(indexed),
                **totals,
                "mean_prompt_tokens": totals["prompt_token_count"] / len(indexed),
                "mean_candidate_tokens": totals["candidates_token_count"] / len(indexed),
                "mean_thought_tokens": totals["thoughts_token_count"] / len(indexed),
                "score_mean": _mean([row["score"] for row in indexed.values()]),
                "standard_api_estimated_cost_usd": (
                    totals["prompt_token_count"] * prices["standard_input"]
                    + (totals["candidates_token_count"] + totals["thoughts_token_count"])
                    * prices["standard_output"]
                )
                / 1_000_000,
            },
            "full_projection": {
                "requests": sum(full_counts.values()),
                "prompt_tokens": projected_prompt,
                "candidate_tokens": projected_candidate,
                "thought_tokens": projected_thought,
                "priced_output_tokens": projected_output,
                "batch_cost_usd": (
                    projected_prompt * prices["batch_input"]
                    + projected_output * prices["batch_output"]
                )
                / 1_000_000,
                "batch_cost_sample_bootstrap_95pct_usd": [
                    _percentile(bootstrap_costs, 0.025),
                    _percentile(bootstrap_costs, 0.975),
                ],
                "standard_cost_usd": (
                    projected_prompt * prices["standard_input"]
                    + projected_output * prices["standard_output"]
                )
                / 1_000_000,
            },
            "cell_projection": cell_projection,
            "model_versions": sorted({row.get("model_version", "") for row in indexed.values()}),
        }

    keys = sorted(request_by_key)
    pro_scores = [results_by_model[args.pro_model][key]["score"] for key in keys]
    flash_scores = [results_by_model["gemini-2.5-flash"][key]["score"] for key in keys]
    differences = [pro - flash for pro, flash in zip(pro_scores, flash_scores)]
    comparison = {
        "aligned_requests": len(keys),
        "pearson": _pearson(pro_scores, flash_scores),
        "mean_pro_minus_flash": _mean(differences),
        "mean_absolute_difference": _mean([abs(value) for value in differences]),
        "exact_agreement": sum(pro == flash for pro, flash in zip(pro_scores, flash_scores)),
        "pro_higher": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "flash_higher": sum(value < 0 for value in differences),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "pricing": {
            "date": PRICING_DATE,
            "source": PRICING_SOURCE,
            "per_million_tokens_usd": PRICES_PER_MILLION,
            "output_includes_thinking_tokens": True,
        },
        "sample_manifest_sha256": sha256_file(args.manifest),
        "requests_sha256": sha256_file(args.requests),
        "models": model_summaries,
        "score_comparison": comparison,
    }
    atomic_write_text(
        args.output_json,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    lines = [
        "# Gemini LLM-as-a-judge 100-request pilot",
        "",
        f"- Sample: {len(keys)} system outputs, including paired RASST/InfiniSST outputs.",
        f"- Full projection: {sum(full_counts.values()):,} system outputs; baseline already included.",
        f"- Prompt: WMT25 Appendix A; generation config omitted; model-default thinking.",
        "",
        "| Model | Avg input | Avg visible output | Avg thinking | Pilot standard cost | Full Batch projection |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in (args.pro_model, "gemini-2.5-flash"):
        sample = model_summaries[model]["sample"]
        projection = model_summaries[model]["full_projection"]
        lines.append(
            f"| {model} | {sample['mean_prompt_tokens']:.2f} | "
            f"{sample['mean_candidate_tokens']:.2f} | {sample['mean_thought_tokens']:.2f} | "
            f"${sample['standard_api_estimated_cost_usd']:.4f} | ${projection['batch_cost_usd']:.2f} "
            f"(sample bootstrap ${projection['batch_cost_sample_bootstrap_95pct_usd'][0]:.2f}–"
            f"${projection['batch_cost_sample_bootstrap_95pct_usd'][1]:.2f}) |"
        )
    lines.extend(
        [
            "",
            "## Score comparison",
            "",
            f"- Pearson: {comparison['pearson']:.4f}",
            f"- Mean Pro - Flash: {comparison['mean_pro_minus_flash']:.3f}",
            f"- Mean absolute difference: {comparison['mean_absolute_difference']:.3f}",
            f"- Pro higher / tie / Flash higher: {comparison['pro_higher']} / "
            f"{comparison['ties']} / {comparison['flash_higher']}",
            "",
        ]
    )
    atomic_write_text(args.output_markdown, "\n".join(lines))
    print(json.dumps(report["models"], ensure_ascii=False, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--acl-segments", required=True, type=Path)
    prepare.add_argument("--medicine-segments", required=True, type=Path)
    prepare.add_argument("--expected-acl-sha256", required=True)
    prepare.add_argument("--expected-medicine-sha256", required=True)
    prepare.add_argument("--system-requests", required=True, type=int)
    prepare.add_argument("--output-requests", required=True, type=Path)
    prepare.add_argument("--output-manifest", required=True, type=Path)
    prepare.set_defaults(func=prepare_sample)

    score = subparsers.add_parser("score")
    score.add_argument("--requests", required=True, type=Path)
    score.add_argument("--api-key-file", required=True, type=Path)
    score.add_argument("--model", required=True)
    score.add_argument("--workers", required=True, type=int)
    score.add_argument("--limit", type=int)
    score.add_argument("--output", required=True, type=Path)
    score.set_defaults(func=run_model)

    report = subparsers.add_parser("report")
    report.add_argument("--manifest", required=True, type=Path)
    report.add_argument("--requests", required=True, type=Path)
    report.add_argument("--pro-model", required=True)
    report.add_argument("--pro-results", required=True, type=Path)
    report.add_argument("--flash-results", required=True, type=Path)
    report.add_argument("--output-json", required=True, type=Path)
    report.add_argument("--output-markdown", required=True, type=Path)
    report.set_defaults(func=build_report)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if getattr(args, "workers", 1) <= 0:
        raise PilotError("--workers must be positive")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
