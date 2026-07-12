#!/usr/bin/env python3
"""Audit ACL terminology failures, retrieval timing, and paired xCOMET changes."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "rasst_term_failure_chain_v1"
RUNTIME_TYPES_WITH_SEGMENT = {"rag_window", "rag", "llm_input", "llm_output"}
AUDIT_LABELS = {
    "valid_morphology",
    "valid_compound_or_orthography",
    "valid_paraphrase",
    "valid_alignment_boundary",
    "wrong_translation",
    "omitted_term",
    "uncertain",
}
MORPHOLOGY_AWARE_LABELS = {
    "valid_morphology",
    "valid_compound_or_orthography",
}
SEMANTICALLY_VALID_LABELS = MORPHOLOGY_AWARE_LABELS | {
    "valid_paraphrase",
    "valid_alignment_boundary",
}
NOISE_AUDIT_LABELS = {
    "source_morphology_or_semantic_support",
    "alignment_boundary",
    "harmful_unsupported_adoption",
    "benign_unsupported_adoption",
    "uncertain",
}


@dataclass(frozen=True)
class AudioSentence:
    index: int
    paper_id: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class RetrievalEvent:
    paper_id: str
    segment_idx: int
    prompt_start_sec: float
    prompt_end_sec: float
    references: Tuple[Mapping[str, Any], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def match_normalize(value: Any) -> str:
    return normalize_space(value).casefold()


def text_contains_exact(text: str, target: str) -> bool:
    needle = normalize_space(target)
    return bool(needle) and needle in normalize_space(text)


def _coerce_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (value.startswith("\"") and value.endswith("\"")) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_flat_yaml_list(path: Path) -> List[Dict[str, Any]]:
    """Load the flat list-of-mappings schema used by SimulEval audio YAML."""
    rows: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith("- "):
            if current is not None:
                rows.append(current)
            current = {}
            content = raw[2:]
        elif raw.startswith("  ") and current is not None:
            content = raw.strip()
        else:
            raise ValueError(f"Unsupported audio YAML structure at {path}:{line_number}")
        if ":" not in content:
            raise ValueError(f"Invalid audio YAML field at {path}:{line_number}")
        key, value = content.split(":", 1)
        current[key.strip()] = _coerce_yaml_scalar(value)
    if current is not None:
        rows.append(current)
    if not rows:
        raise ValueError(f"No audio rows found in {path}")
    return rows


def load_audio_sentences(path: Path) -> List[AudioSentence]:
    out: List[AudioSentence] = []
    for index, row in enumerate(load_flat_yaml_list(path)):
        wav = normalize_space(row.get("wav"))
        if not wav:
            raise ValueError(f"audio row {index} has no wav: {path}")
        try:
            start = float(row["offset"])
            duration = float(row["duration"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"audio row {index} has invalid offset/duration: {path}") from exc
        if duration <= 0:
            raise ValueError(f"audio row {index} has non-positive duration: {duration}")
        out.append(
            AudioSentence(
                index=index,
                paper_id=Path(wav).stem,
                start_sec=start,
                end_sec=start + duration,
            )
        )
    return out


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield row


def split_runtime_groups(path: Path) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    seen_positive_segment = False
    for row in iter_jsonl(path):
        row_type = row.get("type")
        segment_idx = -1
        if row_type in RUNTIME_TYPES_WITH_SEGMENT:
            try:
                segment_idx = int(row.get("segment_idx", -1))
            except (TypeError, ValueError):
                segment_idx = -1
        if row_type == "rag_window" and segment_idx == 0 and current and seen_positive_segment:
            groups.append(current)
            current = []
            seen_positive_segment = False
        current.append(row)
        if segment_idx > 0:
            seen_positive_segment = True
    if current:
        groups.append(current)
    return groups


def ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def load_retrieval_events(
    runtime_log: Path,
    audio_sentences: Sequence[AudioSentence],
) -> Dict[str, List[RetrievalEvent]]:
    papers = ordered_unique(row.paper_id for row in audio_sentences)
    groups = split_runtime_groups(runtime_log)
    if len(groups) != len(papers):
        raise ValueError(
            f"runtime group count {len(groups)} != audio paper count {len(papers)}"
        )

    result: Dict[str, List[RetrievalEvent]] = {}
    for paper_id, records in zip(papers, groups):
        windows: Dict[int, Mapping[str, Any]] = {}
        for row in records:
            if row.get("type") != "rag_window":
                continue
            try:
                segment_idx = int(row.get("segment_idx", -1))
            except (TypeError, ValueError):
                continue
            if segment_idx >= 0:
                windows[segment_idx] = row

        events: List[RetrievalEvent] = []
        for row in records:
            if row.get("type") != "llm_input":
                continue
            try:
                segment_idx = int(row.get("segment_idx", -1))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid llm_input segment index for {paper_id}") from exc
            window = windows.get(segment_idx)
            if window is None:
                raise ValueError(f"Missing rag_window for {paper_id} segment {segment_idx}")
            try:
                start = float(
                    window.get("current_start_sec", window.get("rag_audio_duration", 0.0))
                )
                end = float(
                    window.get("current_end_sec", window.get("rag_audio_duration", 0.0))
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid retrieval window for {paper_id} segment {segment_idx}"
                ) from exc
            refs = row.get("references") or []
            if not isinstance(refs, list) or not all(isinstance(item, dict) for item in refs):
                raise ValueError(
                    f"Invalid llm_input references for {paper_id} segment {segment_idx}"
                )
            events.append(
                RetrievalEvent(
                    paper_id=paper_id,
                    segment_idx=segment_idx,
                    prompt_start_sec=start,
                    prompt_end_sec=end,
                    references=tuple(refs),
                )
            )
        if not events:
            raise ValueError(f"No llm_input events for {paper_id}")
        result[paper_id] = events
    return result


def reference_matches_term(term: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    source_term = match_normalize(term.get("term"))
    translation = match_normalize(term.get("translation"))
    ref_term = match_normalize(reference.get("term") or reference.get("key"))
    ref_translation = match_normalize(reference.get("translation"))
    if not translation or translation != ref_translation:
        return False
    return source_term == ref_term or translation == ref_translation


def retrieval_timing(
    term: Mapping[str, Any],
    sentence: AudioSentence,
    events: Sequence[RetrievalEvent],
) -> Tuple[str, Optional[float], Optional[float], Optional[int]]:
    hits: List[Tuple[float, float, int]] = []
    for event in events:
        for reference in event.references:
            if not reference_matches_term(term, reference):
                continue
            start_raw = reference.get("time_start")
            end_raw = reference.get("time_end")
            if start_raw is None or end_raw is None:
                raise ValueError(
                    "A matching retrieval reference lacks time_start/time_end; "
                    f"paper={sentence.paper_id} sentence={sentence.index} term={term.get('term')}"
                )
            ref_start = float(start_raw)
            ref_end = float(end_raw)
            if ref_start < sentence.end_sec and ref_end > sentence.start_sec:
                hits.append((event.prompt_end_sec, ref_end, event.segment_idx))
    if not hits:
        return "never_retrieved", None, None, None
    prompt_end, evidence_end, segment_idx = min(hits)
    if prompt_end <= sentence.end_sec + 1e-9:
        return "retrieved_on_time", prompt_end, evidence_end, segment_idx
    return "retrieved_late", prompt_end, evidence_end, segment_idx


def load_xcomet_pairs(
    path: Path,
    dataset: str,
    lang: str,
    lm: int,
    rasst_method: str,
    baseline_method: str,
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    selected: Dict[int, Dict[str, Dict[str, Any]]] = {}
    wanted_methods = {rasst_method, baseline_method}
    for row in iter_jsonl(path):
        try:
            row_lm = int(row.get("lm"))
        except (TypeError, ValueError):
            continue
        if (
            row.get("dataset") != dataset
            or row.get("lang") != lang
            or row_lm != lm
            or row.get("method") not in wanted_methods
        ):
            continue
        index = int(row["sentence_index"])
        method = str(row["method"])
        if method in selected.setdefault(index, {}):
            raise ValueError(f"Duplicate xCOMET row for sentence={index} method={method}")
        selected[index][method] = row
    if not selected:
        raise ValueError(f"No matching xCOMET rows in {path}")
    incomplete = [index for index, rows in selected.items() if set(rows) != wanted_methods]
    if incomplete:
        raise ValueError(f"Unpaired xCOMET rows for sentence indexes: {incomplete[:10]}")
    return selected


def _unicode_tokens(text: str) -> List[str]:
    return re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", text.casefold(), flags=re.UNICODE)


def compact_form(text: str) -> str:
    return re.sub(r"[^0-9a-zäöüß]+", "", text.casefold())


def morphology_candidate(translation: str, hypothesis: str) -> Dict[str, Any]:
    target = normalize_space(translation)
    hyp = normalize_space(hypothesis)
    if target.casefold() in hyp.casefold():
        start = hyp.casefold().find(target.casefold())
        return {
            "candidate_kind": "casefold_or_compound_substring",
            "candidate_span": hyp[start : start + len(target)],
            "candidate_score": 1.0,
        }
    compact_target = compact_form(target)
    compact_hyp = compact_form(hyp)
    if compact_target and compact_target in compact_hyp:
        return {
            "candidate_kind": "spacing_or_hyphen_variant",
            "candidate_span": compact_target,
            "candidate_score": 1.0,
        }

    target_tokens = _unicode_tokens(target)
    hypothesis_tokens = _unicode_tokens(hyp)
    best_score = 0.0
    best_span = ""
    if target_tokens and hypothesis_tokens:
        for width in range(max(1, len(target_tokens) - 1), len(target_tokens) + 2):
            for start in range(0, len(hypothesis_tokens) - width + 1):
                span_tokens = hypothesis_tokens[start : start + width]
                score = difflib.SequenceMatcher(
                    None,
                    " ".join(target_tokens),
                    " ".join(span_tokens),
                ).ratio()
                if score > best_score:
                    best_score = score
                    best_span = " ".join(span_tokens)
    return {
        "candidate_kind": "fuzzy_inflection_candidate" if best_score >= 0.78 else "no_candidate",
        "candidate_span": best_span,
        "candidate_score": best_score,
    }


def load_manual_audit(path: Optional[Path]) -> Dict[Tuple[int, str, str], Dict[str, str]]:
    if path is None:
        return {}
    rows: Dict[Tuple[int, str, str], Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sentence_index", "term", "translation", "audit_label", "audit_note"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manual audit must contain columns {sorted(required)}")
        for row in reader:
            key = (int(row["sentence_index"]), row["term"], row["translation"])
            label = row["audit_label"]
            if label not in AUDIT_LABELS:
                raise ValueError(f"Unsupported audit label {label!r} for {key}")
            if key in rows:
                raise ValueError(f"Duplicate manual audit row: {key}")
            rows[key] = dict(row)
    return rows


def load_retrieval_noise_audit(
    path: Optional[Path],
    lang: str,
) -> Dict[Tuple[int, str, str], Dict[str, str]]:
    if path is None:
        return {}
    rows: Dict[Tuple[int, str, str], Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "lang",
            "sentence_index",
            "term",
            "translation",
            "audit_label",
            "audit_note",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Retrieval-noise audit must contain columns {sorted(required)}")
        for row in reader:
            if row["lang"] != lang:
                continue
            key = (int(row["sentence_index"]), row["term"], row["translation"])
            label = row["audit_label"]
            if label not in NOISE_AUDIT_LABELS:
                raise ValueError(f"Unsupported retrieval-noise audit label {label!r} for {key}")
            if key in rows:
                raise ValueError(f"Duplicate retrieval-noise audit row: {key}")
            rows[key] = dict(row)
    return rows


def summarize_values(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    values = [float(row["xcomet_delta"]) for row in rows]
    return {
        "sentence_count": len(rows),
        "mean_xcomet_delta": statistics.mean(values) if values else None,
        "median_xcomet_delta": statistics.median(values) if values else None,
        "negative_sentence_count": sum(value < 0 for value in values),
        "negative_sentence_fraction": (
            sum(value < 0 for value in values) / len(values) if values else None
        ),
        "xcomet_delta_sum": sum(values),
    }


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=list(fields),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    *,
    term_adoption_path: Path,
    runtime_log_path: Path,
    audio_yaml_path: Path,
    xcomet_segments_path: Path,
    dataset: str,
    lang: str,
    lm: int,
    output_dir: Path,
    rasst_method: str = "RASST",
    baseline_method: str = "InfiniSST",
    manual_audit_path: Optional[Path] = None,
    retrieval_noise_audit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    adoption = json.loads(term_adoption_path.read_text(encoding="utf-8"))
    sentence_rows_raw = adoption.get("sentences")
    if not isinstance(sentence_rows_raw, list):
        raise ValueError(f"term_adoption has no sentences list: {term_adoption_path}")
    audio_sentences = load_audio_sentences(audio_yaml_path)
    if len(audio_sentences) != len(sentence_rows_raw):
        raise ValueError(
            f"audio sentence count {len(audio_sentences)} != adoption sentence count "
            f"{len(sentence_rows_raw)}"
        )
    events_by_paper = load_retrieval_events(runtime_log_path, audio_sentences)
    xcomet_pairs = load_xcomet_pairs(
        xcomet_segments_path,
        dataset,
        lang,
        lm,
        rasst_method,
        baseline_method,
    )
    if len(xcomet_pairs) != len(sentence_rows_raw):
        raise ValueError(
            f"xCOMET sentence count {len(xcomet_pairs)} != adoption sentence count "
            f"{len(sentence_rows_raw)}"
        )
    manual_audit = load_manual_audit(manual_audit_path)
    retrieval_noise_audit = load_retrieval_noise_audit(retrieval_noise_audit_path, lang)

    occurrences: List[Dict[str, Any]] = []
    sentence_rows: List[Dict[str, Any]] = []
    seen_audit_keys = set()
    for raw_sentence, audio in zip(sentence_rows_raw, audio_sentences):
        index = int(raw_sentence["index"])
        if index != audio.index:
            raise ValueError(f"sentence index mismatch: adoption={index} audio={audio.index}")
        pair = xcomet_pairs[index]
        rasst_xcomet = pair[rasst_method]
        baseline_xcomet = pair[baseline_method]
        source = str(raw_sentence.get("source") or "")
        reference = str(raw_sentence.get("reference") or "")
        hypothesis = str(raw_sentence.get("hypothesis") or "")
        for method, row in pair.items():
            if normalize_space(row.get("source")) != normalize_space(source):
                raise ValueError(f"source mismatch for sentence={index} method={method}")
            if normalize_space(row.get("reference")) != normalize_space(reference):
                raise ValueError(f"reference mismatch for sentence={index} method={method}")
        if normalize_space(rasst_xcomet.get("hypothesis")) != normalize_space(hypothesis):
            raise ValueError(f"RASST hypothesis mismatch for sentence={index}")

        terms = raw_sentence.get("terms") or []
        if not isinstance(terms, list):
            raise ValueError(f"Invalid terms for sentence {index}")
        baseline_hypothesis = str(baseline_xcomet.get("hypothesis") or "")
        rasst_correct_count = 0
        baseline_correct_count = 0
        for term in terms:
            if not isinstance(term, dict):
                raise ValueError(f"Invalid term entry for sentence {index}")
            exact_correct = bool(term.get("adopted"))
            baseline_correct = text_contains_exact(baseline_hypothesis, str(term["translation"]))
            rasst_correct_count += int(exact_correct)
            baseline_correct_count += int(baseline_correct)
            timing, prompt_end, evidence_end, retrieval_segment = retrieval_timing(
                term,
                audio,
                events_by_paper[audio.paper_id],
            )
            failure_stage = "exact_correct"
            if not exact_correct:
                if timing == "never_retrieved":
                    failure_stage = "retriever_miss"
                elif timing == "retrieved_late":
                    failure_stage = "retrieved_late_not_exact"
                else:
                    failure_stage = "retrieved_on_time_not_exact"
            candidate = (
                morphology_candidate(str(term["translation"]), hypothesis)
                if lang == "de" and not exact_correct
                else {
                    "candidate_kind": "not_applicable",
                    "candidate_span": "",
                    "candidate_score": "",
                }
            )
            audit_key = (index, str(term["term"]), str(term["translation"]))
            audit = manual_audit.get(audit_key, {})
            if audit:
                seen_audit_keys.add(audit_key)
            occurrence = {
                "dataset": dataset,
                "lang": lang,
                "lm": lm,
                "sentence_index": index,
                "paper_id": audio.paper_id,
                "source_start_sec": audio.start_sec,
                "source_end_sec": audio.end_sec,
                "term": term["term"],
                "translation": term["translation"],
                "exact_correct": exact_correct,
                "baseline_exact_correct": baseline_correct,
                "retrieval_status": timing,
                "failure_stage": failure_stage,
                "first_prompt_end_sec": prompt_end if prompt_end is not None else "",
                "retrieval_evidence_end_sec": evidence_end if evidence_end is not None else "",
                "retrieval_segment_idx": retrieval_segment if retrieval_segment is not None else "",
                **candidate,
                "audit_label": audit.get("audit_label", ""),
                "audit_note": audit.get("audit_note", ""),
                "source": source,
                "reference": reference,
                "rasst_hypothesis": hypothesis,
                "baseline_hypothesis": baseline_hypothesis,
                "rasst_xcomet": float(rasst_xcomet["xcomet_score"]),
                "baseline_xcomet": float(baseline_xcomet["xcomet_score"]),
                "xcomet_delta": float(rasst_xcomet["xcomet_score"])
                - float(baseline_xcomet["xcomet_score"]),
            }
            occurrences.append(occurrence)

        noise_terms = raw_sentence.get("term_map_false_copy_terms") or []
        negative_terms = raw_sentence.get("term_map_negative_terms") or []
        error_spans = rasst_xcomet.get("error_spans") or []
        severity_counts = Counter(
            str(span.get("severity") or "unknown")
            for span in error_spans
            if isinstance(span, dict)
        )
        sentence_rows.append(
            {
                "dataset": dataset,
                "lang": lang,
                "lm": lm,
                "sentence_index": index,
                "paper_id": audio.paper_id,
                "gold_term_count": len(terms),
                "rasst_exact_count": rasst_correct_count,
                "baseline_exact_count": baseline_correct_count,
                "net_exact_term_gain": rasst_correct_count - baseline_correct_count,
                "term_comparison": (
                    "term_gain"
                    if rasst_correct_count > baseline_correct_count
                    else "term_loss"
                    if rasst_correct_count < baseline_correct_count
                    else "term_tie"
                ),
                "raw_unsupported_hint_count": len(negative_terms),
                "raw_false_copy_term_count": len(noise_terms),
                "raw_false_copy_terms_json": json.dumps(noise_terms, ensure_ascii=False),
                "rasst_xcomet": float(rasst_xcomet["xcomet_score"]),
                "baseline_xcomet": float(baseline_xcomet["xcomet_score"]),
                "xcomet_delta": float(rasst_xcomet["xcomet_score"])
                - float(baseline_xcomet["xcomet_score"]),
                "rasst_xcomet_major_spans": severity_counts.get("major", 0),
                "rasst_xcomet_minor_spans": severity_counts.get("minor", 0),
                "source": source,
                "reference": reference,
                "rasst_hypothesis": hypothesis,
                "baseline_hypothesis": baseline_hypothesis,
            }
        )

    unknown_audit = set(manual_audit) - seen_audit_keys
    if unknown_audit:
        raise ValueError(
            f"Manual audit rows do not match exact misses: {sorted(unknown_audit)[:5]}"
        )
    unaudited_candidates = [
        row
        for row in occurrences
        if row["candidate_kind"] not in {"no_candidate", "not_applicable"}
        and not row["audit_label"]
    ]
    raw_false_copy_keys = {
        (int(sentence["index"]), str(term["term"]), str(term["translation"]))
        for sentence in sentence_rows_raw
        for term in (sentence.get("term_map_false_copy_terms") or [])
    }
    unknown_noise_audit = set(retrieval_noise_audit) - raw_false_copy_keys
    if unknown_noise_audit:
        raise ValueError(
            "Retrieval-noise audit rows do not match raw false-copy flags: "
            f"{sorted(unknown_noise_audit)[:5]}"
        )
    missing_noise_audit = raw_false_copy_keys - set(retrieval_noise_audit)
    noise_label_counts = Counter(
        row["audit_label"] for row in retrieval_noise_audit.values()
    )
    sentence_by_index = {int(row["sentence_index"]): row for row in sentence_rows}
    noise_quality_groups: Dict[str, Any] = {}
    for label in sorted(NOISE_AUDIT_LABELS):
        indexes = {
            key[0]
            for key, row in retrieval_noise_audit.items()
            if row["audit_label"] == label
        }
        noise_quality_groups[label] = summarize_values(
            [sentence_by_index[index] for index in sorted(indexes)]
        )
    unsupported_adoption_indexes = {
        key[0]
        for key, row in retrieval_noise_audit.items()
        if row["audit_label"]
        in {"harmful_unsupported_adoption", "benign_unsupported_adoption"}
    }
    harmful_adoption_indexes = {
        key[0]
        for key, row in retrieval_noise_audit.items()
        if row["audit_label"] == "harmful_unsupported_adoption"
    }
    noise_quality_groups["unsupported_adoption_any"] = summarize_values(
        [sentence_by_index[index] for index in sorted(unsupported_adoption_indexes)]
    )
    noise_quality_groups["harmful_unsupported_adoption"] = summarize_values(
        [sentence_by_index[index] for index in sorted(harmful_adoption_indexes)]
    )

    timing_summary: Dict[str, Any] = {}
    for status in ("retrieved_on_time", "retrieved_late", "never_retrieved"):
        rows = [row for row in occurrences if row["retrieval_status"] == status]
        correct = sum(bool(row["exact_correct"]) for row in rows)
        timing_summary[status] = {
            "occurrence_count": len(rows),
            "exact_correct_count": correct,
            "p_exact_correct": correct / len(rows) if rows else None,
        }

    exact_total = len(occurrences)
    exact_correct = sum(bool(row["exact_correct"]) for row in occurrences)
    audit_counts = Counter(str(row["audit_label"]) for row in occurrences if row["audit_label"])
    morphology_additions = sum(
        row["audit_label"] in MORPHOLOGY_AWARE_LABELS for row in occurrences
    )
    semantic_additions = sum(
        row["audit_label"] in SEMANTICALLY_VALID_LABELS for row in occurrences
    )
    failure_counts = Counter(
        str(row["failure_stage"]) for row in occurrences if not row["exact_correct"]
    )
    audited_failure_counts: Counter[str] = Counter()
    for row in occurrences:
        if row["exact_correct"]:
            continue
        label = str(row["audit_label"])
        if label in SEMANTICALLY_VALID_LABELS:
            audited_failure_counts["metric_false_negative"] += 1
        elif not label:
            audited_failure_counts["unaudited_exact_miss"] += 1
        elif row["retrieval_status"] == "never_retrieved":
            audited_failure_counts["retriever_miss"] += 1
        elif row["retrieval_status"] == "retrieved_late":
            audited_failure_counts["retrieved_late"] += 1
        elif label == "omitted_term":
            audited_failure_counts["retrieved_on_time_but_unused"] += 1
        elif label == "wrong_translation":
            audited_failure_counts["retrieved_on_time_wrong_translation"] += 1
        else:
            audited_failure_counts["uncertain"] += 1

    quality_groups = {
        "all": sentence_rows,
        "raw_false_copy_flag": [
            row for row in sentence_rows if int(row["raw_false_copy_term_count"]) > 0
        ],
        "no_raw_false_copy_flag": [
            row for row in sentence_rows if int(row["raw_false_copy_term_count"]) == 0
        ],
        "raw_unsupported_hint_exposed": [
            row for row in sentence_rows if int(row["raw_unsupported_hint_count"]) > 0
        ],
        "no_raw_unsupported_hint": [
            row for row in sentence_rows if int(row["raw_unsupported_hint_count"]) == 0
        ],
        "net_term_gain": [row for row in sentence_rows if row["term_comparison"] == "term_gain"],
        "term_tie": [row for row in sentence_rows if row["term_comparison"] == "term_tie"],
        "net_term_loss": [row for row in sentence_rows if row["term_comparison"] == "term_loss"],
    }
    quality_summary = {name: summarize_values(rows) for name, rows in quality_groups.items()}

    input_paths = {
        "term_adoption": term_adoption_path,
        "runtime_log": runtime_log_path,
        "audio_yaml": audio_yaml_path,
        "xcomet_segments": xcomet_segments_path,
    }
    if manual_audit_path is not None:
        input_paths["manual_audit"] = manual_audit_path
    if retrieval_noise_audit_path is not None:
        input_paths["retrieval_noise_audit"] = retrieval_noise_audit_path
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "lang": lang,
        "lm": lm,
        "methods": {"rasst": rasst_method, "baseline": baseline_method},
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in input_paths.items()
        },
        "definitions": {
            "gold_occurrence": (
                "A glossary pair whose English source and fixed target translation both occur "
                "in the aligned source/reference sentence, matching term_adoption.json."
            ),
            "retrieved_on_time": (
                "The first LLM prompt containing the exact target hint has timestamped acoustic "
                "evidence overlapping the gold sentence and is issued no later than source "
                "sentence end."
            ),
            "retrieved_late": (
                "The same evidence-overlap condition holds, but the first prompt containing the "
                "hint is issued after source sentence end."
            ),
            "never_retrieved": (
                "No prompt contains the exact target hint with timestamped acoustic evidence "
                "overlapping that gold sentence."
            ),
            "raw_false_copy_flag": (
                "The existing exact-match diagnostic flags a retrieved target hint as unsupported "
                "by the aligned English source/reference but present in the RASST hypothesis. "
                "This is a candidate, not confirmed noise: morphology and streaming sentence "
                "boundaries can create false positives."
            ),
        },
        "sentence_count": len(sentence_rows),
        "gold_occurrence_count": exact_total,
        "exact_correct_count": exact_correct,
        "exact_term_accuracy": exact_correct / exact_total if exact_total else None,
        "failure_chain": {
            "runtime_glossary_missing": {
                "occurrence_count": 0,
                "note": (
                    "Zero by construction for this canonical raw-gold-glossary run; this stage "
                    "becomes non-zero only in the realistic paper-derived glossary experiment."
                ),
            },
            **{
                name: {"occurrence_count": failure_counts.get(name, 0)}
                for name in (
                    "retriever_miss",
                    "retrieved_late_not_exact",
                    "retrieved_on_time_not_exact",
                )
            },
        },
        "audited_failure_chain": {
            "note": (
                "Hierarchical attribution over exact misses: validated semantic metric false "
                "negatives first, then retrieval timing, then on-time generation behavior."
            ),
            **{
                name: {"occurrence_count": audited_failure_counts.get(name, 0)}
                for name in (
                    "metric_false_negative",
                    "retriever_miss",
                    "retrieved_late",
                    "retrieved_on_time_but_unused",
                    "retrieved_on_time_wrong_translation",
                    "uncertain",
                    "unaudited_exact_miss",
                )
            },
        },
        "retrieval_conditionals": timing_summary,
        "quality_by_sentence_group": quality_summary,
        "raw_false_copy_term_count": sum(
            int(row["raw_false_copy_term_count"]) for row in sentence_rows
        ),
        "retrieval_noise_audit": {
            "raw_flagged_term_count": len(raw_false_copy_keys),
            "audited_term_count": len(retrieval_noise_audit),
            "all_raw_flags_audited": not missing_noise_audit,
            "missing_audit_count": len(missing_noise_audit),
            "label_counts": dict(sorted(noise_label_counts.items())),
            "quality_by_audit_label": noise_quality_groups,
        },
        "manual_audit": {
            "label_counts": dict(sorted(audit_counts.items())),
            "exact_miss_count": exact_total - exact_correct,
            "audited_exact_miss_count": sum(
                bool(row["audit_label"])
                for row in occurrences
                if not row["exact_correct"]
            ),
            "all_exact_misses_audited": all(
                bool(row["audit_label"])
                for row in occurrences
                if not row["exact_correct"]
            ),
            "candidate_count": sum(
                row["candidate_kind"] not in {"no_candidate", "not_applicable"}
                for row in occurrences
            ),
            "unaudited_candidate_count": len(unaudited_candidates),
            "morphology_aware_added_count": morphology_additions,
            "morphology_aware_correct_count": exact_correct + morphology_additions,
            "morphology_aware_accuracy": (
                (exact_correct + morphology_additions) / exact_total if exact_total else None
            ),
            "semantic_diagnostic_added_count": semantic_additions,
            "semantic_diagnostic_correct_count": exact_correct + semantic_additions,
            "semantic_diagnostic_accuracy": (
                (exact_correct + semantic_additions) / exact_total if exact_total else None
            ),
        },
    }

    occurrence_fields = [
        "dataset",
        "lang",
        "lm",
        "sentence_index",
        "paper_id",
        "source_start_sec",
        "source_end_sec",
        "term",
        "translation",
        "exact_correct",
        "baseline_exact_correct",
        "retrieval_status",
        "failure_stage",
        "first_prompt_end_sec",
        "retrieval_evidence_end_sec",
        "retrieval_segment_idx",
        "candidate_kind",
        "candidate_span",
        "candidate_score",
        "audit_label",
        "audit_note",
        "rasst_xcomet",
        "baseline_xcomet",
        "xcomet_delta",
        "source",
        "reference",
        "rasst_hypothesis",
        "baseline_hypothesis",
    ]
    sentence_fields = [
        "dataset",
        "lang",
        "lm",
        "sentence_index",
        "paper_id",
        "gold_term_count",
        "rasst_exact_count",
        "baseline_exact_count",
        "net_exact_term_gain",
        "term_comparison",
        "raw_unsupported_hint_count",
        "raw_false_copy_term_count",
        "raw_false_copy_terms_json",
        "rasst_xcomet",
        "baseline_xcomet",
        "xcomet_delta",
        "rasst_xcomet_major_spans",
        "rasst_xcomet_minor_spans",
        "source",
        "reference",
        "rasst_hypothesis",
        "baseline_hypothesis",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "occurrences.tsv", occurrences, occurrence_fields)
    write_tsv(output_dir / "sentences.tsv", sentence_rows, sentence_fields)
    if lang == "de":
        write_tsv(
            output_dir / "german_exact_miss_candidates.tsv",
            [row for row in occurrences if not row["exact_correct"]],
            occurrence_fields,
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--term-adoption", required=True)
    parser.add_argument("--runtime-log", required=True)
    parser.add_argument("--audio-yaml", required=True)
    parser.add_argument("--xcomet-segments", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--lang", required=True, choices=("de", "zh"))
    parser.add_argument("--lm", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rasst-method", default="RASST")
    parser.add_argument("--baseline-method", default="InfiniSST")
    parser.add_argument("--manual-audit", default="")
    parser.add_argument("--retrieval-noise-audit", default="")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = analyze(
        term_adoption_path=Path(args.term_adoption),
        runtime_log_path=Path(args.runtime_log),
        audio_yaml_path=Path(args.audio_yaml),
        xcomet_segments_path=Path(args.xcomet_segments),
        dataset=args.dataset,
        lang=args.lang,
        lm=args.lm,
        output_dir=Path(args.output_dir),
        rasst_method=args.rasst_method,
        baseline_method=args.baseline_method,
        manual_audit_path=Path(args.manual_audit) if args.manual_audit else None,
        retrieval_noise_audit_path=(
            Path(args.retrieval_noise_audit) if args.retrieval_noise_audit else None
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
