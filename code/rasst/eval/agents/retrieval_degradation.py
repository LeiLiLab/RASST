"""Deterministic in-domain corruption for retrieved terminology hints."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PLAN_SCHEMA_VERSION = 1


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def source_contains(source_text: str, term: str) -> bool:
    source_norm = normalize_space(source_text).casefold()
    term_norm = normalize_space(term).casefold()
    if not source_norm or not term_norm:
        return False
    if re.fullmatch(r"[a-z0-9][a-z0-9 ._+/#-]*", term_norm):
        pattern = r"(?<![a-z0-9])" + re.escape(term_norm) + r"(?![a-z0-9])"
        return re.search(pattern, source_norm) is not None
    return term_norm in source_norm


def reference_key(reference: Mapping[str, Any]) -> Tuple[str, str]:
    term = normalize_space(reference.get("term") or reference.get("key") or "")
    translation = normalize_space(reference.get("translation") or "")
    return term.casefold(), translation


def load_glossary(path: Path, target_lang: str) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries: Iterable[Any] = data.values() if isinstance(data, dict) else data
    if not isinstance(data, (dict, list)):
        raise ValueError(f"Unsupported glossary format: {path}")

    references: List[Dict[str, str]] = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        term = normalize_space(entry.get("term") or entry.get("source") or "")
        translations = entry.get("target_translations")
        translation = ""
        if isinstance(translations, dict):
            translation = normalize_space(translations.get(target_lang) or "")
        if not translation:
            translation = normalize_space(
                entry.get("translation")
                or entry.get("target_translation")
                or entry.get(target_lang)
                or ""
            )
        key = (term.casefold(), translation)
        if term and translation and key not in seen:
            seen.add(key)
            references.append({"term": term, "translation": translation})
    if not references:
        raise ValueError(f"No {target_lang} glossary entries found in {path}")
    return references


def _stable_uint64(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class RetrievalDegrader:
    """Replace relevant retrieved hints while preserving prompt cardinality."""

    def __init__(self, plan_path: str, rate: float, seed: int) -> None:
        if not 0.0 <= float(rate) <= 1.0:
            raise ValueError(f"retrieval degradation rate must be in [0, 1], got {rate}")
        self.plan_path = str(Path(plan_path).expanduser().resolve())
        self.rate = float(rate)
        self.seed = int(seed)
        self.plan = json.loads(Path(self.plan_path).read_text(encoding="utf-8"))
        if int(self.plan.get("schema_version", -1)) != PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported retrieval degradation plan schema: {self.plan.get('schema_version')}"
            )
        self.instances = list(self.plan.get("instances") or [])
        self.distractors = list(self.plan.get("glossary") or [])
        if not self.instances or not self.distractors:
            raise ValueError(f"Incomplete retrieval degradation plan: {self.plan_path}")

    def _relevant_references(
        self,
        instance_index: int,
        window_start_sec: float,
        window_end_sec: float,
    ) -> List[Dict[str, str]]:
        if instance_index < 0 or instance_index >= len(self.instances):
            raise IndexError(
                f"instance_index={instance_index} outside plan with {len(self.instances)} instances"
            )
        relevant: Dict[Tuple[str, str], Dict[str, str]] = {}
        for sentence in self.instances[instance_index].get("sentences") or []:
            start = float(sentence["start_sec"])
            end = float(sentence["end_sec"])
            if window_start_sec < end and start < window_end_sec:
                for ref in sentence.get("references") or []:
                    relevant[reference_key(ref)] = {
                        "term": normalize_space(ref.get("term")),
                        "translation": normalize_space(ref.get("translation")),
                    }
        return list(relevant.values())

    def degrade(
        self,
        references: Sequence[Mapping[str, Any]],
        *,
        instance_index: int,
        segment_idx: int,
        current_start_sec: float,
        current_end_sec: float,
        lookback_sec: float,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        original = [dict(ref) for ref in references]
        window_start_sec = max(0.0, float(current_start_sec) - float(lookback_sec))
        window_end_sec = float(current_end_sec)
        relevant_refs = self._relevant_references(
            instance_index, window_start_sec, window_end_sec
        )
        relevant_keys = {reference_key(ref) for ref in relevant_refs}
        original_keys = [reference_key(ref) for ref in original]
        occupied_terms = {key[0] for key in original_keys}
        candidate_pool = [
            dict(ref)
            for ref in self.distractors
            if reference_key(ref) not in relevant_keys
            and reference_key(ref)[0] not in occupied_terms
        ]
        if self.rate > 0.0 and relevant_keys and not candidate_pool:
            raise ValueError("No in-domain distractors remain after relevance filtering")

        degraded: List[Dict[str, Any]] = []
        replacements: List[Dict[str, Any]] = []
        available_pool = list(candidate_pool)
        relevant_original = 0
        for rank, (reference, key) in enumerate(zip(original, original_keys)):
            is_relevant = key in relevant_keys
            relevant_original += int(is_relevant)
            draw = _stable_uint64(
                self.seed,
                instance_index,
                segment_idx,
                rank,
                key[0],
                key[1],
                "replace",
            ) / float(2**64)
            if not is_relevant or draw >= self.rate:
                degraded.append(reference)
                continue

            if not available_pool:
                raise ValueError("Not enough unique in-domain distractors for this prompt")

            pool_index = _stable_uint64(
                self.seed,
                instance_index,
                segment_idx,
                rank,
                key[0],
                "distractor",
            ) % len(available_pool)
            distractor = available_pool.pop(pool_index)
            replacement = dict(reference)
            replacement["term"] = distractor["term"]
            replacement["translation"] = distractor["translation"]
            replacement["key"] = distractor["term"]
            replacement["retrieval_degradation_original"] = {
                "term": normalize_space(reference.get("term") or reference.get("key")),
                "translation": normalize_space(reference.get("translation")),
            }
            replacement["retrieval_degradation_distractor"] = True
            degraded.append(replacement)
            replacements.append(
                {
                    "rank": rank,
                    "original": replacement["retrieval_degradation_original"],
                    "distractor": distractor,
                }
            )

        final_keys = [reference_key(ref) for ref in degraded]
        relevant_final = sum(key in relevant_keys for key in final_keys)
        gold_count = len(relevant_keys)

        def ratio(numerator: int, denominator: int) -> float:
            return float(numerator) / float(denominator) if denominator else 0.0

        audit = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "plan_path": self.plan_path,
            "configured_rate": self.rate,
            "seed": self.seed,
            "instance_index": int(instance_index),
            "segment_idx": int(segment_idx),
            "current_start_sec": round(float(current_start_sec), 6),
            "current_end_sec": round(float(current_end_sec), 6),
            "lookback_sec": round(float(lookback_sec), 6),
            "relevance_window_start_sec": round(window_start_sec, 6),
            "relevance_window_end_sec": round(window_end_sec, 6),
            "hint_count_original": len(original),
            "hint_count_final": len(degraded),
            "relevant_gold_count": gold_count,
            "relevant_hint_count_original": relevant_original,
            "relevant_hint_count_final": relevant_final,
            "replaced_relevant_hint_count": len(replacements),
            "achieved_replacement_rate": ratio(len(replacements), relevant_original),
            "retrieval_precision_original": ratio(relevant_original, len(original)),
            "retrieval_precision_final": ratio(relevant_final, len(degraded)),
            "retrieval_recall_original": ratio(relevant_original, gold_count),
            "retrieval_recall_final": ratio(relevant_final, gold_count),
            "replacements": replacements,
        }
        if len(original) != len(degraded):
            raise AssertionError("retrieval degradation changed hint count")
        return degraded, audit
