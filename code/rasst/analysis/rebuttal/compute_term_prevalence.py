#!/usr/bin/env python3
"""Measure how sparse gold terminology is in source and target references."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


Tokenize = Callable[[str], List[str]]


@dataclass(frozen=True)
class GlossaryTerm:
    source: str
    target: str


@dataclass(frozen=True)
class PrevalenceResult:
    dataset: str
    lang: str
    tokenizer: str
    sentences: int
    source_tokens: int
    source_glossary_term_tokens: int
    source_glossary_term_share_pct: float
    source_glossary_term_sentences: int
    source_glossary_term_sentence_share_pct: float
    target_tokens: int
    aligned_gold_term_tokens: int
    aligned_gold_term_share_pct: float
    aligned_gold_term_sentences: int
    aligned_gold_term_sentence_share_pct: float
    aligned_term_pair_occurrences: int
    matched_unique_term_pairs: int
    glossary_term_pairs: int
    source_sha256: str
    reference_sha256: str
    glossary_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_space(text: Any) -> str:
    return " ".join(str(text or "").split())


def load_glossary(path: Path, lang: str) -> List[GlossaryTerm]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries: Iterable[Any]
    if isinstance(data, dict):
        entries = data.values()
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(f"Unsupported glossary root in {path}: {type(data).__name__}")

    terms: List[GlossaryTerm] = []
    seen: set[Tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source = _normalise_space(entry.get("term") or entry.get("source"))
        translations = entry.get("target_translations")
        target = ""
        if isinstance(translations, dict):
            target = _normalise_space(translations.get(lang))
        if not target:
            target = _normalise_space(
                entry.get("translation")
                or entry.get("target_translation")
                or entry.get(lang)
            )
        key = (source.casefold(), target.casefold())
        if not source or not target or key in seen:
            continue
        seen.add(key)
        terms.append(GlossaryTerm(source=source, target=target))
    if not terms:
        raise ValueError(f"No {lang} term pairs loaded from {path}")
    return terms


def build_sacrebleu_tokenizer(name: str) -> Tokenize:
    try:
        from sacrebleu.metrics import BLEU
    except ImportError as exc:
        raise RuntimeError("sacrebleu is required to reproduce BLEU tokenization") from exc
    tokenizer = BLEU(tokenize=name).tokenizer
    return lambda text: tokenizer(str(text or "")).split()


def find_token_spans(
    tokens: Sequence[str],
    phrase_tokens: Sequence[str],
    *,
    allow_single_token_substring: bool = False,
) -> List[Tuple[int, int]]:
    if not tokens or not phrase_tokens or len(phrase_tokens) > len(tokens):
        return []
    folded = [token.casefold() for token in tokens]
    phrase = [token.casefold() for token in phrase_tokens]
    width = len(phrase)
    spans = [
        (start, start + width)
        for start in range(len(folded) - width + 1)
        if folded[start : start + width] == phrase
    ]
    if spans or not allow_single_token_substring or width != 1:
        return spans
    return [
        (index, index + 1)
        for index, token in enumerate(folded)
        if phrase[0] in token
    ]


def source_contains(text: str, term: str) -> bool:
    text_norm = _normalise_space(text).casefold()
    term_norm = _normalise_space(term).casefold()
    if not text_norm or not term_norm:
        return False
    if re.fullmatch(r"[a-z0-9][a-z0-9 ._+/#-]*", term_norm):
        pattern = r"(?<![a-z0-9])" + re.escape(term_norm) + r"(?![a-z0-9])"
        return re.search(pattern, text_norm) is not None
    return term_norm in text_norm


def target_contains(text: str, translation: str) -> bool:
    text_norm = _normalise_space(text)
    translation_norm = _normalise_space(translation)
    return bool(translation_norm) and translation_norm in text_norm


def _mark_spans(mask: List[bool], spans: Sequence[Tuple[int, int]]) -> None:
    for start, end in spans:
        for index in range(start, end):
            mask[index] = True


def compute_prevalence(
    *,
    dataset: str,
    lang: str,
    tokenizer_name: str,
    source_path: Path,
    reference_path: Path,
    glossary_path: Path,
    tokenize_source: Tokenize,
    tokenize_target: Tokenize,
) -> PrevalenceResult:
    sources = source_path.read_text(encoding="utf-8").splitlines()
    references = reference_path.read_text(encoding="utf-8").splitlines()
    if len(sources) != len(references):
        raise ValueError(
            f"Source/reference line mismatch: {len(sources)} != {len(references)}"
        )
    terms = load_glossary(glossary_path, lang)
    source_term_tokens = {term: tokenize_source(term.source) for term in terms}
    target_term_tokens = {term: tokenize_target(term.target) for term in terms}

    source_token_total = 0
    source_term_token_total = 0
    source_term_sentences = 0
    target_token_total = 0
    aligned_term_token_total = 0
    aligned_term_sentences = 0
    aligned_occurrences = 0
    matched_pairs: set[Tuple[str, str]] = set()

    for source, reference in zip(sources, references):
        source_tokens = tokenize_source(source)
        target_tokens = tokenize_target(reference)
        source_token_total += len(source_tokens)
        target_token_total += len(target_tokens)
        source_mask = [False] * len(source_tokens)
        target_mask = [False] * len(target_tokens)
        sentence_has_source_term = False
        sentence_has_aligned_term = False

        for term in terms:
            if not source_contains(source, term.source):
                continue
            source_spans = find_token_spans(source_tokens, source_term_tokens[term])
            sentence_has_source_term = True
            _mark_spans(source_mask, source_spans)
            if not target_contains(reference, term.target):
                continue
            target_spans = find_token_spans(
                target_tokens,
                target_term_tokens[term],
                allow_single_token_substring=True,
            )
            sentence_has_aligned_term = True
            _mark_spans(target_mask, target_spans)
            aligned_occurrences += 1
            matched_pairs.add((term.source.casefold(), term.target.casefold()))

        source_term_token_total += sum(source_mask)
        aligned_term_token_total += sum(target_mask)
        source_term_sentences += int(sentence_has_source_term)
        aligned_term_sentences += int(sentence_has_aligned_term)

    sentence_total = len(sources)
    return PrevalenceResult(
        dataset=dataset,
        lang=lang,
        tokenizer=tokenizer_name,
        sentences=sentence_total,
        source_tokens=source_token_total,
        source_glossary_term_tokens=source_term_token_total,
        source_glossary_term_share_pct=(100.0 * source_term_token_total / source_token_total),
        source_glossary_term_sentences=source_term_sentences,
        source_glossary_term_sentence_share_pct=(100.0 * source_term_sentences / sentence_total),
        target_tokens=target_token_total,
        aligned_gold_term_tokens=aligned_term_token_total,
        aligned_gold_term_share_pct=(100.0 * aligned_term_token_total / target_token_total),
        aligned_gold_term_sentences=aligned_term_sentences,
        aligned_gold_term_sentence_share_pct=(100.0 * aligned_term_sentences / sentence_total),
        aligned_term_pair_occurrences=aligned_occurrences,
        matched_unique_term_pairs=len(matched_pairs),
        glossary_term_pairs=len(terms),
        source_sha256=_sha256(source_path),
        reference_sha256=_sha256(reference_path),
        glossary_sha256=_sha256(glossary_path),
    )


def _write_tsv(path: Path, result: PrevalenceResult) -> None:
    row: Dict[str, Any] = asdict(result)
    row = {
        key: f"{value:.6f}" if isinstance(value, float) else value
        for key, value in row.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--glossary", required=True, type=Path)
    parser.add_argument("--target-tokenizer", required=True, choices=["13a", "zh", "ja-mecab"])
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-tsv", required=True, type=Path)
    args = parser.parse_args()

    result = compute_prevalence(
        dataset=args.dataset,
        lang=args.lang,
        tokenizer_name=args.target_tokenizer,
        source_path=args.source,
        reference_path=args.reference,
        glossary_path=args.glossary,
        tokenize_source=build_sacrebleu_tokenizer("13a"),
        tokenize_target=build_sacrebleu_tokenizer(args.target_tokenizer),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_tsv(args.output_tsv, result)
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
