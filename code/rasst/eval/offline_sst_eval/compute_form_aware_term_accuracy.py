#!/usr/bin/env python3
"""Compute exact-form and conservative form-aware terminology accuracy.

The denominator and exact-form match reproduce ``stream_laal_term.py``: target
translations are deduplicated, a term is eligible only when its source form
occurs in the aligned source sentence and its target form occurs in the
reference, and each eligible target form is counted once per sentence.

The secondary form-aware diagnostic accepts Unicode/case/spacing/hyphenation
variants in every language and, when requested, contiguous German lemma
sequences produced by a pinned spaCy pipeline. It does not accept synonyms or
semantic paraphrases.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "rasst_form_aware_term_accuracy_v1"
TERM_TAG_RE = re.compile(r"</?\s*(?:term|t)\s*>", flags=re.IGNORECASE)


@dataclass(frozen=True)
class GlossaryTerm:
    source: str
    target: str


@dataclass(frozen=True)
class AlignedSentence:
    index: int
    wav: str
    source: str
    reference: str
    hypothesis: str


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    kind: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_surface(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return normalize_space(text)


def compact_surface(value: Any) -> str:
    return "".join(
        character
        for character in normalize_surface(value)
        if unicodedata.category(character)[0] in {"L", "N"}
    )


def source_contains(source_text: str, source_term: str) -> bool:
    source_norm = normalize_space(source_text).casefold()
    term_norm = normalize_space(source_term).casefold()
    if not source_norm or not term_norm:
        return False
    if re.fullmatch(r"[a-z0-9][a-z0-9 ._+/#-]*", term_norm):
        pattern = r"(?<![a-z0-9])" + re.escape(term_norm) + r"(?![a-z0-9])"
        return re.search(pattern, source_norm) is not None
    return term_norm in source_norm


def _is_latin_or_digit(text: str) -> bool:
    compact = compact_surface(text)
    return bool(compact) and all(
        character.isdigit() or "LATIN" in unicodedata.name(character, "")
        for character in compact
    )


def orthographic_match(hypothesis: str, target: str) -> MatchResult:
    target_norm = normalize_surface(target)
    hypothesis_norm = normalize_surface(hypothesis)
    if not target_norm:
        return MatchResult(False, "none")

    if _is_latin_or_digit(target_norm) and len(compact_surface(target_norm)) < 4:
        pattern = r"(?<!\w)" + re.escape(target_norm) + r"(?!\w)"
        if re.search(pattern, hypothesis_norm, flags=re.UNICODE):
            return MatchResult(True, "casefold_boundary")
    elif target_norm in hypothesis_norm:
        return MatchResult(True, "casefold_or_compound")

    target_compact = compact_surface(target_norm)
    hypothesis_compact = compact_surface(hypothesis_norm)
    min_length = 4 if _is_latin_or_digit(target_norm) else 2
    if len(target_compact) >= min_length and target_compact in hypothesis_compact:
        return MatchResult(True, "spacing_hyphen_or_width")
    return MatchResult(False, "none")


def _content_lemmas(nlp: Callable[[str], Iterable[Any]], text: str) -> List[str]:
    lemmas: List[str] = []
    for token in nlp(text):
        if bool(getattr(token, "is_space", False)) or bool(getattr(token, "is_punct", False)):
            continue
        lemma = normalize_surface(getattr(token, "lemma_", "") or getattr(token, "text", ""))
        if lemma:
            lemmas.append(lemma)
    return lemmas


def lemma_match(
    hypothesis: str,
    target: str,
    nlp: Optional[Callable[[str], Iterable[Any]]],
) -> MatchResult:
    if nlp is None:
        return MatchResult(False, "none")
    target_lemmas = _content_lemmas(nlp, target)
    hypothesis_lemmas = _content_lemmas(nlp, hypothesis)
    if not target_lemmas or len(target_lemmas) > len(hypothesis_lemmas):
        return MatchResult(False, "none")
    width = len(target_lemmas)
    for start in range(len(hypothesis_lemmas) - width + 1):
        if hypothesis_lemmas[start : start + width] == target_lemmas:
            return MatchResult(True, "german_lemma_sequence")
    return MatchResult(False, "none")


def load_spacy_pipeline(model: str) -> Tuple[Callable[[str], Iterable[Any]], Dict[str, Any]]:
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("spaCy is required when --spacy-model is set") from exc
    nlp = spacy.load(model, disable=["parser", "ner"])
    metadata = {
        "spacy_version": spacy.__version__,
        "pipeline": model,
        "pipeline_version": nlp.meta.get("version", ""),
        "pipeline_sources": nlp.meta.get("sources", []),
    }
    return nlp, metadata


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield row


def _coerce_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
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
            raise ValueError(f"Unsupported audio YAML at {path}:{line_number}")
        if ":" not in content:
            raise ValueError(f"Invalid audio YAML field at {path}:{line_number}")
        key, value = content.split(":", 1)
        current[key.strip()] = _coerce_yaml_scalar(value)
    if current is not None:
        rows.append(current)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _glossary_entries(data: Any) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    if isinstance(data, dict):
        for key, entry in data.items():
            if isinstance(entry, dict):
                yield str(key), entry
        return
    if isinstance(data, list):
        for index, entry in enumerate(data):
            if isinstance(entry, dict):
                yield str(index), entry
        return
    raise ValueError("Glossary must be a JSON object or list")


def load_glossary(path: Path, target_lang: str) -> List[GlossaryTerm]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_target: Dict[str, GlossaryTerm] = {}
    for key, entry in _glossary_entries(data):
        translations = entry.get("target_translations")
        target = ""
        if isinstance(translations, dict):
            target = normalize_space(translations.get(target_lang))
        if not target:
            target = normalize_space(
                entry.get("translation")
                or entry.get("target_translation")
                or entry.get(target_lang)
            )
        source = normalize_space(entry.get("term") or entry.get("source") or key)
        if source and target and target not in by_target:
            by_target[target] = GlossaryTerm(source=source, target=target)
    return [by_target[target] for target in sorted(by_target)]


def strip_output_tags(text: str) -> str:
    return TERM_TAG_RE.sub("", str(text or ""))


def _instance_wav(row: Mapping[str, Any]) -> str:
    source = row.get("source")
    if isinstance(source, list) and source:
        return Path(str(source[0])).name
    if isinstance(source, str) and source:
        return Path(source).name
    raise ValueError("SimulEval instance has no source wav")


def _mwer_command() -> str:
    command = shutil.which("mwerSegmenter")
    if command:
        return command
    root = os.environ.get("MWERSEGMENTER_ROOT", "").strip()
    if root:
        candidate = Path(root) / "mwerSegmenter"
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("mwerSegmenter is not in PATH and MWERSEGMENTER_ROOT is not set")


def segment_prediction(prediction: str, references: Sequence[str], latency_unit: str) -> List[str]:
    command = _mwer_command()
    character_level = latency_unit == "char"
    prediction_text = strip_output_tags(prediction)
    reference_texts = [str(reference) for reference in references]
    if character_level:
        prediction_text = " ".join(prediction_text)
        reference_texts = [" ".join(reference) for reference in reference_texts]

    with tempfile.TemporaryDirectory() as temp_dir:
        prediction_file = Path(temp_dir) / "prediction.txt"
        reference_file = Path(temp_dir) / "reference.txt"
        prediction_file.write_text(prediction_text, encoding="utf-8")
        reference_file.write_text(
            "".join(reference + "\n" for reference in reference_texts),
            encoding="utf-8",
        )
        subprocess.run(
            [
                command,
                "-mref",
                str(reference_file),
                "-hypfile",
                str(prediction_file),
                "-usecase",
                "1",
            ],
            cwd=temp_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        segments = (Path(temp_dir) / "__segments").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    if character_level:
        segments = [re.sub(r"(.)\s", r"\1", segment).strip() for segment in segments]
    else:
        segments = [segment.strip() for segment in segments]
    if len(segments) != len(references):
        raise RuntimeError(
            f"mwerSegmenter returned {len(segments)} segments for {len(references)} references"
        )
    return segments


def align_instances(
    instances_log: Path,
    source_file: Path,
    reference_file: Path,
    audio_yaml: Path,
    latency_unit: str,
) -> List[AlignedSentence]:
    instances = {_instance_wav(row): row for row in iter_jsonl(instances_log)}
    sources = source_file.read_text(encoding="utf-8", errors="replace").splitlines()
    references = reference_file.read_text(encoding="utf-8", errors="replace").splitlines()
    audio_rows = load_flat_yaml_list(audio_yaml)
    if not (len(sources) == len(references) == len(audio_rows)):
        raise ValueError(
            "Source/reference/audio row mismatch: "
            f"{len(sources)}/{len(references)}/{len(audio_rows)}"
        )

    indices_by_wav: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(audio_rows):
        wav = Path(str(row.get("wav") or "")).name
        if not wav:
            raise ValueError(f"Audio row {index} has no wav")
        indices_by_wav[wav].append(index)
    if set(instances) != set(indices_by_wav):
        raise ValueError(
            f"Instance/audio wav mismatch: instances={sorted(instances)}, "
            f"audio={sorted(indices_by_wav)}"
        )

    aligned: List[AlignedSentence] = []
    for wav, indices in indices_by_wav.items():
        row = instances[wav]
        local_references = [references[index] for index in indices]
        local_hypotheses = segment_prediction(
            str(row.get("prediction") or ""),
            local_references,
            latency_unit,
        )
        for index, hypothesis in zip(indices, local_hypotheses):
            aligned.append(
                AlignedSentence(
                    index=index,
                    wav=wav,
                    source=sources[index].strip(),
                    reference=references[index].strip(),
                    hypothesis=hypothesis,
                )
            )
    return sorted(aligned, key=lambda sentence: sentence.index)


def score_sentences(
    sentences: Sequence[AlignedSentence],
    glossary: Sequence[GlossaryTerm],
    lang: str,
    nlp: Optional[Callable[[str], Iterable[Any]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    exact_correct = 0
    orthographic_correct = 0
    lemma_correct = 0
    form_correct = 0
    total = 0
    match_kinds: Dict[str, int] = defaultdict(int)
    occurrences: List[Dict[str, Any]] = []

    for sentence in sentences:
        for term in glossary:
            if not source_contains(sentence.source, term.source) or term.target not in sentence.reference:
                continue
            total += 1
            exact = term.target in sentence.hypothesis
            orthographic = MatchResult(False, "none")
            lemma = MatchResult(False, "none")
            if not exact:
                orthographic = orthographic_match(sentence.hypothesis, term.target)
            if not exact and not orthographic.matched and lang == "de":
                lemma = lemma_match(sentence.hypothesis, term.target, nlp)
            form_aware = exact or orthographic.matched or lemma.matched
            kind = "exact" if exact else orthographic.kind if orthographic.matched else lemma.kind
            if not form_aware:
                kind = "miss"
            match_kinds[kind] += 1
            exact_correct += int(exact)
            orthographic_correct += int(exact or orthographic.matched)
            lemma_correct += int(exact or lemma.matched)
            form_correct += int(form_aware)
            occurrences.append(
                {
                    "sentence_index": sentence.index,
                    "wav": sentence.wav,
                    "term": term.source,
                    "translation": term.target,
                    "exact_correct": exact,
                    "orthographic_correct": exact or orthographic.matched,
                    "lemma_correct": exact or lemma.matched,
                    "form_aware_correct": form_aware,
                    "match_kind": kind,
                    "source": sentence.source,
                    "reference": sentence.reference,
                    "hypothesis": sentence.hypothesis,
                }
            )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "language": lang,
        "sentence_count": len(sentences),
        "glossary_target_forms": len(glossary),
        "total_terms": total,
        "exact_correct": exact_correct,
        "exact_accuracy": exact_correct / total if total else 0.0,
        "orthographic_correct": orthographic_correct,
        "orthographic_accuracy": orthographic_correct / total if total else 0.0,
        "lemma_correct": lemma_correct,
        "lemma_accuracy": lemma_correct / total if total else 0.0,
        "form_aware_correct": form_correct,
        "form_aware_accuracy": form_correct / total if total else 0.0,
        "match_kinds": dict(sorted(match_kinds.items())),
    }
    return summary, occurrences


def prefer_stripped_instances(path: Path) -> Path:
    if path.name == "instances.log":
        stripped = path.with_name("instances.strip_term.log")
        if stripped.is_file():
            return stripped
    return path


def load_expected_results(path: Optional[Path]) -> Dict[Tuple[str, str, str, int], Tuple[int, int]]:
    if path is None:
        return {}
    expected: Dict[Tuple[str, str, str, int], Tuple[int, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                lm = int(row["lm"])
                correct = int(row["term_correct"])
                total = int(row["term_total"])
            except (KeyError, TypeError, ValueError):
                continue
            expected[(row["dataset"], row["method"], row["lang"], lm)] = (correct, total)
    return expected


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


def run_manifest(
    manifest: Path,
    output_dir: Path,
    acl_glossary: Path,
    eso_glossary: Path,
    expected_results: Optional[Path],
    spacy_model: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = load_expected_results(expected_results)
    nlp: Optional[Callable[[str], Iterable[Any]]] = None
    dependency_info: Dict[str, Any] = {}
    if spacy_model:
        nlp, dependency_info = load_spacy_pipeline(spacy_model)

    summaries: List[Dict[str, Any]] = []
    occurrence_path = output_dir / "occurrences.jsonl"
    with manifest.open("r", encoding="utf-8", newline="") as handle, occurrence_path.open(
        "w", encoding="utf-8"
    ) as occurrence_handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            dataset = row["dataset"]
            method = row["method"]
            lang = row["lang"]
            lm = int(row["lm"])
            instances = prefer_stripped_instances(Path(row["instances_log"]))
            glossary_path = acl_glossary if dataset == "acl_tagged_raw" else eso_glossary
            glossary = load_glossary(glossary_path, lang)
            sentences = align_instances(
                instances,
                Path(row["source_text"]),
                Path(row["reference"]),
                Path(row["audio_yaml"]),
                row["latency_unit"],
            )
            summary, occurrences = score_sentences(sentences, glossary, lang, nlp=nlp)
            key = (dataset, method, lang, lm)
            expected_pair = expected.get(key)
            if expected_pair is not None and expected_pair != (
                int(summary["exact_correct"]),
                int(summary["total_terms"]),
            ):
                raise ValueError(
                    f"Exact metric mismatch for {key}: computed="
                    f"{summary['exact_correct']}/{summary['total_terms']} expected="
                    f"{expected_pair[0]}/{expected_pair[1]}"
                )
            record = {
                "dataset": dataset,
                "method": method,
                "lang": lang,
                "lm": lm,
                "instances_log": str(instances),
                **summary,
            }
            summaries.append(record)
            for occurrence in occurrences:
                occurrence_handle.write(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "method": method,
                            "lang": lang,
                            "lm": lm,
                            **occurrence,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    summary_fields = [
        "dataset",
        "method",
        "lang",
        "lm",
        "sentence_count",
        "glossary_target_forms",
        "total_terms",
        "exact_correct",
        "exact_accuracy",
        "orthographic_correct",
        "orthographic_accuracy",
        "lemma_correct",
        "lemma_accuracy",
        "form_aware_correct",
        "form_aware_accuracy",
        "match_kinds",
        "instances_log",
    ]
    write_tsv(output_dir / "summary.tsv", summaries, summary_fields)
    run_metadata = {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "acl_glossary": str(acl_glossary),
        "acl_glossary_sha256": sha256_file(acl_glossary),
        "eso_glossary": str(eso_glossary),
        "eso_glossary_sha256": sha256_file(eso_glossary),
        "expected_results": str(expected_results) if expected_results else "",
        "expected_results_sha256": sha256_file(expected_results) if expected_results else "",
        "dependency_info": dependency_info,
        "system_count": len(summaries),
        "occurrence_rows": sum(int(row["total_terms"]) for row in summaries),
        "summary_sha256": sha256_file(output_dir / "summary.tsv"),
        "occurrences_sha256": sha256_file(occurrence_path),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_metadata


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--acl-glossary", required=True)
    parser.add_argument("--eso-glossary", required=True)
    parser.add_argument("--expected-results", default="")
    parser.add_argument("--spacy-model", default="")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_manifest(
        manifest=Path(args.manifest),
        output_dir=Path(args.output_dir),
        acl_glossary=Path(args.acl_glossary),
        eso_glossary=Path(args.eso_glossary),
        expected_results=Path(args.expected_results) if args.expected_results else None,
        spacy_model=args.spacy_model,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
