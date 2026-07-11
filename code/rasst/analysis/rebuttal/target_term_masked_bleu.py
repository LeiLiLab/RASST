#!/usr/bin/env python3
"""Compute BLEU after removing glossary target terms from hypotheses and references."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Pattern, Sequence, Tuple


_ALNUM_TERM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+/#&%()-]*$")
_CJK_OR_KANA_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


@dataclass(frozen=True)
class MaskedBleuResult:
    bleu: float
    hypothesis_terms_removed: int
    reference_terms_removed: int
    term_types: int
    talks: int
    segments: int
    sacrebleu_tokenizer: str


def _normalise_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _basename(path: Any) -> str:
    return str(path or "").replace("\\", "/").rsplit("/", 1)[-1]


def _require_file(path: Path, label: str, *, executable: bool = False) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise PermissionError(f"{label} is not executable: {path}")


def _term_to_mask_regex(term: str) -> Pattern[str]:
    term_norm = _normalise_space(term)
    if not term_norm:
        raise ValueError("Cannot build a mask regex for an empty term")
    escaped = re.escape(term_norm).replace(r"\ ", r"\s+")
    if _ALNUM_TERM_RE.fullmatch(term_norm):
        return re.compile(
            r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])",
            flags=re.IGNORECASE,
        )
    if len(term_norm) == 1 and _CJK_OR_KANA_RE.search(term_norm):
        return re.compile(
            r"(?<![\u3040-\u30ff\u3400-\u9fff])"
            + escaped
            + r"(?![\u3040-\u30ff\u3400-\u9fff])"
        )
    flags = 0 if _CJK_OR_KANA_RE.search(term_norm) else re.IGNORECASE
    return re.compile(escaped, flags=flags)


def _compile_term_mask_patterns(target_terms: Sequence[str]) -> List[Pattern[str]]:
    ordered = sorted(target_terms, key=lambda text: (len(text), text), reverse=True)
    return [_term_to_mask_regex(term) for term in ordered]


def _mask_target_terms(text: str, term_patterns: Sequence[Pattern[str]]) -> Tuple[str, int]:
    masked = str(text or "")
    removed = 0
    for pattern in term_patterns:
        masked, count = pattern.subn(" ", masked)
        removed += count
    return _normalise_space(masked), removed


def _load_target_terms(glossary_path: Path, target_language: str) -> List[str]:
    data = json.loads(glossary_path.read_text(encoding="utf-8"))
    if isinstance(data, Mapping):
        raw_entries: Iterable[Any] = data.values()
    elif isinstance(data, list):
        raw_entries = data
    else:
        raise ValueError(f"Glossary must be a JSON object or list: {glossary_path}")

    terms: List[str] = []
    seen = set()
    for entry in raw_entries:
        if not isinstance(entry, Mapping):
            continue
        translations = entry.get("target_translations")
        translation = ""
        if isinstance(translations, Mapping):
            translation = _normalise_space(translations.get(target_language))
        if not translation:
            translation = _normalise_space(
                entry.get("translation")
                or entry.get("target_translation")
                or entry.get(target_language)
            )
        if not translation:
            continue
        key = translation.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(translation)
    terms.sort(key=lambda text: (len(text), text), reverse=True)
    return terms


def _read_reference_groups(
    reference_path: Path,
    audio_yaml_path: Path,
) -> Dict[str, List[str]]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required; install the pinned release requirements"
        ) from exc
    references = reference_path.read_text(encoding="utf-8").splitlines()
    audio = yaml.safe_load(audio_yaml_path.read_text(encoding="utf-8"))
    if not isinstance(audio, list):
        raise ValueError(f"Audio YAML must contain a list: {audio_yaml_path}")
    if len(audio) != len(references):
        raise ValueError(
            f"Audio/reference length mismatch: audio={len(audio)} "
            f"references={len(references)}"
        )

    grouped: Dict[str, List[str]] = {}
    for index, (item, reference) in enumerate(zip(audio, references), start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Audio row {index} is not an object: {item!r}")
        wav = _basename(item.get("wav"))
        if not wav:
            raise ValueError(f"Audio row {index} has no wav path")
        grouped.setdefault(wav, []).append(reference)
    return grouped


def _read_talk_predictions(instances_log: Path) -> Dict[str, str]:
    predictions: Dict[str, str] = {}
    with instances_log.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {instances_log}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"Instance row {line_number} is not an object")
            source = row.get("source")
            if not isinstance(source, list) or not source:
                raise ValueError(f"Instance row {line_number} has no source[0]")
            if "prediction" not in row:
                raise ValueError(f"Instance row {line_number} has no prediction")
            wav = _basename(source[0])
            if not wav:
                raise ValueError(f"Instance row {line_number} has an empty source[0]")
            if wav in predictions:
                raise ValueError(
                    f"Duplicate talk prediction for {wav} at line {line_number}"
                )
            predictions[wav] = str(row.get("prediction") or "")
    if not predictions:
        raise ValueError(f"No talk predictions found in {instances_log}")
    return predictions


def _segment_talk(
    prediction: str,
    references: Sequence[str],
    *,
    latency_unit: str,
    mwer_segmenter: Path,
) -> List[str]:
    character_level = latency_unit == "char"
    hypothesis_text = str(prediction or "")
    reference_lines = [str(reference or "") for reference in references]
    if character_level:
        hypothesis_text = " ".join(hypothesis_text)
        reference_lines = [" ".join(reference) for reference in reference_lines]

    with tempfile.TemporaryDirectory(prefix="rasst-masked-bleu-") as tmp:
        tmp_path = Path(tmp)
        hypothesis_path = tmp_path / "hypothesis.txt"
        reference_path = tmp_path / "reference.txt"
        segments_path = tmp_path / "__segments"
        hypothesis_path.write_text(hypothesis_text, encoding="utf-8")
        reference_path.write_text(
            "".join(reference + "\n" for reference in reference_lines),
            encoding="utf-8",
        )
        process = subprocess.run(
            [
                str(mwer_segmenter),
                "-mref",
                str(reference_path),
                "-hypfile",
                str(hypothesis_path),
                "-usecase",
                "1",
            ],
            cwd=tmp_path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"mwerSegmenter failed with exit code {process.returncode}: "
                f"{process.stdout.strip()}"
            )
        if not segments_path.is_file():
            raise RuntimeError("mwerSegmenter did not create __segments")
        segments = segments_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()

    if character_level:
        segments = [re.sub(r"(.)\s", r"\1", line).strip() for line in segments]
    else:
        segments = [line.strip() for line in segments]
    if len(segments) != len(references):
        raise RuntimeError(
            f"mwerSegmenter returned {len(segments)} segments for "
            f"{len(references)} references"
        )
    return segments


def compute_target_term_masked_bleu(
    *,
    instances_log: Path,
    reference_path: Path,
    audio_yaml_path: Path,
    glossary_path: Path,
    target_language: str,
    sacrebleu_tokenizer: str,
    latency_unit: str,
    mwer_segmenter: Path,
) -> MaskedBleuResult:
    for path, label in (
        (instances_log, "instances log"),
        (reference_path, "reference file"),
        (audio_yaml_path, "audio YAML"),
        (glossary_path, "glossary"),
    ):
        _require_file(path, label)
    _require_file(mwer_segmenter, "mwerSegmenter", executable=True)
    if latency_unit not in {"char", "word"}:
        raise ValueError(f"latency_unit must be char or word, got {latency_unit!r}")

    reference_groups = _read_reference_groups(reference_path, audio_yaml_path)
    predictions = _read_talk_predictions(instances_log)
    missing = sorted(set(reference_groups) - set(predictions))
    extra = sorted(set(predictions) - set(reference_groups))
    if missing or extra:
        raise ValueError(
            "Reference/prediction talk mismatch: "
            f"missing_predictions={missing} extra_predictions={extra}"
        )

    terms = _load_target_terms(glossary_path, target_language)
    patterns = _compile_term_mask_patterns(terms)
    masked_hypotheses: List[str] = []
    masked_references: List[str] = []
    hypothesis_removed = 0
    reference_removed = 0
    for wav, references in reference_groups.items():
        hypotheses = _segment_talk(
            predictions[wav],
            references,
            latency_unit=latency_unit,
            mwer_segmenter=mwer_segmenter,
        )
        for hypothesis, reference in zip(hypotheses, references):
            masked_hypothesis, hyp_count = _mask_target_terms(hypothesis, patterns)
            masked_reference, ref_count = _mask_target_terms(reference, patterns)
            masked_hypotheses.append(masked_hypothesis)
            masked_references.append(masked_reference)
            hypothesis_removed += hyp_count
            reference_removed += ref_count

    try:
        import sacrebleu
    except ImportError as exc:
        raise RuntimeError(
            "sacrebleu is required; install the pinned release requirements"
        ) from exc
    bleu = sacrebleu.corpus_bleu(
        masked_hypotheses,
        [masked_references],
        tokenize=sacrebleu_tokenizer,
    ).score
    return MaskedBleuResult(
        bleu=float(bleu),
        hypothesis_terms_removed=hypothesis_removed,
        reference_terms_removed=reference_removed,
        term_types=len(terms),
        talks=len(reference_groups),
        segments=len(masked_references),
        sacrebleu_tokenizer=sacrebleu_tokenizer,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-log", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--audio-yaml", type=Path, required=True)
    parser.add_argument("--glossary", type=Path, required=True)
    parser.add_argument("--target-language", required=True)
    parser.add_argument("--sacrebleu-tokenizer", required=True)
    parser.add_argument("--latency-unit", choices=("char", "word"), required=True)
    parser.add_argument("--mwer-segmenter", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = compute_target_term_masked_bleu(
        instances_log=args.instances_log,
        reference_path=args.reference,
        audio_yaml_path=args.audio_yaml,
        glossary_path=args.glossary,
        target_language=args.target_language,
        sacrebleu_tokenizer=args.sacrebleu_tokenizer,
        latency_unit=args.latency_unit,
        mwer_segmenter=args.mwer_segmenter,
    )
    payload = json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
