#!/usr/bin/env python3
"""Score merged ACL outputs with explicit mwerSegmenter and raw-gold terms."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


LANGUAGE_DEFAULTS = {
    "zh": {"latency_unit": "char", "tokenizer": "zh"},
    "de": {"latency_unit": "word", "tokenizer": "13a"},
    "ja": {"latency_unit": "char", "tokenizer": "ja-mecab"},
}
PROPER_TERM_TAG = re.compile(r"</?\s*term\s*>", flags=re.IGNORECASE)
PROPER_TERM_OR_T_TAG = re.compile(r"</?\s*(?:term|t)\s*>", flags=re.IGNORECASE)
MALFORMED_TERM_PREFIX = re.compile(
    r"<\s*term\b(?!\s*>)|<\s*term(?=[^\s>/])",
    flags=re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, *, executable: bool = False) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty file: {path}")
    if executable and (path.stat().st_mode & 0o111) == 0:
        raise PermissionError(f"File is not executable: {path}")
    return path.resolve()


def _normalise_space(text: Any) -> str:
    return " ".join(str(text or "").split())


def _edge_space_patterns(*, include_short_t: bool) -> Tuple[re.Pattern[str], ...]:
    tag = r"(?:term|t)" if include_short_t else r"term"
    punctuation = r"[,.;:!?，。；：！？)\]\}]"
    return (
        re.compile(rf"</\s*{tag}\s*>(?P<ws>\s+)(?=-)", flags=re.IGNORECASE),
        re.compile(rf"-(?P<ws>\s+)(?=<\s*{tag}\s*>)", flags=re.IGNORECASE),
        re.compile(rf"</\s*{tag}\s*>(?P<ws>\s+)(?={punctuation})", flags=re.IGNORECASE),
        re.compile(rf"[(\[\{{](?P<ws>\s+)(?=<\s*{tag}\s*>)", flags=re.IGNORECASE),
    )


def strip_output_tags(text: str, *, mode: str, latency_unit: str) -> Tuple[str, int]:
    if mode == "none":
        return str(text or ""), 0
    if mode not in {"term", "term_t"}:
        raise ValueError(f"Unsupported strip mode: {mode}")
    include_short_t = mode == "term_t"
    tag_pattern = PROPER_TERM_OR_T_TAG if include_short_t else PROPER_TERM_TAG
    text = str(text or "")
    if latency_unit == "word":
        words: List[str] = []
        removed = 0
        for word in text.split():
            cleaned, proper_count = tag_pattern.subn("", word)
            cleaned, malformed_count = MALFORMED_TERM_PREFIX.subn("", cleaned)
            removed += proper_count + malformed_count
            if cleaned:
                words.append(cleaned)
        return " ".join(words), removed

    keep = [True] * len(text)
    removed = 0
    for pattern in (tag_pattern, MALFORMED_TERM_PREFIX):
        for match in pattern.finditer(text):
            removed += 1
            for index in range(match.start(), match.end()):
                keep[index] = False
    for pattern in _edge_space_patterns(include_short_t=include_short_t):
        for match in pattern.finditer(text):
            for index in range(match.start("ws"), match.end("ws")):
                keep[index] = False
    return "".join(char for char, should_keep in zip(text, keep) if should_keep), removed


def load_instances(
    path: Path,
    *,
    strip_mode: str,
    latency_unit: str,
) -> Tuple["OrderedDict[str, str]", Dict[str, int]]:
    predictions: "OrderedDict[str, str]" = OrderedDict()
    stats = {"instances": 0, "instances_with_removed_tags": 0, "removed_tag_spans": 0}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Malformed merged instance at {path}:{line_number}")
        source = row.get("source")
        if not isinstance(source, list) or len(source) != 1:
            raise ValueError(f"Malformed merged instance at {path}:{line_number}")
        wav_name = Path(str(source[0])).name
        if not wav_name or wav_name in predictions:
            raise ValueError(f"Missing or duplicate wav in merged instances: {wav_name!r}")
        prediction, removed = strip_output_tags(
            str(row.get("prediction") or ""),
            mode=strip_mode,
            latency_unit=latency_unit,
        )
        predictions[wav_name] = prediction
        stats["instances"] += 1
        if removed:
            stats["instances_with_removed_tags"] += 1
            stats["removed_tag_spans"] += removed
    if not predictions:
        raise ValueError(f"Merged instances file is empty: {path}")
    return predictions, stats


def load_reference_groups(
    *,
    source_path: Path,
    reference_path: Path,
    audio_manifest_path: Path,
) -> "OrderedDict[str, List[Dict[str, str]]]":
    sources = source_path.read_text(encoding="utf-8").splitlines()
    references = reference_path.read_text(encoding="utf-8").splitlines()
    audio_rows = json.loads(audio_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(audio_rows, list) or not all(isinstance(row, dict) for row in audio_rows):
        raise ValueError(f"Audio manifest must be a JSON array of objects: {audio_manifest_path}")
    if not sources or len(sources) != len(references) or len(references) != len(audio_rows):
        raise ValueError(
            f"Aligned input length mismatch: source={len(sources)}, "
            f"reference={len(references)}, audio={len(audio_rows)}"
        )
    groups: "OrderedDict[str, List[Dict[str, str]]]" = OrderedDict()
    for index, (source, reference, audio) in enumerate(zip(sources, references, audio_rows)):
        wav_name = Path(str(audio.get("wav") or "")).name
        if not wav_name or not source.strip() or not reference.strip():
            raise ValueError(f"Malformed aligned input row {index}")
        groups.setdefault(wav_name, []).append(
            {"source": source.strip(), "reference": reference.strip()}
        )
    return groups


def resegment_prediction(
    *,
    prediction: str,
    references: Sequence[str],
    mwer_segmenter: Path,
    character_level: bool,
) -> List[str]:
    if not references:
        raise ValueError("Cannot resegment against an empty reference list")
    prediction_for_tool = prediction
    references_for_tool = list(references)
    if character_level:
        prediction_for_tool = " ".join(prediction)
        references_for_tool = [" ".join(reference) for reference in references]
    with tempfile.TemporaryDirectory(prefix="rasst_mwer_") as temporary:
        temporary_path = Path(temporary)
        prediction_path = temporary_path / "prediction.txt"
        reference_path = temporary_path / "reference.txt"
        prediction_path.write_text(prediction_for_tool, encoding="utf-8")
        reference_path.write_text("\n".join(references_for_tool) + "\n", encoding="utf-8")
        command = [
            str(mwer_segmenter),
            "-mref",
            str(reference_path),
            "-hypfile",
            str(prediction_path),
            "-usecase",
            "1",
        ]
        result = subprocess.run(
            command,
            cwd=str(temporary_path),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"mwerSegmenter failed with code {result.returncode}: {result.stdout.strip()}"
            )
        segments_path = temporary_path / "__segments"
        if not segments_path.is_file():
            raise RuntimeError("mwerSegmenter did not create __segments")
        segments = segments_path.read_text(encoding="utf-8").splitlines()
    if character_level:
        segments = [re.sub(r"(.)\s", r"\1", segment) for segment in segments]
    segments = [segment.strip() for segment in segments]
    if len(segments) != len(references):
        raise ValueError(
            f"mwerSegmenter row mismatch: predictions={len(segments)}, references={len(references)}"
        )
    return segments


def resegment_corpus(
    *,
    predictions: Mapping[str, str],
    reference_groups: Mapping[str, Sequence[Mapping[str, str]]],
    mwer_segmenter: Path,
    latency_unit: str,
) -> List[Dict[str, Any]]:
    if set(predictions) != set(reference_groups):
        raise ValueError(
            f"Talk set mismatch: predictions={sorted(predictions)}, "
            f"references={sorted(reference_groups)}"
        )
    rows: List[Dict[str, Any]] = []
    for wav_name, sentence_rows in reference_groups.items():
        references = [str(row["reference"]) for row in sentence_rows]
        resegmented = resegment_prediction(
            prediction=predictions[wav_name],
            references=references,
            mwer_segmenter=mwer_segmenter,
            character_level=latency_unit == "char",
        )
        for sentence_index, (aligned, prediction) in enumerate(zip(sentence_rows, resegmented)):
            rows.append(
                {
                    "wav": wav_name,
                    "sentence_index_in_talk": sentence_index,
                    "source": str(aligned["source"]),
                    "reference": str(aligned["reference"]),
                    "prediction": prediction,
                }
            )
    return rows


def load_target_terms(path: Path, target_language: str) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        entries: Iterable[Tuple[str, Any]] = data.items()
    elif isinstance(data, list):
        entries = ((str(index), value) for index, value in enumerate(data))
    else:
        raise ValueError(f"Unsupported glossary root type: {type(data).__name__}")
    by_target: Dict[str, str] = {}
    for key, entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Malformed glossary entry: {key!r}")
        translations = entry.get("target_translations")
        if not isinstance(translations, dict):
            continue
        target = str(translations.get(target_language) or "").strip()
        source = str(entry.get("term") or key).strip()
        if target and source and target not in by_target:
            by_target[target] = source
    if not by_target:
        raise ValueError(f"Glossary has no {target_language} target terms: {path}")
    return [
        {"target": target, "source": by_target[target]}
        for target in sorted(by_target)
    ]


def source_contains(source_text: str, term: str) -> bool:
    source_norm = _normalise_space(source_text).casefold()
    term_norm = _normalise_space(term).casefold()
    if not source_norm or not term_norm:
        return False
    if re.fullmatch(r"[a-z0-9][a-z0-9 ._+/#-]*", term_norm):
        pattern = r"(?<![a-z0-9])" + re.escape(term_norm) + r"(?![a-z0-9])"
        return re.search(pattern, source_norm) is not None
    return term_norm in source_norm


def compute_exact_term_accuracy(
    rows: Sequence[Mapping[str, Any]],
    terms: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    correct = 0
    total = 0
    mismatches: List[Dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        source = str(row.get("source") or "")
        reference = str(row.get("reference") or "")
        prediction = str(row.get("prediction") or "")
        for term in terms:
            source_term = str(term["source"])
            target_term = str(term["target"])
            if source_contains(source, source_term) and target_term in reference:
                total += 1
                if target_term in prediction:
                    correct += 1
                else:
                    mismatches.append(
                        {
                            "row_index": row_index,
                            "wav": row.get("wav"),
                            "source_term": source_term,
                            "target_term": target_term,
                            "source": source,
                            "reference": reference,
                            "prediction": prediction,
                        }
                    )
    if total <= 0:
        raise ValueError("Raw-gold exact TERM_ACC denominator is zero")
    return {
        "term_acc": correct / total,
        "term_correct": correct,
        "term_total": total,
        "mismatches": mismatches,
    }


def compute_corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    tokenizer: str,
) -> Tuple[float, str]:
    if len(hypotheses) != len(references) or not hypotheses:
        raise ValueError("BLEU hypothesis/reference length mismatch or empty corpus")
    try:
        import sacrebleu
    except ImportError as exc:
        raise RuntimeError("sacrebleu is required for corpus BLEU") from exc
    score = sacrebleu.corpus_bleu(
        list(hypotheses),
        [list(references)],
        tokenize=tokenizer,
    )
    return float(score.score), str(score)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-log", required=True, type=Path)
    parser.add_argument("--source-file", required=True, type=Path)
    parser.add_argument("--reference-file", required=True, type=Path)
    parser.add_argument("--audio-manifest", required=True, type=Path)
    parser.add_argument("--glossary", required=True, type=Path)
    parser.add_argument("--target-language", required=True, choices=sorted(LANGUAGE_DEFAULTS))
    parser.add_argument("--latency-unit", required=True, choices=["word", "char"])
    parser.add_argument("--sacrebleu-tokenizer", required=True)
    parser.add_argument("--mwer-segmenter", required=True, type=Path)
    parser.add_argument("--strip-output-tags", choices=["none", "term", "term_t"], default="term_t")
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--resegmented-jsonl", required=True, type=Path)
    args = parser.parse_args()

    inputs = {
        "instances_log": _require_file(args.instances_log),
        "source_file": _require_file(args.source_file),
        "reference_file": _require_file(args.reference_file),
        "audio_manifest": _require_file(args.audio_manifest),
        "glossary": _require_file(args.glossary),
        "mwer_segmenter": _require_file(args.mwer_segmenter, executable=True),
    }
    defaults = LANGUAGE_DEFAULTS[args.target_language]
    if args.latency_unit != defaults["latency_unit"]:
        raise ValueError(
            f"Unexpected latency unit for {args.target_language}: {args.latency_unit}"
        )
    if args.sacrebleu_tokenizer != defaults["tokenizer"]:
        raise ValueError(
            f"Unexpected tokenizer for {args.target_language}: {args.sacrebleu_tokenizer}"
        )

    predictions, strip_stats = load_instances(
        inputs["instances_log"],
        strip_mode=args.strip_output_tags,
        latency_unit=args.latency_unit,
    )
    reference_groups = load_reference_groups(
        source_path=inputs["source_file"],
        reference_path=inputs["reference_file"],
        audio_manifest_path=inputs["audio_manifest"],
    )
    resegmented = resegment_corpus(
        predictions=predictions,
        reference_groups=reference_groups,
        mwer_segmenter=inputs["mwer_segmenter"],
        latency_unit=args.latency_unit,
    )
    terms = load_target_terms(inputs["glossary"], args.target_language)
    term_metrics = compute_exact_term_accuracy(resegmented, terms)
    bleu, bleu_signature = compute_corpus_bleu(
        [str(row["prediction"]) for row in resegmented],
        [str(row["reference"]) for row in resegmented],
        tokenizer=args.sacrebleu_tokenizer,
    )

    report = {
        "schema_version": 1,
        "kind": "rasst_realistic_glossary_merged_score",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_language": args.target_language,
        "sacrebleu_tokenizer": args.sacrebleu_tokenizer,
        "strip_output_tags": args.strip_output_tags,
        "talks": len(predictions),
        "sentences": len(resegmented),
        "BLEU": bleu,
        "BLEU_signature": bleu_signature,
        "TERM_ACC": term_metrics["term_acc"],
        "TERM_CORRECT": term_metrics["term_correct"],
        "TERM_TOTAL": term_metrics["term_total"],
        "term_mismatches": term_metrics["mismatches"],
        "strip_stats": strip_stats,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
    }
    _write_jsonl(args.resegmented_jsonl, resegmented)
    _write_json(args.output_json, report)
    header = [
        "lang_code",
        "BLEU",
        "TERM_ACC",
        "TERM_CORRECT",
        "TERM_TOTAL",
        "talks",
        "sentences",
        "instances_log",
        "glossary",
    ]
    values = [
        args.target_language,
        f"{bleu:.6f}",
        f"{term_metrics['term_acc']:.6f}",
        str(term_metrics["term_correct"]),
        str(term_metrics["term_total"]),
        str(len(predictions)),
        str(len(resegmented)),
        str(inputs["instances_log"]),
        str(inputs["glossary"]),
    ]
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    temporary_tsv = Path(str(args.output_tsv) + ".tmp")
    temporary_tsv.write_text("\t".join(header) + "\n" + "\t".join(values) + "\n", encoding="utf-8")
    temporary_tsv.replace(args.output_tsv)
    print(
        "BLEU\t{:.6f}\tTERM_ACC\t{:.6f}\tCORRECT_TERMS\t{}\tTOTAL_TERMS\t{}".format(
            bleu,
            term_metrics["term_acc"],
            term_metrics["term_correct"],
            term_metrics["term_total"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
