#!/usr/bin/env python3
"""Build sentence-aligned relevance plans for retrieval degradation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


EVAL_ROOT = Path(__file__).resolve().parents[1] / "eval"
import sys

sys.path.insert(0, str(EVAL_ROOT))

from agents.retrieval_degradation import (  # noqa: E402
    PLAN_SCHEMA_VERSION,
    load_glossary,
    source_contains,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan(
    *,
    source_list: Path,
    source_text: Path,
    audio_yaml: Path,
    glossary: Path,
    target_lang: str,
) -> Dict[str, Any]:
    source_paths = source_list.read_text(encoding="utf-8").splitlines()
    source_sentences = source_text.read_text(encoding="utf-8").splitlines()
    audio_rows = yaml.safe_load(audio_yaml.read_text(encoding="utf-8"))
    if not source_paths:
        raise ValueError(f"Empty source list: {source_list}")
    if not isinstance(audio_rows, list) or len(audio_rows) != len(source_sentences):
        raise ValueError(
            f"audio/source sentence mismatch: {len(audio_rows or [])} != {len(source_sentences)}"
        )
    glossary_refs = load_glossary(glossary, target_lang)

    sentences_by_paper: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    cursor_by_paper: Dict[str, float] = defaultdict(float)
    for audio_row, sentence_text in zip(audio_rows, source_sentences):
        if not isinstance(audio_row, dict):
            raise ValueError(f"Invalid audio row: {audio_row!r}")
        paper_id = Path(str(audio_row.get("wav") or "")).stem
        if not paper_id:
            raise ValueError(f"Audio row has no wav: {audio_row!r}")
        start = float(audio_row.get("offset", cursor_by_paper[paper_id]))
        end = start + float(audio_row.get("duration") or 0.0)
        relevant = [
            reference
            for reference in glossary_refs
            if source_contains(sentence_text, reference["term"])
        ]
        sentences_by_paper[paper_id].append(
            {
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "references": relevant,
            }
        )
        cursor_by_paper[paper_id] = end

    instances = []
    for instance_index, source_path_value in enumerate(source_paths):
        paper_id = Path(source_path_value).stem
        if paper_id not in sentences_by_paper:
            raise ValueError(
                f"source instance {source_path_value} has no audio.yaml sentences"
            )
        instances.append(
            {
                "instance_index": instance_index,
                "source_path": source_path_value,
                "paper_id": paper_id,
                "sentences": sentences_by_paper[paper_id],
            }
        )
    extra_papers = sorted(set(sentences_by_paper) - {row["paper_id"] for row in instances})
    if extra_papers:
        raise ValueError(f"audio.yaml contains unlisted papers: {extra_papers}")

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "definition": "sentence-aligned source-term relevance over the timeline retrieval window",
        "target_lang": target_lang,
        "source_list": str(source_list.resolve()),
        "source_text": str(source_text.resolve()),
        "audio_yaml": str(audio_yaml.resolve()),
        "glossary_path": str(glossary.resolve()),
        "input_sha256": {
            "source_list": _sha256(source_list),
            "source_text": _sha256(source_text),
            "audio_yaml": _sha256(audio_yaml),
            "glossary": _sha256(glossary),
        },
        "glossary": glossary_refs,
        "instances": instances,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-list", required=True)
    parser.add_argument("--source-text", required=True)
    parser.add_argument("--audio-yaml", required=True)
    parser.add_argument("--glossary", required=True)
    parser.add_argument("--target-lang", choices=["zh", "de", "ja"], required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    plan = build_plan(
        source_list=Path(args.source_list),
        source_text=Path(args.source_text),
        audio_yaml=Path(args.audio_yaml),
        glossary=Path(args.glossary),
        target_lang=args.target_lang,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sentence_count = sum(len(row["sentences"]) for row in plan["instances"])
    relevant_count = sum(
        len(sentence["references"])
        for row in plan["instances"]
        for sentence in row["sentences"]
    )
    print(
        f"wrote={output} instances={len(plan['instances'])} sentences={sentence_count} "
        f"glossary={len(plan['glossary'])} sentence_term_occurrences={relevant_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
