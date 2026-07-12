#!/usr/bin/env python3
"""Recompute every row in a masked-BLEU snapshot and report discrepancies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

from target_term_masked_bleu import compute_target_term_masked_bleu


def _input_paths(
    release_data_root: Path,
    dataset: str,
    language: str,
) -> Tuple[Path, Path, Path]:
    if dataset == "acl_tagged_raw":
        input_dir = release_data_root / f"main_result/inputs/acl_{language}"
        return (
            input_dir / "ref.txt",
            input_dir / "audio.yaml",
            release_data_root / "glossaries/acl6060_tagged_gt_raw_min_norm2.json",
        )
    if dataset == "medicine_hardraw":
        input_dir = release_data_root / f"main_result/inputs/medicine_{language}"
        return (
            input_dir / f"medicine.ref.{language}__medicine5_hardraw.txt",
            input_dir / "medicine.audio__medicine5_hardraw.yaml",
            release_data_root
            / "glossaries/hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json",
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def _language_settings(language: str) -> Tuple[str, str]:
    settings = {
        "zh": ("zh", "char"),
        "ja": ("ja-mecab", "char"),
        "de": ("13a", "word"),
    }
    try:
        return settings[language]
    except KeyError as exc:
        raise ValueError(f"Unsupported language: {language}") from exc


def _expected_values(row: Dict[str, str]) -> Dict[str, str]:
    return {
        "bleu": row["MASKED_TERMS_BLEU"],
        "hypothesis_terms_removed": row["MASKED_TERMS_HYP_REMOVED"],
        "reference_terms_removed": row["MASKED_TERMS_REF_REMOVED"],
        "term_types": row["MASKED_TERMS_TYPES"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-tsv", type=Path, required=True)
    parser.add_argument("--release-data-root", type=Path, required=True)
    parser.add_argument("--mwer-segmenter", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    with args.summary_tsv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"No rows found in {args.summary_tsv}")

    results = []
    for index, row in enumerate(rows, start=1):
        if row.get("status") != "ok":
            raise ValueError(
                f"Snapshot row {index} is not computable: status={row.get('status')!r}"
            )
        dataset = row["dataset"]
        language = row["lang"]
        reference, audio_yaml, glossary = _input_paths(
            args.release_data_root,
            dataset,
            language,
        )
        tokenizer, latency_unit = _language_settings(language)
        result = compute_target_term_masked_bleu(
            instances_log=Path(row["instances_log"]),
            reference_path=reference,
            audio_yaml_path=audio_yaml,
            glossary_path=glossary,
            target_language=language,
            sacrebleu_tokenizer=tokenizer,
            latency_unit=latency_unit,
            mwer_segmenter=args.mwer_segmenter,
        )
        actual = {
            "bleu": f"{result.bleu:.4f}",
            "hypothesis_terms_removed": str(result.hypothesis_terms_removed),
            "reference_terms_removed": str(result.reference_terms_removed),
            "term_types": str(result.term_types),
        }
        expected = _expected_values(row)
        mismatches = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in expected
            if expected[key] != actual[key]
        }
        results.append(
            {
                "dataset": dataset,
                "method": row["method"],
                "lang": language,
                "lm": row["lm"],
                "talks": result.talks,
                "segments": result.segments,
                "expected": expected,
                "actual": actual,
                "mismatches": mismatches,
            }
        )
        print(
            f"[{index:02d}/{len(rows)}] {dataset} {row['method']} "
            f"{language} lm{row['lm']} mismatches={len(mismatches)}",
            flush=True,
        )

    exact = sum(not item["mismatches"] for item in results)
    payload = {
        "source_summary": str(args.summary_tsv),
        "mwer_segmenter": str(args.mwer_segmenter),
        "rows": len(results),
        "rows_exact": exact,
        "rows_with_discrepancies": len(results) - exact,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: payload[key]
                for key in ("rows", "rows_exact", "rows_with_discrepancies")
            }
        ),
        flush=True,
    )
    return 0 if exact == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
