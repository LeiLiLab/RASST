#!/usr/bin/env python3
"""Compute target-term-masked BLEU for the release main-result artifacts."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULT_DIR = REPO_ROOT / "docs/results/main_result_global_cache30_30_20_20"
DEFAULT_MAIN_RESULT_TSV = DEFAULT_RESULT_DIR / "main_result.tsv"
DEFAULT_OUTPUT_TSV = DEFAULT_RESULT_DIR / "masked_terms_quality.tsv"
DEFAULT_COMPARE_TSV = DEFAULT_RESULT_DIR / "masked_terms_quality_compare_vs_infinisst.tsv"
DEFAULT_ARTIFACT_MAP_TSV = DEFAULT_RESULT_DIR / "masked_terms_artifacts.tsv"
DEFAULT_RELEASE_DATA_ROOT = Path(
    "/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/hf_datasets/rasst-main-result-data"
)
DEFAULT_MWERSEGMENTER_ROOT = Path("/mnt/taurus/home/jiaxuanluo/mwerSegmenter")

OFFLINE_EVAL_PATH = REPO_ROOT / "code/rasst/eval/offline_sst_eval/offline_streamlaal_eval.py"

LANG_DEFAULTS = {
    "zh": {"tokenizer": "zh", "latency_unit": "char", "term_lang": "zh"},
    "ja": {"tokenizer": "ja-mecab", "latency_unit": "char", "term_lang": "ja"},
    "de": {"tokenizer": "13a", "latency_unit": "word", "term_lang": "de"},
}


def _load_offline_eval_module() -> Any:
    spec = importlib.util.spec_from_file_location("rasst_offline_streamlaal_eval", OFFLINE_EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import offline eval module: {OFFLINE_EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _read_single_tsv_row(path: Path) -> Dict[str, str]:
    rows = _read_tsv(path)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one TSV row in {path}, found {len(rows)}")
    return rows[0]


def _write_tsv(path: Path, rows: List[Dict[str, str]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_artifact_map(path: Path) -> Dict[Tuple[str, str, str, str], Dict[str, str]]:
    if not path.is_file():
        return {}
    return {
        (row["dataset"], row["method"], row["lang"], row["lm"]): row
        for row in _read_tsv(path)
    }


def _artifact_inputs(
    release_data_root: Path,
    dataset: str,
    lang: str,
) -> Tuple[Path, Path, Path]:
    if dataset == "acl_tagged_raw":
        input_dir = release_data_root / f"main_result/inputs/acl_{lang}"
        return (
            input_dir / "ref.txt",
            input_dir / "audio.yaml",
            release_data_root / "glossaries/acl6060_tagged_gt_raw_min_norm2.json",
        )
    if dataset == "medicine_hardraw":
        input_dir = release_data_root / f"main_result/inputs/medicine_{lang}"
        return (
            input_dir / f"medicine.ref.{lang}__medicine5_hardraw.txt",
            input_dir / "medicine.audio__medicine5_hardraw.yaml",
            release_data_root / "glossaries/hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json",
        )
    raise ValueError(f"Unsupported dataset for masked terms quality: {dataset}")


def _source_eval_row(source_path: str) -> Tuple[Optional[Dict[str, str]], str]:
    if not source_path:
        return None, "empty_source_path"
    path = Path(source_path)
    if not path.is_file():
        return None, f"source_path_not_file:{source_path}"
    try:
        return _read_single_tsv_row(path), "ok"
    except Exception as exc:
        return None, f"source_tsv_unreadable:{exc}"


def _instances_log_from_source(row: Dict[str, str], source_eval: Optional[Dict[str, str]]) -> Tuple[str, str]:
    if source_eval is not None:
        instances_log = source_eval.get("instances_log", "").strip()
        if instances_log and Path(instances_log).is_file():
            return instances_log, "ok"
        if instances_log:
            return instances_log, f"instances_log_not_file:{instances_log}"

    source_path = row.get("source_path", "").strip()
    if not source_path or not Path(source_path).is_file():
        return "", "source_artifact_unavailable"
    parent = Path(source_path).parent
    for name in ("instances.strip_term.log", "instances.log"):
        candidate = parent / name
        if candidate.is_file():
            return str(candidate), "ok"
    return "", f"no_instances_log_near:{source_path}"


def _mapped_source(
    artifact_row: Optional[Dict[str, str]],
) -> Tuple[Optional[Dict[str, str]], str, str]:
    if artifact_row is None:
        return None, "", "missing_artifact_map"

    instances_log = artifact_row.get("instances_log", "").strip()
    eval_results = artifact_row.get("eval_results", "").strip()
    if not instances_log:
        return None, "", "artifact_map_missing_instances_log"
    if not Path(instances_log).is_file():
        return None, instances_log, f"artifact_instances_log_not_file:{instances_log}"

    if eval_results:
        if not Path(eval_results).is_file():
            return None, instances_log, f"artifact_eval_results_not_file:{eval_results}"
        try:
            return _read_single_tsv_row(Path(eval_results)), instances_log, "ok"
        except Exception as exc:
            return None, instances_log, f"artifact_eval_results_unreadable:{exc}"
    return None, instances_log, "ok"


def _set_mwer_env(mwersegmenter_root: Path) -> None:
    os.environ.setdefault("MWERSEGMENTER_ROOT", str(mwersegmenter_root))
    path_items = os.environ.get("PATH", "").split(os.pathsep)
    if str(mwersegmenter_root) not in path_items:
        os.environ["PATH"] = str(mwersegmenter_root) + os.pathsep + os.environ.get("PATH", "")


def _as_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _format_delta(lhs: str, rhs: str) -> str:
    left = _as_float(lhs)
    right = _as_float(rhs)
    if left is None or right is None:
        return ""
    return f"{left - right:+.4f}"


def _compute_rows(args: argparse.Namespace) -> List[Dict[str, str]]:
    offline_eval = _load_offline_eval_module()
    rows = _read_tsv(Path(args.input_tsv))
    artifact_map = _load_artifact_map(Path(args.artifact_map))
    include_methods = {item.strip() for item in args.include_methods.split(",") if item.strip()}

    out_rows: List[Dict[str, str]] = []
    for row in rows:
        dataset = row.get("dataset", "")
        method = row.get("method", "")
        lang = row.get("lang", "")
        lm = row.get("lm", "")
        if method not in include_methods:
            continue
        if dataset not in {"acl_tagged_raw", "medicine_hardraw"}:
            continue
        if lang not in LANG_DEFAULTS or not lm or lm == "NA":
            continue

        artifact_key = (dataset, method, lang, lm)
        mapped_eval, mapped_instances_log, mapped_status = _mapped_source(artifact_map.get(artifact_key))
        if mapped_status == "ok":
            source_eval = mapped_eval
            source_status = "ok"
            instances_log = mapped_instances_log
            instances_status = "ok"
        else:
            source_eval, source_status = _source_eval_row(row.get("source_path", ""))
            instances_log, instances_status = _instances_log_from_source(row, source_eval)
        base_out = {
            "dataset": dataset,
            "method": method,
            "lang": lang,
            "lm": lm,
            "BLEU": row.get("BLEU", ""),
            "MASKED_TERMS_BLEU": "",
            "DELTA_MASKED_MINUS_BLEU": "",
            "TERM_ACC": row.get("TERM_ACC", ""),
            "TERM_FCR": source_eval.get("TERM_FCR", "") if source_eval else "",
            "FALSE_COPY": source_eval.get("FALSE_COPY", "") if source_eval else "",
            "NEG_TOTAL": source_eval.get("NEG_TOTAL", "") if source_eval else "",
            "FALSE_COPY_TERMS": source_eval.get("FALSE_COPY_TERMS", "") if source_eval else "",
            "MASKED_TERMS_HYP_REMOVED": "",
            "MASKED_TERMS_REF_REMOVED": "",
            "MASKED_TERMS_TYPES": "",
            "instances_log": instances_log,
            "source_path": row.get("source_path", ""),
            "status": "",
            "note": "",
        }
        if source_status != "ok" or instances_status != "ok":
            base_out["status"] = "unavailable"
            base_out["note"] = ";".join(part for part in (source_status, instances_status) if part != "ok")
            out_rows.append(base_out)
            continue

        ref_file, audio_yaml, glossary_path = _artifact_inputs(Path(args.release_data_root), dataset, lang)
        missing_inputs = [str(path) for path in (ref_file, audio_yaml, glossary_path) if not path.is_file()]
        if missing_inputs:
            base_out["status"] = "unavailable"
            base_out["note"] = "missing_inputs:" + ",".join(missing_inputs)
            out_rows.append(base_out)
            continue

        defaults = LANG_DEFAULTS[lang]
        try:
            masked = offline_eval._compute_masked_terms_bleu(
                instances_path=Path(instances_log),
                ref_file=ref_file,
                audio_yaml=audio_yaml,
                sacrebleu_tokenizer=defaults["tokenizer"],
                latency_unit=defaults["latency_unit"],
                target_lang=defaults["term_lang"],
                glossary_path=glossary_path,
            )
        except Exception as exc:
            base_out["status"] = "error"
            base_out["note"] = str(exc)
            out_rows.append(base_out)
            continue

        base_out["MASKED_TERMS_BLEU"] = f"{float(masked.bleu):.4f}"
        base_out["DELTA_MASKED_MINUS_BLEU"] = _format_delta(base_out["MASKED_TERMS_BLEU"], base_out["BLEU"])
        base_out["MASKED_TERMS_HYP_REMOVED"] = masked.hyp_terms_removed
        base_out["MASKED_TERMS_REF_REMOVED"] = masked.ref_terms_removed
        base_out["MASKED_TERMS_TYPES"] = masked.term_types
        base_out["status"] = "ok"
        out_rows.append(base_out)
    return out_rows


def _comparison_rows(metric_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_key = {
        (row["dataset"], row["lang"], row["lm"], row["method"]): row
        for row in metric_rows
    }
    compare: List[Dict[str, str]] = []
    for row in metric_rows:
        if row["method"] != "RASST":
            continue
        key = (row["dataset"], row["lang"], row["lm"])
        baseline = by_key.get((key[0], key[1], key[2], "InfiniSST"))
        out = {
            "dataset": key[0],
            "lang": key[1],
            "lm": key[2],
            "RASST_BLEU": row.get("BLEU", ""),
            "InfiniSST_BLEU": baseline.get("BLEU", "") if baseline else "",
            "delta_BLEU_vs_InfiniSST": "",
            "RASST_MASKED_TERMS_BLEU": row.get("MASKED_TERMS_BLEU", ""),
            "InfiniSST_MASKED_TERMS_BLEU": baseline.get("MASKED_TERMS_BLEU", "") if baseline else "",
            "delta_MASKED_TERMS_BLEU_vs_InfiniSST": "",
            "masked_delta_minus_original_delta": "",
            "status": "ok",
            "note": "",
        }
        out["delta_BLEU_vs_InfiniSST"] = _format_delta(out["RASST_BLEU"], out["InfiniSST_BLEU"])
        out["delta_MASKED_TERMS_BLEU_vs_InfiniSST"] = _format_delta(
            out["RASST_MASKED_TERMS_BLEU"],
            out["InfiniSST_MASKED_TERMS_BLEU"],
        )
        out["masked_delta_minus_original_delta"] = _format_delta(
            out["delta_MASKED_TERMS_BLEU_vs_InfiniSST"],
            out["delta_BLEU_vs_InfiniSST"],
        )
        if row.get("status") != "ok":
            out["status"] = "rasst_unavailable"
            out["note"] = row.get("note", "")
        elif baseline is None:
            out["status"] = "baseline_missing"
        elif baseline.get("status") != "ok":
            out["status"] = "baseline_unavailable"
            out["note"] = baseline.get("note", "")
        compare.append(out)
    return compare


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", default=str(DEFAULT_MAIN_RESULT_TSV))
    parser.add_argument("--output-tsv", default=str(DEFAULT_OUTPUT_TSV))
    parser.add_argument("--output-compare-tsv", default=str(DEFAULT_COMPARE_TSV))
    parser.add_argument("--artifact-map", default=str(DEFAULT_ARTIFACT_MAP_TSV))
    parser.add_argument("--release-data-root", default=str(DEFAULT_RELEASE_DATA_ROOT))
    parser.add_argument("--mwersegmenter-root", default=str(DEFAULT_MWERSEGMENTER_ROOT))
    parser.add_argument("--include-methods", default="RASST,InfiniSST")
    args = parser.parse_args()

    _set_mwer_env(Path(args.mwersegmenter_root))
    metric_rows = _compute_rows(args)
    metric_fields = [
        "dataset",
        "method",
        "lang",
        "lm",
        "BLEU",
        "MASKED_TERMS_BLEU",
        "DELTA_MASKED_MINUS_BLEU",
        "TERM_ACC",
        "TERM_FCR",
        "FALSE_COPY",
        "NEG_TOTAL",
        "FALSE_COPY_TERMS",
        "MASKED_TERMS_HYP_REMOVED",
        "MASKED_TERMS_REF_REMOVED",
        "MASKED_TERMS_TYPES",
        "instances_log",
        "source_path",
        "status",
        "note",
    ]
    _write_tsv(Path(args.output_tsv), metric_rows, metric_fields)

    compare_rows = _comparison_rows(metric_rows)
    compare_fields = [
        "dataset",
        "lang",
        "lm",
        "RASST_BLEU",
        "InfiniSST_BLEU",
        "delta_BLEU_vs_InfiniSST",
        "RASST_MASKED_TERMS_BLEU",
        "InfiniSST_MASKED_TERMS_BLEU",
        "delta_MASKED_TERMS_BLEU_vs_InfiniSST",
        "masked_delta_minus_original_delta",
        "status",
        "note",
    ]
    _write_tsv(Path(args.output_compare_tsv), compare_rows, compare_fields)

    ok_rows = sum(1 for row in metric_rows if row["status"] == "ok")
    print(f"Wrote {args.output_tsv} ({ok_rows}/{len(metric_rows)} rows computed)")
    print(f"Wrote {args.output_compare_tsv} ({len(compare_rows)} comparison rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
