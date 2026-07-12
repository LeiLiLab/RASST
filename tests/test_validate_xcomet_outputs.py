from __future__ import annotations

import csv
import importlib.util
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code/rasst/analysis/rebuttal/validate_xcomet_outputs.py"
)
SPEC = importlib.util.spec_from_file_location("validate_xcomet_outputs", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


SUMMARY_FIELDS = (
    "dataset",
    "method",
    "lang",
    "lm",
    "talks",
    "segments",
    "xcomet_mean",
    "xcomet_mean_x100",
    "xcomet_talk_macro_mean",
    "model_id",
    "model_revision",
    "scoring_config_sha256",
    "instances_log_sha256",
)
PAIRED_FIELDS = (
    "dataset",
    "lang",
    "lm",
    "rasst_method",
    "infinisst_method",
    "paired_talks",
    "paired_segments",
    "rasst_xcomet_mean",
    "infinisst_xcomet_mean",
    "delta_rasst_minus_infinisst",
    "paired_delta_stddev",
    "rasst_wins",
    "ties",
    "infinisst_wins",
)


def _format(value: Any) -> str:
    return f"{value:.10f}" if isinstance(value, float) else str(value)


def _write_tsv(path: Path, rows: List[Dict[str, Any]], fields: Tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row[field]) for field in fields})


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(temp_dir: Path) -> Dict[str, Any]:
    manifest = temp_dir / "manifest.tsv"
    summary = temp_dir / "summary.tsv"
    paired = temp_dir / "paired.tsv"
    segments = temp_dir / "segments.jsonl"
    systems = [
        ("acl", "InfiniSST", "de", "1", ["talk-a", "talk-a", "talk-c"], [0.3, 0.6, 0.9]),
        ("acl", "RASST", "de", "1", ["talk-a", "talk-a", "talk-c"], [0.4, 0.6, 0.7]),
        ("eso", "InfiniSST", "zh", "2", ["talk-b"], [0.2]),
        ("eso", "RASST", "zh", "2", ["talk-b"], [0.1]),
    ]
    manifest.write_text(
        "\t".join(MODULE.SYSTEM_KEY_FIELDS)
        + "\n"
        + "".join("\t".join(system[:4]) + "\n" for system in systems),
        encoding="utf-8",
    )

    segment_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for dataset, method, lang, lm, talk_ids, scores in systems:
        talk_sentence_counts: Dict[str, int] = {}
        for index, (talk_id, score) in enumerate(zip(talk_ids, scores)):
            talk_sentence_index = talk_sentence_counts.get(talk_id, 0)
            talk_sentence_counts[talk_id] = talk_sentence_index + 1
            segment_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "lang": lang,
                    "lm": lm,
                    "talk_id": talk_id,
                    "talk_sentence_index": talk_sentence_index,
                    "source": f"source {dataset} {index}",
                    "reference": f"reference {dataset} {index}",
                    "xcomet_score": score,
                    "error_spans": [],
                    "model": {
                        "id": "Unbabel/toy",
                        "revision": "revision",
                    },
                    "provenance_hashes": {
                        "manifest_sha256": "c" * 64,
                        "instances_log_sha256": "b" * 64,
                        "scoring_config_sha256": "a" * 64,
                    },
                    "talk_prediction_sha256": "d" * 64,
                    "scoring_input_sha256": "e" * 64,
                }
            )
        mean_score = statistics.fmean(scores)
        talk_means = [
            statistics.fmean(
                score for score, score_talk_id in zip(scores, talk_ids) if score_talk_id == talk_id
            )
            for talk_id in sorted(set(talk_ids))
        ]
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "lang": lang,
                "lm": lm,
                "talks": len(set(talk_ids)),
                "segments": len(scores),
                "xcomet_mean": mean_score,
                "xcomet_mean_x100": mean_score * 100.0,
                "xcomet_talk_macro_mean": statistics.fmean(talk_means),
                "model_id": "Unbabel/toy",
                "model_revision": "revision",
                "scoring_config_sha256": "a" * 64,
                "instances_log_sha256": "b" * 64,
            }
        )

    first_deltas = [0.4 - 0.3, 0.6 - 0.6, 0.7 - 0.9]
    paired_rows = [
        {
            "dataset": "acl",
            "lang": "de",
            "lm": "1",
            "rasst_method": "RASST",
            "infinisst_method": "InfiniSST",
            "paired_talks": 2,
            "paired_segments": 3,
            "rasst_xcomet_mean": statistics.fmean([0.4, 0.6, 0.7]),
            "infinisst_xcomet_mean": statistics.fmean([0.3, 0.6, 0.9]),
            "delta_rasst_minus_infinisst": statistics.fmean(first_deltas),
            "paired_delta_stddev": statistics.stdev(first_deltas),
            "rasst_wins": 1,
            "ties": 1,
            "infinisst_wins": 1,
        },
        {
            "dataset": "eso",
            "lang": "zh",
            "lm": "2",
            "rasst_method": "RASST",
            "infinisst_method": "InfiniSST",
            "paired_talks": 1,
            "paired_segments": 1,
            "rasst_xcomet_mean": 0.1,
            "infinisst_xcomet_mean": 0.2,
            "delta_rasst_minus_infinisst": -0.1,
            "paired_delta_stddev": 0.0,
            "rasst_wins": 0,
            "ties": 0,
            "infinisst_wins": 1,
        },
    ]
    _write_tsv(summary, summary_rows, SUMMARY_FIELDS)
    _write_tsv(paired, paired_rows, PAIRED_FIELDS)
    _write_jsonl(segments, segment_rows)
    return {
        "manifest": manifest,
        "summary": summary,
        "paired": paired,
        "segments": segments,
        "summary_rows": summary_rows,
        "paired_rows": paired_rows,
        "segment_rows": segment_rows,
    }


class ValidateXCometOutputsTest(unittest.TestCase):
    def test_validates_small_fixture_and_writes_hashed_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            fixture = _fixture(Path(temp_dir_raw))
            report_path = Path(temp_dir_raw) / "validation.json"
            exit_code = MODULE.main(
                [
                    "--manifest",
                    str(fixture["manifest"]),
                    "--summary-tsv",
                    str(fixture["summary"]),
                    "--paired-tsv",
                    str(fixture["paired"]),
                    "--segments-jsonl",
                    str(fixture["segments"]),
                    "--expected-systems",
                    "4",
                    "--expected-pairs",
                    "2",
                    "--expected-segments",
                    "8",
                    "--report-json",
                    str(report_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["validated_counts"]["segments"], 8)
            self.assertEqual(
                report["sha256"]["segments_jsonl"],
                MODULE.sha256_file(fixture["segments"]),
            )
            first_pair = next(row for row in report["pairs"] if row["dataset"] == "acl")
            self.assertEqual(first_pair["rasst_wins"], 1)
            self.assertEqual(first_pair["ties"], 1)
            self.assertEqual(first_pair["infinisst_wins"], 1)
            first_system = next(
                row
                for row in report["systems"]
                if row["dataset"] == "acl" and row["method"] == "RASST"
            )
            self.assertNotEqual(
                first_system["xcomet_mean"],
                first_system["xcomet_talk_macro_mean"],
            )

    def test_rejects_incorrect_summary_mean(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            fixture = _fixture(Path(temp_dir_raw))
            fixture["summary_rows"][0]["xcomet_mean"] = 0.9
            _write_tsv(fixture["summary"], fixture["summary_rows"], SUMMARY_FIELDS)

            with self.assertRaisesRegex(MODULE.XCometValidationError, "xcomet_mean mismatch"):
                MODULE.validate_outputs(
                    manifest=fixture["manifest"],
                    summary_tsv=fixture["summary"],
                    paired_tsv=fixture["paired"],
                    segments_jsonl=fixture["segments"],
                    expected_systems=4,
                    expected_pairs=2,
                    expected_segments=8,
                )

    def test_rejects_non_finite_segment_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            fixture = _fixture(Path(temp_dir_raw))
            fixture["segment_rows"][0]["xcomet_score"] = float("nan")
            _write_jsonl(fixture["segments"], fixture["segment_rows"])

            with self.assertRaisesRegex(MODULE.XCometValidationError, "Invalid JSON"):
                MODULE.validate_outputs(
                    manifest=fixture["manifest"],
                    summary_tsv=fixture["summary"],
                    paired_tsv=fixture["paired"],
                    segments_jsonl=fixture["segments"],
                    expected_systems=4,
                    expected_pairs=2,
                    expected_segments=8,
                )

    def test_rejects_malformed_error_spans(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            fixture = _fixture(Path(temp_dir_raw))
            fixture["segment_rows"][0]["error_spans"] = [{"severity": "minor"}, "bad"]
            _write_jsonl(fixture["segments"], fixture["segment_rows"])

            with self.assertRaisesRegex(MODULE.XCometValidationError, "error_spans"):
                MODULE.validate_outputs(
                    manifest=fixture["manifest"],
                    summary_tsv=fixture["summary"],
                    paired_tsv=fixture["paired"],
                    segments_jsonl=fixture["segments"],
                    expected_systems=4,
                    expected_pairs=2,
                    expected_segments=8,
                )

    def test_rejects_globally_inconsistent_summary_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            fixture = _fixture(Path(temp_dir_raw))
            fixture["summary_rows"][0]["model_id"] = "Unbabel/other"
            _write_tsv(fixture["summary"], fixture["summary_rows"], SUMMARY_FIELDS)

            with self.assertRaisesRegex(MODULE.XCometValidationError, "not globally consistent"):
                MODULE.validate_outputs(
                    manifest=fixture["manifest"],
                    summary_tsv=fixture["summary"],
                    paired_tsv=fixture["paired"],
                    segments_jsonl=fixture["segments"],
                    expected_systems=4,
                    expected_pairs=2,
                    expected_segments=8,
                )

    def test_rejects_segment_provenance_mismatch_and_bad_hash_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            fixture = _fixture(Path(temp_dir_raw))
            fixture["segment_rows"][0]["provenance_hashes"][
                "scoring_config_sha256"
            ] = "not-a-sha256"
            _write_jsonl(fixture["segments"], fixture["segment_rows"])

            with self.assertRaisesRegex(MODULE.XCometValidationError, "64-hex SHA256"):
                MODULE.validate_outputs(
                    manifest=fixture["manifest"],
                    summary_tsv=fixture["summary"],
                    paired_tsv=fixture["paired"],
                    segments_jsonl=fixture["segments"],
                    expected_systems=4,
                    expected_pairs=2,
                    expected_segments=8,
                )

    def test_rejects_segment_model_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            fixture = _fixture(Path(temp_dir_raw))
            fixture["segment_rows"][0]["model"]["revision"] = "other-revision"
            _write_jsonl(fixture["segments"], fixture["segment_rows"])

            with self.assertRaisesRegex(MODULE.XCometValidationError, "model.revision mismatch"):
                MODULE.validate_outputs(
                    manifest=fixture["manifest"],
                    summary_tsv=fixture["summary"],
                    paired_tsv=fixture["paired"],
                    segments_jsonl=fixture["segments"],
                    expected_systems=4,
                    expected_pairs=2,
                    expected_segments=8,
                )

    def test_rejects_well_formed_but_different_scoring_config_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            fixture = _fixture(Path(temp_dir_raw))
            fixture["segment_rows"][0]["provenance_hashes"][
                "scoring_config_sha256"
            ] = "f" * 64
            _write_jsonl(fixture["segments"], fixture["segment_rows"])

            with self.assertRaisesRegex(
                MODULE.XCometValidationError,
                "scoring_config_sha256 mismatch",
            ):
                MODULE.validate_outputs(
                    manifest=fixture["manifest"],
                    summary_tsv=fixture["summary"],
                    paired_tsv=fixture["paired"],
                    segments_jsonl=fixture["segments"],
                    expected_systems=4,
                    expected_pairs=2,
                    expected_segments=8,
                )

    def test_rejects_source_reference_mismatch_in_strict_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            fixture = _fixture(Path(temp_dir_raw))
            baseline_row = next(
                row
                for row in fixture["segment_rows"]
                if row["dataset"] == "acl"
                and row["method"] == "InfiniSST"
                and row["talk_sentence_index"] == 0
            )
            baseline_row["source"] = "different source"
            _write_jsonl(fixture["segments"], fixture["segment_rows"])

            with self.assertRaisesRegex(MODULE.XCometValidationError, "Source/reference mismatch"):
                MODULE.validate_outputs(
                    manifest=fixture["manifest"],
                    summary_tsv=fixture["summary"],
                    paired_tsv=fixture["paired"],
                    segments_jsonl=fixture["segments"],
                    expected_systems=4,
                    expected_pairs=2,
                    expected_segments=8,
                )


if __name__ == "__main__":
    unittest.main()
