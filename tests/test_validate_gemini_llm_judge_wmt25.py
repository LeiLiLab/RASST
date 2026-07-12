from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Iterator, List
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCORER_PATH = REPO_ROOT / "code/rasst/analysis/rebuttal/gemini_llm_judge_wmt25.py"
VALIDATOR_PATH = (
    REPO_ROOT
    / "code/rasst/analysis/rebuttal/validate_gemini_llm_judge_wmt25.py"
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCORER = _load_module("gemini_llm_judge_wmt25_test_fixture", SCORER_PATH)
VALIDATOR = _load_module("validate_gemini_llm_judge_wmt25", VALIDATOR_PATH)


SMALL_SEGMENTS_PER_SYSTEM = {
    SCORER.ACL_DATASET: 5,
    SCORER.MEDICINE_DATASET: 5,
}
SMALL_ACL_SEGMENTS = 2 * 3 * 4 * 5
SMALL_MEDICINE_SEGMENTS = 2 * 1 * 4 * 5
SMALL_SEGMENTS = SMALL_ACL_SEGMENTS + SMALL_MEDICINE_SEGMENTS


@contextlib.contextmanager
def _small_matrix() -> Iterator[None]:
    with ExitStack() as stack:
        for module in (SCORER, VALIDATOR):
            stack.enter_context(
                mock.patch.object(
                    module,
                    "EXPECTED_SEGMENTS_PER_SYSTEM",
                    dict(SMALL_SEGMENTS_PER_SYSTEM),
                )
            )
            stack.enter_context(
                mock.patch.object(module, "EXPECTED_SEGMENTS", SMALL_SEGMENTS)
            )
            stack.enter_context(
                mock.patch.object(
                    module, "EXPECTED_ACL_SEGMENTS", SMALL_ACL_SEGMENTS
                )
            )
            stack.enter_context(
                mock.patch.object(
                    module, "EXPECTED_MEDICINE_SEGMENTS", SMALL_MEDICINE_SEGMENTS
                )
            )
        yield


def _source_rows(dataset: str, languages: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method in SCORER.METHODS:
        for lang in languages:
            for lm in SCORER.LM_SETTINGS:
                for talk_number in range(5):
                    talk_id = f"{dataset}-talk-{talk_number}"
                    source = f"source {dataset} {lang} lm{lm} talk{talk_number}"
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "lang": lang,
                            "lm": lm,
                            "talk_id": talk_id,
                            "talk_sentence_index": 0,
                            "source": source,
                            "hypothesis": f"{method} translation {lang} {lm} {talk_number}",
                            "reference": f"reference {dataset} {lang} {lm} {talk_number}",
                        }
                    )
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _score(sidecar: Dict[str, Any]) -> int:
    base = 45 + int(sidecar["lm"]) + {"de": 0, "ja": 2, "zh": 4}[sidecar["lang"]]
    if sidecar["method"] == "RASST":
        base += (int(str(sidecar["talk_id"]).rsplit("-", 1)[1]) % 3) - 1
    return base


def _build_fixture(
    root: Path, *, split_medicine: bool = True, complete: bool = True
) -> Dict[str, Any]:
    acl_source = root / "acl_source.jsonl"
    medicine_source = root / "medicine_source.jsonl"
    _write_jsonl(
        acl_source,
        _source_rows(SCORER.ACL_DATASET, ["zh", "de", "ja"]),
    )
    _write_jsonl(
        medicine_source,
        _source_rows(SCORER.MEDICINE_DATASET, ["de"]),
    )
    medicine_infinisst_lm123_source = root / "medicine_infinisst_lm123_source.jsonl"
    medicine_infinisst_lm4_source = root / "medicine_infinisst_lm4_source.jsonl"
    medicine_rasst_source = root / "medicine_rasst_source.jsonl"
    for path in (
        medicine_infinisst_lm123_source,
        medicine_infinisst_lm4_source,
        medicine_rasst_source,
    ):
        _write_jsonl(path, _source_rows(SCORER.MEDICINE_DATASET, ["de"]))
    output_dir = root / "judge_run"
    prepare_kwargs = dict(
        output_dir=output_dir,
        run_id="unit-test",
        model="gemini-2.5-pro",
        generation_config_mode="api-default",
        acl_segments=acl_source,
        acl_segments_sha256=SCORER.sha256_file(acl_source),
    )
    if split_medicine:
        prepare_kwargs.update(
            medicine_segments=None,
            medicine_segments_sha256=None,
            medicine_infinisst_lm123_segments=medicine_infinisst_lm123_source,
            medicine_infinisst_lm123_segments_sha256=SCORER.sha256_file(
                medicine_infinisst_lm123_source
            ),
            medicine_infinisst_lm4_segments=medicine_infinisst_lm4_source,
            medicine_infinisst_lm4_segments_sha256=SCORER.sha256_file(
                medicine_infinisst_lm4_source
            ),
            medicine_rasst_segments=medicine_rasst_source,
            medicine_rasst_segments_sha256=SCORER.sha256_file(medicine_rasst_source),
        )
    else:
        prepare_kwargs.update(
            medicine_segments=medicine_source,
            medicine_segments_sha256=SCORER.sha256_file(medicine_source),
        )
    prepare_args = argparse.Namespace(**prepare_kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        SCORER.prepare_run(prepare_args)
    if not complete:
        return {
            "output_dir": output_dir,
            "acl_source": acl_source,
            "medicine_source": medicine_source,
        }
    manifest = _read_json(output_dir / "run_manifest.json")
    for shard in manifest["shards"]:
        shard_id = shard["shard_id"]
        sidecars = [
            row
            for _, row in SCORER.iter_jsonl(
                output_dir / shard["sidecar_path"], label="test sidecar"
            )
        ]
        response_rows = []
        for sidecar in sidecars:
            score = _score(sidecar)
            response_rows.append(
                {
                    "key": sidecar["request_key"],
                    "response": {
                        "candidates": [
                            {
                                "finishReason": "STOP",
                                "content": {"parts": [{"text": str(score)}]},
                            }
                        ],
                        "modelVersion": "gemini-2.5-pro-001",
                        "usageMetadata": {
                            "promptTokenCount": 10,
                            "candidatesTokenCount": 1,
                            "totalTokenCount": 11,
                        },
                        "responseId": f"response-{sidecar['request_key'][-12:]}",
                    },
                }
            )
        response_path = output_dir / "responses" / f"{shard_id}.jsonl"
        response_path.parent.mkdir(parents=True, exist_ok=True)
        SCORER.atomic_write_text(
            response_path,
            "".join(SCORER.json_line(row) for row in response_rows),
        )
        response_file_name = f"files/{shard_id}-responses"
        raw_status = {
            "name": f"batches/{shard_id}",
            "metadata": {
                "state": "BATCH_STATE_SUCCEEDED",
                "batchStats": {
                    "requestCount": str(len(response_rows)),
                    "successfulRequestCount": str(len(response_rows)),
                    "failedRequestCount": "0",
                    "pendingRequestCount": "0",
                },
            },
            "done": True,
            "response": {"responsesFile": response_file_name},
        }
        raw_status_path = output_dir / "status" / f"{shard_id}.json"
        SCORER.atomic_write_json(raw_status_path, raw_status)
        state_path = output_dir / "states" / f"{shard_id}.json"
        state = _read_json(state_path)
        state.update(
            {
                "status": "DOWNLOADED",
                "raw_api_state": "BATCH_STATE_SUCCEEDED",
                "raw_status_path": str(raw_status_path.relative_to(output_dir)),
                "raw_status_sha256": SCORER.sha256_file(raw_status_path),
                "response_file_name": response_file_name,
                "response_path": str(response_path.relative_to(output_dir)),
                "response_sha256": SCORER.sha256_file(response_path),
                "response_bytes": response_path.stat().st_size,
                "response_rows": len(response_rows),
            }
        )
        SCORER.atomic_write_json(state_path, state)
    with contextlib.redirect_stdout(io.StringIO()):
        SCORER.collect_command(argparse.Namespace(output_dir=output_dir))
    return {
        "output_dir": output_dir,
        "acl_source": acl_source,
        "medicine_source": medicine_source,
    }


def _mutate_first_tsv_value(path: Path, column: str, replacement: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    values = lines[1].split("\t")
    values[header.index(column)] = replacement
    lines[1] = "\t".join(values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ValidateGeminiLlmJudgeWmt25Test(unittest.TestCase):
    def test_validates_prepared_split_source_layout(self) -> None:
        with _small_matrix(), tempfile.TemporaryDirectory() as raw_temp:
            fixture = _build_fixture(Path(raw_temp), complete=False)
            report = VALIDATOR.validate_prepared_output_dir(
                output_dir=fixture["output_dir"]
            )
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["validation_scope"], "prepared-only")
            self.assertEqual(report["validated_counts"]["segments"], SMALL_SEGMENTS)
            self.assertEqual(report["validated_counts"]["source_artifacts"], 4)

    def test_validates_all_artifacts_and_writes_report(self) -> None:
        with _small_matrix(), tempfile.TemporaryDirectory() as raw_temp:
            fixture = _build_fixture(Path(raw_temp))
            report_path = Path(raw_temp) / "validation.json"
            exit_code = VALIDATOR.main(
                [
                    "--output-dir",
                    str(fixture["output_dir"]),
                    "--report-json",
                    str(report_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            report = _read_json(report_path)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["validated_counts"]["systems"], 32)
            self.assertEqual(report["validated_counts"]["pairs"], 16)
            self.assertEqual(
                report["validated_counts"]["segments"], SMALL_SEGMENTS
            )
            self.assertEqual(report["validated_counts"]["talk_pairs"], 80)
            self.assertEqual(report["validated_counts"]["groups"], 6)

    def test_validates_legacy_combined_source_layout(self) -> None:
        with _small_matrix(), tempfile.TemporaryDirectory() as raw_temp:
            fixture = _build_fixture(Path(raw_temp), split_medicine=False)
            report = VALIDATOR.validate_output_dir(output_dir=fixture["output_dir"])
            self.assertEqual(report["status"], "ok")

    def test_rejects_changed_source_artifact_hash(self) -> None:
        with _small_matrix(), tempfile.TemporaryDirectory() as raw_temp:
            fixture = _build_fixture(Path(raw_temp))
            fixture["acl_source"].write_text(
                fixture["acl_source"].read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                VALIDATOR.GeminiJudgeValidationError,
                "source artifact hash mismatch",
            ):
                VALIDATOR.validate_output_dir(output_dir=fixture["output_dir"])

    def test_rejects_inconsistent_raw_batch_stats_even_with_updated_hash(self) -> None:
        with _small_matrix(), tempfile.TemporaryDirectory() as raw_temp:
            fixture = _build_fixture(Path(raw_temp))
            output_dir = fixture["output_dir"]
            state_path = sorted((output_dir / "states").glob("*.json"))[0]
            state = _read_json(state_path)
            raw_path = output_dir / state["raw_status_path"]
            raw_status = _read_json(raw_path)
            raw_status["metadata"]["batchStats"]["successfulRequestCount"] = "0"
            _write_json(raw_path, raw_status)
            state["raw_status_sha256"] = VALIDATOR.sha256_file(raw_path)
            _write_json(state_path, state)
            with self.assertRaisesRegex(
                VALIDATOR.GeminiJudgeValidationError,
                "successfulRequestCount.*mismatch",
            ):
                VALIDATOR.validate_output_dir(output_dir=output_dir)

    def test_rejects_segment_score_model_config_and_usage_mismatches(self) -> None:
        mutations = (
            ("judge_score", 100, "judge_score mismatch"),
            ("judge_model", "gemini-other", "judge_model mismatch"),
            ("generation_config_sha256", "f" * 64, "generation_config_sha256 mismatch"),
            (
                "usage_tokens",
                {
                    "prompt_tokens": 999,
                    "candidate_tokens": 1,
                    "thinking_tokens": 0,
                    "total_tokens": 11,
                },
                "usage_tokens mismatch",
            ),
        )
        for field, value, message in mutations:
            with self.subTest(field=field), _small_matrix(), tempfile.TemporaryDirectory() as raw_temp:
                fixture = _build_fixture(Path(raw_temp))
                segments_path = fixture["output_dir"] / "segments.jsonl"
                rows = [
                    json.loads(line)
                    for line in segments_path.read_text(encoding="utf-8").splitlines()
                ]
                rows[0][field] = value
                _write_jsonl(segments_path, rows)
                with self.assertRaisesRegex(
                    VALIDATOR.GeminiJudgeValidationError, message
                ):
                    VALIDATOR.validate_output_dir(output_dir=fixture["output_dir"])

    def test_rejects_recomputed_summary_paired_talk_and_group_values(self) -> None:
        cases = (
            ("summary.tsv", "llm_judge_mean", "999", "llm_judge_mean mismatch"),
            ("paired.tsv", "rasst_wins", "999", "rasst_wins mismatch"),
            (
                "talk_paired.tsv",
                "delta_rasst_minus_infinisst",
                "999",
                "delta_rasst_minus_infinisst mismatch",
            ),
            ("group_summary.tsv", "cells", "999", "cells mismatch"),
        )
        for name, field, value, message in cases:
            with self.subTest(artifact=name), _small_matrix(), tempfile.TemporaryDirectory() as raw_temp:
                fixture = _build_fixture(Path(raw_temp))
                _mutate_first_tsv_value(
                    fixture["output_dir"] / name,
                    field,
                    value,
                )
                with self.assertRaisesRegex(
                    VALIDATOR.GeminiJudgeValidationError, message
                ):
                    VALIDATOR.validate_output_dir(output_dir=fixture["output_dir"])

    def test_rejects_incorrect_collection_usage_totals(self) -> None:
        with _small_matrix(), tempfile.TemporaryDirectory() as raw_temp:
            fixture = _build_fixture(Path(raw_temp))
            collection_path = fixture["output_dir"] / "collection_manifest.json"
            collection = _read_json(collection_path)
            collection["usage_totals"]["prompt_tokens"] += 1
            _write_json(collection_path, collection)
            with self.assertRaisesRegex(
                VALIDATOR.GeminiJudgeValidationError,
                "usage_totals.prompt_tokens.*mismatch",
            ):
                VALIDATOR.validate_output_dir(output_dir=fixture["output_dir"])


if __name__ == "__main__":
    unittest.main()
