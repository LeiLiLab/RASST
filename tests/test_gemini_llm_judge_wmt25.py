from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterator
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code/rasst/analysis/rebuttal/gemini_llm_judge_wmt25.py"
)
SPEC = importlib.util.spec_from_file_location("gemini_llm_judge_wmt25", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


EXPECTED_PROMPT_TEMPLATE = """Score the following translation from {source_lang} to
{target_lang} on a scale from 0 to 100, where a score of 0 means a
broken or poor translation; 33 indicates a flawed translation
with significant issues; 66 indicates a good translation with
only minor issues in grammar, fluency, or consistency; and 100
represents a perfect translation in both meaning and grammar.
Answer with only a whole number representing the score, and
nothing else.
{source_lang} source text:
{source_seg}
{target_lang} translation:
{target_seg}"""
EXPECTED_PROMPT_SHA256 = "56c396ed097093f51c8febc748c8862f1866ca5c83516ef74f6667c5d682e859"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_rows(
    *, dataset: str, languages: tuple[str, ...], segments_per_system: int
) -> Iterator[dict[str, Any]]:
    talk_prefix = "acl" if dataset == MODULE.ACL_DATASET else "medicine"
    for language in languages:
        for lm in MODULE.LM_SETTINGS:
            for sentence_number in range(segments_per_system):
                talk_number = sentence_number % MODULE.EXPECTED_TALKS_PER_SYSTEM
                talk_id = f"{talk_prefix}_talk_{talk_number}"
                talk_sentence_index = sentence_number // MODULE.EXPECTED_TALKS_PER_SYSTEM
                shared_identity = f"{dataset}:{language}:lm{lm}:{talk_id}:{talk_sentence_index}"
                for method_number, method in enumerate(MODULE.METHODS):
                    yield {
                        "dataset": dataset,
                        "method": method,
                        "lang": language,
                        "lm": lm,
                        "talk_id": talk_id,
                        "talk_sentence_index": talk_sentence_index,
                        "source": f"source::{shared_identity}",
                        "hypothesis": f"translation-variant-{method_number}::{shared_identity}",
                        "reference": f"REFERENCE_ONLY_SENTINEL::{shared_identity}",
                    }


def _write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(_nested_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_nested_keys(child))
        return keys
    return set()


class GeminiLlmJudgeWmt25Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls.fixture_dir = Path(cls._temp_dir.name)
        cls.acl_path = cls.fixture_dir / "acl_release_cache_segments.jsonl"
        cls.medicine_path = cls.fixture_dir / "medicine_paper_exact_segments.jsonl"

        acl_rows = _matrix_rows(
            dataset=MODULE.ACL_DATASET,
            languages=("zh", "de", "ja"),
            segments_per_system=MODULE.EXPECTED_SEGMENTS_PER_SYSTEM[MODULE.ACL_DATASET],
        )
        _write_jsonl(cls.acl_path, acl_rows)
        with cls.acl_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "dataset": MODULE.MEDICINE_DATASET,
                        "hypothesis": "OLD_MEDICINE_MUST_BE_IGNORED",
                        "legacy": True,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        _write_jsonl(
            cls.medicine_path,
            _matrix_rows(
                dataset=MODULE.MEDICINE_DATASET,
                languages=("de",),
                segments_per_system=MODULE.EXPECTED_SEGMENTS_PER_SYSTEM[
                    MODULE.MEDICINE_DATASET
                ],
            ),
        )
        cls.acl_sha256 = _sha256_file(cls.acl_path)
        cls.medicine_sha256 = _sha256_file(cls.medicine_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_dir.cleanup()

    def test_prompt_template_and_sha_are_exact(self) -> None:
        self.assertEqual(MODULE.PROMPT_TEMPLATE, EXPECTED_PROMPT_TEMPLATE)
        self.assertEqual(
            hashlib.sha256(MODULE.PROMPT_TEMPLATE.encode("utf-8")).hexdigest(),
            EXPECTED_PROMPT_SHA256,
        )
        expected = EXPECTED_PROMPT_TEMPLATE.format(
            source_lang="English",
            target_lang="German",
            source_seg="SOURCE_SENTINEL",
            target_seg="HYPOTHESIS_SENTINEL",
        )
        self.assertEqual(
            MODULE.format_prompt("de", "SOURCE_SENTINEL", "HYPOTHESIS_SENTINEL"),
            expected,
        )
        self.assertFalse(expected.endswith("\n"))
        self.assertNotIn("reference", expected.casefold())

    def test_score_and_success_response_parsing_are_strict(self) -> None:
        for text, expected in (("0", 0), (" 9\n", 9), ("33", 33), ("99", 99), ("100", 100)):
            with self.subTest(text=text):
                self.assertEqual(MODULE.parse_score_text(text), expected)
        for text in (
            None,
            85,
            "",
            " ",
            "-1",
            "+85",
            "101",
            "085",
            "85.0",
            "85%",
            "Score: 85",
            '"85"',
            "85\nexplanation",
        ):
            with self.subTest(text=text):
                with self.assertRaises(MODULE.LLMJudgeError):
                    MODULE.parse_score_text(text)

        response = {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "role": "model",
                        "parts": [
                            {"thought": True, "text": "private reasoning"},
                            {"text": " 66\n"},
                        ],
                    },
                }
            ],
            "modelVersion": "gemini-2.5-pro-001",
            "usageMetadata": {
                "promptTokenCount": 42,
                "candidatesTokenCount": 1,
                "totalTokenCount": 43,
            },
            "responseId": "response-id",
        }
        parsed = MODULE._parse_success_response(response)
        self.assertEqual(parsed["judge_score"], 66)
        self.assertEqual(parsed["judge_raw_text"], " 66\n")
        self.assertEqual(parsed["model_version"], "gemini-2.5-pro-001")
        self.assertEqual(
            parsed["usage_tokens"],
            {
                "prompt_tokens": 42,
                "candidate_tokens": 1,
                "thinking_tokens": 0,
                "total_tokens": 43,
            },
        )

        invalid_responses = []
        invalid = copy.deepcopy(response)
        invalid["candidates"] = []
        invalid_responses.append(invalid)
        invalid = copy.deepcopy(response)
        invalid["candidates"][0]["finishReason"] = "MAX_TOKENS"
        invalid_responses.append(invalid)
        invalid = copy.deepcopy(response)
        invalid["candidates"][0]["content"]["parts"] = [{"thought": True, "text": "85"}]
        invalid_responses.append(invalid)
        invalid = copy.deepcopy(response)
        invalid["candidates"][0]["content"]["parts"].append({"text": "67"})
        invalid_responses.append(invalid)
        invalid = copy.deepcopy(response)
        invalid["candidates"][0]["content"]["parts"][1]["text"] = "Score: 66"
        invalid_responses.append(invalid)
        invalid = copy.deepcopy(response)
        del invalid["modelVersion"]
        invalid_responses.append(invalid)
        invalid = copy.deepcopy(response)
        invalid["usageMetadata"] = []
        invalid_responses.append(invalid)
        for invalid in invalid_responses:
            with self.subTest(invalid=invalid):
                with self.assertRaises(MODULE.LLMJudgeError):
                    MODULE._parse_success_response(invalid)

    def test_generation_config_modes_are_explicit(self) -> None:
        self.assertEqual(MODULE.generation_config("api-default"), {})
        self.assertEqual(
            MODULE.generation_config("temperature-zero"),
            {
                "temperature": 0.0,
                "candidateCount": 1,
                "responseMimeType": "text/plain",
            },
        )
        with self.assertRaisesRegex(MODULE.LLMJudgeError, "generation-config-mode"):
            MODULE.generation_config("implicit-guess")

    def test_lexical_absolute_preserves_host_qualified_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "physical"
            target.mkdir()
            alias = root / "host-qualified"
            alias.symlink_to(target, target_is_directory=True)
            preserved = MODULE.lexical_absolute(alias / "artifact.jsonl")
            self.assertEqual(preserved, alias / "artifact.jsonl")
            self.assertNotEqual(preserved, preserved.resolve())

    def test_api_key_file_requires_owner_only_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            key_path = root / "gemini.key"
            key_path.write_text("not-a-real-key\n", encoding="utf-8")
            key_path.chmod(0o644)
            with self.assertRaisesRegex(MODULE.LLMJudgeError, "group/other"):
                MODULE.read_private_api_key(key_path)

            key_path.chmod(0o600)
            self.assertEqual(MODULE.read_private_api_key(key_path), "not-a-real-key")

            symlink_path = root / "gemini-link.key"
            symlink_path.symlink_to(key_path)
            with self.assertRaisesRegex(MODULE.LLMJudgeError, "symlink"):
                MODULE.read_private_api_key(symlink_path)

            key_path.write_text("two tokens\n", encoding="utf-8")
            key_path.chmod(0o600)
            with self.assertRaisesRegex(MODULE.LLMJudgeError, "one non-whitespace token"):
                MODULE.read_private_api_key(key_path)

    def test_prepare_full_strict_matrix_and_blind_requests(self) -> None:
        output_dir = self.fixture_dir / "prepared"
        args = argparse.Namespace(
            acl_segments=self.acl_path,
            acl_segments_sha256=self.acl_sha256,
            medicine_segments=self.medicine_path,
            medicine_segments_sha256=self.medicine_sha256,
            output_dir=output_dir,
            run_id="unit-full-matrix",
            model="gemini-2.5-pro",
            generation_config_mode="api-default",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = MODULE.prepare_run(args)

        self.assertEqual(
            manifest["matrix"],
            {
                "systems": 32,
                "pairs": 16,
                "segments": 22_728,
                "acl_segments": 11_232,
                "medicine_segments": 11_496,
                "shards": 16,
                "empty_hypotheses": 0,
            },
        )
        self.assertEqual(len(manifest["shards"]), 16)
        self.assertEqual(sum(shard["request_count"] for shard in manifest["shards"]), 22_728)
        self.assertEqual(
            sorted(shard["request_count"] for shard in manifest["shards"]),
            [936] * 12 + [2_874] * 4,
        )
        acl_artifact = next(
            artifact
            for artifact in manifest["source_artifacts"]
            if artifact["role"] == "acl_release_cache"
        )
        self.assertEqual(acl_artifact["selected_rows"], 11_232)
        self.assertEqual(acl_artifact["total_rows"], 11_233)
        self.assertEqual(acl_artifact["dataset_counts"][MODULE.MEDICINE_DATASET], 1)

        systems: set[tuple[str, str, str, str]] = set()
        pairs: set[tuple[str, str, str]] = set()
        request_count = 0
        for shard in manifest["shards"]:
            sidecars = {
                row["request_key"]: row
                for _, row in MODULE.iter_jsonl(
                    output_dir / shard["sidecar_path"], label="test sidecar"
                )
            }
            for _, row in MODULE.iter_jsonl(
                output_dir / shard["request_path"], label="test request"
            ):
                request_count += 1
                self.assertEqual(set(row), {"key", "request"})
                self.assertEqual(set(row["request"]), {"contents"})
                self.assertNotIn("generation_config", row["request"])
                self.assertTrue({"reference", "method"}.isdisjoint(_nested_keys(row["request"])))
                sidecar = sidecars[row["key"]]
                request_text = row["request"]["contents"][0]["parts"][0]["text"]
                self.assertEqual(
                    request_text,
                    MODULE.format_prompt(
                        sidecar["lang"], sidecar["source"], sidecar["hypothesis"]
                    ),
                )
                serialized_request = json.dumps(row["request"], ensure_ascii=False)
                self.assertNotIn("REFERENCE_ONLY_SENTINEL", serialized_request)
                self.assertNotIn("OLD_MEDICINE_MUST_BE_IGNORED", serialized_request)
                self.assertNotIn("RASST", serialized_request)
                self.assertNotIn("InfiniSST", serialized_request)
                systems.add(
                    (
                        sidecar["dataset"],
                        sidecar["method"],
                        sidecar["lang"],
                        sidecar["lm"],
                    )
                )
                pairs.add((sidecar["dataset"], sidecar["lang"], sidecar["lm"]))
        self.assertEqual(request_count, 22_728)
        self.assertEqual(len(systems), 32)
        self.assertEqual(len(pairs), 16)

    def test_tampered_input_hash_and_pair_mismatch_fail_closed(self) -> None:
        with self.assertRaisesRegex(MODULE.LLMJudgeError, "SHA-256 mismatch"):
            MODULE.load_selected_source_records(
                acl_segments=self.acl_path,
                acl_expected_sha256="0" * 64,
                medicine_segments=self.medicine_path,
                medicine_expected_sha256=self.medicine_sha256,
            )

        records, _ = MODULE.load_selected_source_records(
            acl_segments=self.acl_path,
            acl_expected_sha256=self.acl_sha256,
            medicine_segments=self.medicine_path,
            medicine_expected_sha256=self.medicine_sha256,
        )
        tampered_records = list(records)
        tampered_index = next(
            index for index, record in enumerate(tampered_records) if record.method == "RASST"
        )
        tampered_records[tampered_index] = dataclasses.replace(
            tampered_records[tampered_index],
            source=tampered_records[tampered_index].source + "::TAMPERED",
        )
        with self.assertRaisesRegex(MODULE.LLMJudgeError, "Source/reference mismatch"):
            MODULE.validate_rebuttal_matrix(tampered_records)

    def test_submit_is_fail_closed_after_success_or_ambiguous_create(self) -> None:
        output_dir = self.fixture_dir / "prepared_submit_state"
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = MODULE.prepare_run(
                argparse.Namespace(
                    acl_segments=self.acl_path,
                    acl_segments_sha256=self.acl_sha256,
                    medicine_segments=self.medicine_path,
                    medicine_segments_sha256=self.medicine_sha256,
                    output_dir=output_dir,
                    run_id="unit-submit-state",
                    model="gemini-2.5-flash",
                    generation_config_mode="api-default",
                )
            )

        class ConfigObject:
            def __init__(self, **kwargs: Any) -> None:
                self.values = kwargs

        fake_types = type(
            "FakeTypes",
            (),
            {
                "UploadFileConfig": ConfigObject,
                "CreateBatchJobConfig": ConfigObject,
                "HttpOptions": ConfigObject,
                "HttpRetryOptions": ConfigObject,
            },
        )

        class FakeFiles:
            def __init__(self) -> None:
                self.calls = 0

            def upload(self, **_: Any) -> Any:
                self.calls += 1
                return type("Uploaded", (), {"name": f"files/input-{self.calls}"})()

        class FakeBatches:
            def __init__(self, *, fail: bool) -> None:
                self.calls = 0
                self.fail = fail

            def create(self, **_: Any) -> Any:
                self.calls += 1
                if self.fail:
                    raise TimeoutError("synthetic ambiguous create")
                state = type("State", (), {"name": "JOB_STATE_PENDING"})()
                return type("Job", (), {"name": "batches/unit-job", "state": state})()

        class FakeClient:
            def __init__(self, *, fail: bool) -> None:
                self.files = FakeFiles()
                self.batches = FakeBatches(fail=fail)

        def run_submit(shard_id: str, client: FakeClient) -> None:
            fake_genai = type("FakeGenAI", (), {"Client": lambda *args, **kwargs: client})
            args = argparse.Namespace(
                output_dir=output_dir,
                shard_id=shard_id,
                api_key_file=Path("unused-by-mock"),
                confirm_run_config_sha256=manifest["run_config_sha256"],
            )
            with (
                mock.patch.object(MODULE, "read_private_api_key", return_value="private-test-key"),
                mock.patch.object(
                    MODULE,
                    "_load_google_genai",
                    return_value=(fake_genai, fake_types, "test-sdk"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                MODULE.submit_shard(args)

        successful_shard = manifest["shards"][0]["shard_id"]
        successful_client = FakeClient(fail=False)
        run_submit(successful_shard, successful_client)
        successful_state = MODULE.read_json(
            output_dir / "states" / f"{successful_shard}.json", label="successful state"
        )
        self.assertEqual(successful_state["status"], "SUBMITTED")
        self.assertEqual(successful_client.files.calls, 1)
        self.assertEqual(successful_client.batches.calls, 1)
        with self.assertRaisesRegex(MODULE.LLMJudgeError, "duplicate or ambiguous"):
            run_submit(successful_shard, successful_client)
        self.assertEqual(successful_client.batches.calls, 1)

        ambiguous_shard = manifest["shards"][1]["shard_id"]
        ambiguous_client = FakeClient(fail=True)
        with self.assertRaisesRegex(MODULE.LLMJudgeError, "must not be retried"):
            run_submit(ambiguous_shard, ambiguous_client)
        ambiguous_state = MODULE.read_json(
            output_dir / "states" / f"{ambiguous_shard}.json", label="ambiguous state"
        )
        self.assertEqual(ambiguous_state["status"], "SUBMISSION_UNCERTAIN")
        self.assertEqual(ambiguous_client.batches.calls, 1)
        with self.assertRaisesRegex(MODULE.LLMJudgeError, "duplicate or ambiguous"):
            run_submit(ambiguous_shard, ambiguous_client)
        self.assertEqual(ambiguous_client.batches.calls, 1)

        uploaded_shard = manifest["shards"][2]["shard_id"]
        uploaded_path = output_dir / "states" / f"{uploaded_shard}.json"
        uploaded_state = MODULE.read_json(uploaded_path, label="uploaded state")
        MODULE.update_state(
            uploaded_path,
            uploaded_state,
            status="UPLOADED",
            details={"uploaded_file_name": "files/already-uploaded"},
        )
        resumed_client = FakeClient(fail=False)
        run_submit(uploaded_shard, resumed_client)
        self.assertEqual(resumed_client.files.calls, 0)
        self.assertEqual(resumed_client.batches.calls, 1)


if __name__ == "__main__":
    unittest.main()
