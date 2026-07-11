from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "code"
    / "rasst"
    / "analysis"
    / "rebuttal"
    / "materialize_xcomet_portable_bundle.py"
)
SPEC = importlib.util.spec_from_file_location(
    "materialize_xcomet_portable_bundle", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


FIELDS = (
    "dataset",
    "method",
    "lang",
    "lm",
    "instances_log",
    "source_text",
    "reference",
    "audio_yaml",
    "latency_unit",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_inputs(root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    payloads = {
        "instances_log": b'{"prediction":"hello"}\n',
        "source_text": b"source\n",
        "reference": b"reference\n",
        "audio_yaml": b'[{"wav":"talk.wav"}]\n',
    }
    for field, payload in payloads.items():
        path = root / f"{field}.txt"
        path.write_bytes(payload)
        values[field] = path.name
    return values


def _base_row(paths: dict[str, str], method: str = "RASST") -> dict[str, str]:
    return {
        "dataset": "acl6060",
        "method": method,
        "lang": "de",
        "lm": "2",
        **paths,
        "latency_unit": "word",
    }


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _tree_contents(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class MaterializeXCometPortableBundleTest(unittest.TestCase):
    def test_tsv_rewrites_relative_paths_and_deduplicates_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            paths = _write_inputs(temp_dir)
            duplicate_dir = temp_dir / "duplicate"
            duplicate_dir.mkdir()
            duplicate_instances = duplicate_dir / "instances_log.txt"
            duplicate_instances.write_bytes((temp_dir / paths["instances_log"]).read_bytes())

            first = _base_row(paths)
            second = _base_row(paths, method="InfiniSST")
            second["instances_log"] = "duplicate/instances_log.txt"
            manifest = temp_dir / "input.tsv"
            _write_tsv(manifest, [first, second])

            output = temp_dir / "bundle"
            metadata = MODULE.materialize_bundle(
                manifest, output, "xcomet_inputs.tsv"
            )

            with (output / "xcomet_inputs.tsv").open(
                encoding="utf-8", newline=""
            ) as handle:
                portable_rows = list(csv.DictReader(handle, delimiter="\t"))
            for row in portable_rows:
                for field in MODULE.PATH_FIELDS:
                    relative = Path(row[field])
                    self.assertFalse(relative.is_absolute())
                    self.assertEqual(relative.parts[0], "files")
                    self.assertTrue((output / relative).is_file())

            self.assertEqual(
                portable_rows[0]["instances_log"],
                portable_rows[1]["instances_log"],
            )
            self.assertEqual(metadata["payload"]["referenced_source_file_count"], 5)
            self.assertEqual(metadata["payload"]["content_address_count"], 4)
            self.assertEqual(len(list((output / "files").rglob("*.txt"))), 4)

            provenance = [
                json.loads(line)
                for line in (output / "provenance.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            duplicate_records = [
                row
                for row in provenance
                if row["bundle_path"] == portable_rows[0]["instances_log"]
            ]
            self.assertEqual(len(duplicate_records), 2)
            for row in provenance:
                payload = (output / row["bundle_path"]).read_bytes()
                self.assertEqual(row["sha256"], _sha256(payload))
                self.assertEqual(row["bytes"], len(payload))
                self.assertEqual(Path(row["bundle_path"]).name, Path(row["original_path"]).name)

    def test_jsonl_output_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            paths = _write_inputs(temp_dir)
            manifest = temp_dir / "input.jsonl"
            manifest.write_text(
                json.dumps(_base_row(paths), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            first = temp_dir / "first"
            second = temp_dir / "second"
            MODULE.materialize_bundle(manifest, first)
            MODULE.materialize_bundle(manifest, second)

            self.assertEqual(_tree_contents(first), _tree_contents(second))
            portable = json.loads(
                (first / "manifest.jsonl").read_text(encoding="utf-8")
            )
            self.assertTrue(
                all(not Path(portable[field]).is_absolute() for field in MODULE.PATH_FIELDS)
            )

    def test_missing_input_fails_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            paths = _write_inputs(temp_dir)
            paths["reference"] = "missing.txt"
            manifest = temp_dir / "input.tsv"
            _write_tsv(manifest, [_base_row(paths)])
            output = temp_dir / "bundle"

            with self.assertRaises(MODULE.PortableBundleError):
                MODULE.materialize_bundle(manifest, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(temp_dir.glob(".bundle.tmp-*")), [])

    def test_corrupt_copy_fails_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            paths = _write_inputs(temp_dir)
            manifest = temp_dir / "input.tsv"
            _write_tsv(manifest, [_base_row(paths)])
            output = temp_dir / "bundle"

            def corrupt_copy(_source: Path, destination: Path) -> None:
                Path(destination).write_bytes(b"corrupt")

            with mock.patch.object(MODULE.shutil, "copyfile", side_effect=corrupt_copy):
                with self.assertRaisesRegex(
                    MODULE.PortableBundleError, "failed verification"
                ):
                    MODULE.materialize_bundle(manifest, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(temp_dir.glob(".bundle.tmp-*")), [])

    def test_changed_deduplicated_source_fails_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            paths = _write_inputs(temp_dir)
            duplicate_dir = temp_dir / "duplicate"
            duplicate_dir.mkdir()
            duplicate_instances = duplicate_dir / "instances_log.txt"
            duplicate_instances.write_bytes((temp_dir / paths["instances_log"]).read_bytes())
            first = _base_row(paths)
            second = _base_row(paths, method="InfiniSST")
            second["instances_log"] = "duplicate/instances_log.txt"
            manifest = temp_dir / "input.tsv"
            _write_tsv(manifest, [first, second])
            output = temp_dir / "bundle"

            original_plan = MODULE._plan_bundle

            def mutate_after_plan(*arguments: object) -> object:
                plan = original_plan(*arguments)
                duplicate_instances.write_text("changed\n", encoding="utf-8")
                return plan

            with mock.patch.object(MODULE, "_plan_bundle", side_effect=mutate_after_plan):
                with self.assertRaisesRegex(
                    MODULE.PortableBundleError, "changed after bundle planning"
                ):
                    MODULE.materialize_bundle(manifest, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(temp_dir.glob(".bundle.tmp-*")), [])

    def test_existing_output_and_unsafe_manifest_name_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            paths = _write_inputs(temp_dir)
            manifest = temp_dir / "input.tsv"
            _write_tsv(manifest, [_base_row(paths)])
            output = temp_dir / "bundle"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(MODULE.PortableBundleError, "already exists"):
                MODULE.materialize_bundle(manifest, output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            with self.assertRaisesRegex(MODULE.PortableBundleError, "single filename"):
                MODULE.materialize_bundle(manifest, temp_dir / "other", "../escape.tsv")

    def test_malformed_tsv_and_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            malformed_tsv = temp_dir / "bad.tsv"
            malformed_tsv.write_text(
                "\t".join(FIELDS) + "\n" + "too\tfew\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(MODULE.PortableBundleError, "columns"):
                MODULE.read_manifest(malformed_tsv)

            malformed_jsonl = temp_dir / "bad.jsonl"
            malformed_jsonl.write_text(
                '{"instances_log":"a","instances_log":"b"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(MODULE.PortableBundleError, "duplicate key"):
                MODULE.read_manifest(malformed_jsonl)


if __name__ == "__main__":
    unittest.main()
