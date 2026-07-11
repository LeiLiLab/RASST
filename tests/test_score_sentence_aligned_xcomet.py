from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import tempfile
import types
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
    / "score_sentence_aligned_xcomet.py"
)
SPEC = importlib.util.spec_from_file_location("score_sentence_aligned_xcomet", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakePrediction(dict):
    def __getattr__(self, name: str):
        return self[name]


def _write_fake_segmenter(path: Path) -> None:
    script = f"""#!{sys.executable}
import pathlib
import sys

arguments = sys.argv[1:]
hypothesis_path = pathlib.Path(arguments[arguments.index('-hypfile') + 1])
reference_path = pathlib.Path(arguments[arguments.index('-mref') + 1])
hypothesis = hypothesis_path.read_text(encoding='utf-8')
references = reference_path.read_text(encoding='utf-8').splitlines()
segments = hypothesis.split('|||')
if len(segments) != len(references):
    print(f'bad fake input: {{len(segments)}} vs {{len(references)}}', file=sys.stderr)
    raise SystemExit(7)
pathlib.Path('__segments').write_text('\\n'.join(segments) + '\\n', encoding='utf-8')
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class SentenceAlignedXCometTest(unittest.TestCase):
    def test_strip_only_explicit_named_tags(self) -> None:
        text = "<TERM>Hello</term> <other>keep</other> <term kind='x'>keep too</termish>"
        cleaned = MODULE.strip_explicit_output_tags(text, ["term"])
        self.assertEqual(
            cleaned,
            "Hello <other>keep</other> <term kind='x'>keep too</termish>",
        )
        self.assertEqual(MODULE.strip_explicit_output_tags("<t>词</t>", []), "<t>词</t>")
        with self.assertRaises(MODULE.XCometScoringError):
            MODULE.normalise_output_tag_names(["term|script"])

    def test_load_tsv_and_jsonl_manifest_with_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            for name in ("instances.log", "source.txt", "reference.txt", "audio.yaml"):
                (temp_dir / name).write_text("x\n", encoding="utf-8")
            fields = {
                "dataset": "acl6060",
                "method": "RASST",
                "lang": "de",
                "lm": "2",
                "instances_log": "instances.log",
                "source_text": "source.txt",
                "reference": "reference.txt",
                "audio_yaml": "audio.yaml",
                "latency_unit": "word",
            }
            tsv = temp_dir / "manifest.tsv"
            tsv.write_text(
                "\t".join(MODULE.MANIFEST_FIELDS)
                + "\n"
                + "\t".join(fields[name] for name in MODULE.MANIFEST_FIELDS)
                + "\n",
                encoding="utf-8",
            )
            jsonl = temp_dir / "manifest.jsonl"
            jsonl.write_text(json.dumps(fields) + "\n", encoding="utf-8")

            for manifest in (tsv, jsonl):
                rows = MODULE.load_manifest(manifest)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0].system_key, ("acl6060", "RASST", "de", "2"))
                self.assertEqual(rows[0].instances_log, (temp_dir / "instances.log").resolve())

    def test_prepare_system_resegments_every_talk_in_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            source = temp_dir / "source.txt"
            reference = temp_dir / "reference.txt"
            audio_yaml = temp_dir / "audio.yaml"
            instances = temp_dir / "instances.log"
            segmenter = temp_dir / "mwerSegmenter"
            source.write_text("source a1\nsource a2\nsource b1\n", encoding="utf-8")
            reference.write_text("reference a1\nreference a2\nreference b1\n", encoding="utf-8")
            audio_yaml.write_text(
                json.dumps(
                    [
                        {"wav": "/audio/talk-a.wav"},
                        {"wav": "/audio/talk-a.wav"},
                        {"wav": "/audio/talk-b.wav"},
                    ]
                ),
                encoding="utf-8",
            )
            rows = [
                {"index": 8, "source": ["/runs/talk-b.wav"], "prediction": "hyp b1"},
                {
                    "index": 3,
                    "source": ["/runs/talk-a.wav"],
                    "prediction": "<term>hyp a1</term>|||hyp a2",
                },
            ]
            instances.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            _write_fake_segmenter(segmenter)
            manifest = MODULE.ManifestRow(
                dataset="acl6060",
                method="RASST",
                lang="de",
                lm="2",
                instances_log=instances,
                source_text=source,
                reference=reference,
                audio_yaml=audio_yaml,
                latency_unit="word",
                manifest_row=2,
            )
            hasher = MODULE.FileHasher()
            prepared = MODULE.prepare_system(
                manifest,
                manifest_sha256="a" * 64,
                runner_sha256="b" * 64,
                checkpoint_sha256="c" * 64,
                checkpoint_hparams_sha256="d" * 64,
                segmenter=segmenter,
                segmenter_sha256=hasher.sha256(segmenter),
                output_tags=["term"],
                timeout_seconds=5.0,
                file_hasher=hasher,
            )

            self.assertEqual(prepared.talk_count, 2)
            self.assertEqual(len(prepared.segments), 3)
            self.assertEqual(
                [(row["talk_id"], row["sentence_index"]) for row in prepared.segments],
                [("talk-b", 2), ("talk-a", 0), ("talk-a", 1)],
            )
            self.assertEqual(
                [row["hypothesis"] for row in prepared.segments],
                ["hyp b1", "hyp a1", "hyp a2"],
            )
            self.assertIn("instances_log_sha256", prepared.provenance_hashes)

    def test_prepare_system_rejects_unmapped_talk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            source = temp_dir / "source.txt"
            reference = temp_dir / "reference.txt"
            audio_yaml = temp_dir / "audio.yaml"
            instances = temp_dir / "instances.log"
            segmenter = temp_dir / "mwerSegmenter"
            source.write_text("source a\n", encoding="utf-8")
            reference.write_text("reference a\n", encoding="utf-8")
            audio_yaml.write_text(json.dumps([{"wav": "talk-a.wav"}]), encoding="utf-8")
            instances.write_text(
                json.dumps({"source": ["talk-missing.wav"], "prediction": "hyp"}) + "\n",
                encoding="utf-8",
            )
            _write_fake_segmenter(segmenter)
            manifest = MODULE.ManifestRow(
                "acl6060",
                "RASST",
                "de",
                "1",
                instances,
                source,
                reference,
                audio_yaml,
                "word",
                2,
            )
            hasher = MODULE.FileHasher()
            with self.assertRaisesRegex(MODULE.XCometScoringError, "does not occur"):
                MODULE.prepare_system(
                    manifest,
                    manifest_sha256="a" * 64,
                    runner_sha256="b" * 64,
                    checkpoint_sha256="c" * 64,
                    checkpoint_hparams_sha256="d" * 64,
                    segmenter=segmenter,
                    segmenter_sha256=hasher.sha256(segmenter),
                    output_tags=[],
                    timeout_seconds=5.0,
                    file_hasher=hasher,
                )

    def test_extract_xcomet_scores_and_spans(self) -> None:
        output = FakePrediction(
            scores=[0.8, 0.9],
            metadata=FakePrediction(
                error_spans=[[], [{"start": 1, "end": 3, "severity": "minor"}]]
            ),
        )
        scores, spans = MODULE.extract_xcomet_output(output, 2)
        self.assertEqual(scores, [0.8, 0.9])
        self.assertEqual(spans[1][0]["severity"], "minor")

    def test_load_xcomet_allowlists_prediction_for_torch_load(self) -> None:
        class StubPrediction(dict):
            pass

        safe_global_calls = []
        checkpoint_calls = []
        torch_module = types.ModuleType("torch")
        torch_module.serialization = types.SimpleNamespace(
            add_safe_globals=lambda values: safe_global_calls.append(values)
        )
        comet_module = types.ModuleType("comet")
        comet_models_module = types.ModuleType("comet.models")
        comet_utils_module = types.ModuleType("comet.models.utils")
        comet_utils_module.Prediction = StubPrediction
        expected_model = object()

        def fake_load_from_checkpoint(path: str, *, local_files_only: bool):
            checkpoint_calls.append((path, local_files_only))
            return expected_model

        comet_module.load_from_checkpoint = fake_load_from_checkpoint
        fake_modules = {
            "torch": torch_module,
            "comet": comet_module,
            "comet.models": comet_models_module,
            "comet.models.utils": comet_utils_module,
        }
        with mock.patch.dict(sys.modules, fake_modules):
            model = MODULE.load_xcomet(Path("/models/xcomet.ckpt"))

        self.assertIs(model, expected_model)
        self.assertEqual(safe_global_calls, [[StubPrediction]])
        self.assertEqual(checkpoint_calls, [("/models/xcomet.ckpt", True)])

    def test_output_paths_cannot_overwrite_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            protected = temp_dir / "manifest.tsv"
            distinct = temp_dir / "summary.tsv"
            with self.assertRaisesRegex(MODULE.XCometScoringError, "overwrite an input"):
                MODULE.validate_output_paths([protected, distinct], [protected])
            with self.assertRaisesRegex(MODULE.XCometScoringError, "must be distinct"):
                MODULE.validate_output_paths([distinct, distinct], [protected])

    def test_build_strict_paired_comparison(self) -> None:
        def make_system(method: str, scores: list[float]) -> object:
            manifest = MODULE.ManifestRow(
                "acl6060",
                method,
                "de",
                "2",
                Path("instances.log"),
                Path("source.txt"),
                Path("reference.txt"),
                Path("audio.yaml"),
                "word",
                2,
            )
            segments = []
            for index, score in enumerate(scores):
                segments.append(
                    {
                        "talk_id": "talk-a",
                        "talk_sentence_index": index,
                        "source": f"source {index}",
                        "reference": f"reference {index}",
                        "xcomet_score": score,
                    }
                )
            return MODULE.PreparedSystem(manifest, 1, segments, {})

        rasst = make_system("RASST", [0.8, 0.7])
        baseline = make_system("InfiniSST", [0.6, 0.7])
        rows = MODULE.build_paired_rows(
            [rasst, baseline],
            rasst_method="RASST",
            baseline_method="InfiniSST",
        )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["delta_rasst_minus_infinisst"], 0.1)
        self.assertEqual(rows[0]["rasst_wins"], 1)
        self.assertEqual(rows[0]["ties"], 1)

        baseline.segments.pop()
        with self.assertRaisesRegex(MODULE.XCometScoringError, "Unpaired segment keys"):
            MODULE.build_paired_rows(
                [rasst, baseline],
                rasst_method="RASST",
                baseline_method="InfiniSST",
            )


if __name__ == "__main__":
    unittest.main()
