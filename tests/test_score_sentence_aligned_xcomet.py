from __future__ import annotations

import contextlib
import importlib.util
import io
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
                sentences_per_segment=1,
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
                    sentences_per_segment=1,
                    timeout_seconds=5.0,
                    file_hasher=hasher,
                )

    def test_prepare_system_groups_streaming_pairs_before_mwer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            source = temp_dir / "source.txt"
            reference = temp_dir / "reference.txt"
            audio_yaml = temp_dir / "audio.yaml"
            instances = temp_dir / "instances.log"
            segmenter = temp_dir / "mwerSegmenter"
            source.write_text(
                "source 0\nsource 1\nsource 2\nsource 3\nsource 4\nsource 5\n",
                encoding="utf-8",
            )
            reference.write_text(
                "ref 0\nref 1\nref 2\nref 3\nref 4\nref 5\n",
                encoding="utf-8",
            )
            audio_yaml.write_text(
                json.dumps([{"wav": "talk-a.wav"}] * 6),
                encoding="utf-8",
            )
            instances.write_text(
                json.dumps(
                    {
                        "source": ["talk-a.wav"],
                        "prediction": "hypotheses 0 through 4|||hypothesis 5",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _write_fake_segmenter(segmenter)
            manifest = MODULE.ManifestRow(
                "acl6060",
                "RASST",
                "ja",
                "2",
                instances,
                source,
                reference,
                audio_yaml,
                "word",
                2,
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
                output_tags=[],
                sentences_per_segment=5,
                timeout_seconds=5.0,
                file_hasher=hasher,
            )

            self.assertEqual(len(prepared.segments), 2)
            first, second = prepared.segments
            self.assertEqual(
                first["source"],
                "source 0 source 1 source 2 source 3 source 4",
            )
            self.assertEqual(first["reference"], "ref 0 ref 1 ref 2 ref 3 ref 4")
            self.assertEqual(first["first_sentence_index"], 0)
            self.assertEqual(first["last_sentence_index"], 4)
            self.assertEqual(first["sentence_count"], 5)
            self.assertEqual(second["first_sentence_index"], 5)
            self.assertEqual(second["last_sentence_index"], 5)
            self.assertEqual(second["sentence_count"], 1)

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
        with self.assertRaisesRegex(MODULE.XCometScoringError, "error-span rows"):
            MODULE.extract_xcomet_output(
                FakePrediction(
                    scores=[0.8, 0.9],
                    metadata=FakePrediction(error_spans=[[]]),
                ),
                2,
            )

    def test_load_xcomet_uses_local_checkpoint(self) -> None:
        checkpoint_calls = []
        comet_module = types.ModuleType("comet")
        expected_model = object()

        def fake_load_from_checkpoint(path: str, *, local_files_only: bool):
            checkpoint_calls.append((path, local_files_only))
            return expected_model

        comet_module.load_from_checkpoint = fake_load_from_checkpoint
        with mock.patch.dict(sys.modules, {"comet": comet_module}):
            model = MODULE.load_xcomet(Path("/models/xcomet.ckpt"))

        self.assertIs(model, expected_model)
        self.assertEqual(checkpoint_calls, [("/models/xcomet.ckpt", True)])

    def test_normal_scoring_assigns_scores_without_recovery(self) -> None:
        manifest = MODULE.ManifestRow(
            "acl6060",
            "RASST",
            "de",
            "1",
            Path("instances.log"),
            Path("source.txt"),
            Path("reference.txt"),
            Path("audio.yaml"),
            "word",
            2,
        )
        system = MODULE.PreparedSystem(
            manifest,
            1,
            [
                {
                    "source": "source",
                    "hypothesis": "hypothesis",
                    "reference": "reference",
                }
            ],
            {},
        )

        class StubModel:
            def __init__(self) -> None:
                self.calls = []

            def predict(self, model_inputs, **kwargs):
                self.calls.append((model_inputs, kwargs))
                return FakePrediction(
                    scores=[0.75],
                    metadata=FakePrediction(error_spans=[[{"severity": "minor"}]]),
                )

        model = StubModel()
        with mock.patch.object(
            MODULE,
            "restricted_comet_prediction_gather_loads",
            return_value=contextlib.nullcontext(),
        ):
            MODULE.score_prepared_systems(
                [system],
                model=model,
                devices=[4, 5],
                batch_size=16,
                num_workers=0,
                progress_bar=False,
            )

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][1]["devices"], [4, 5])
        self.assertEqual(system.segments[0]["xcomet_score"], 0.75)
        self.assertEqual(system.segments[0]["error_spans"][0]["severity"], "minor")

    def test_recovery_arguments_and_gather_directory_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            gather_dir = temp_dir / "gather"
            gather_dir.mkdir()
            for rank in range(2):
                (gather_dir / f"pred_{rank}.pt").write_bytes(f"pred-{rank}".encode())
                (gather_dir / f"batch_indices_{rank}.pt").write_bytes(
                    f"indices-{rank}".encode()
                )

            with self.assertRaisesRegex(MODULE.XCometScoringError, "supplied together"):
                MODULE.resolve_recovery_config(
                    prediction_gather_dir=gather_dir,
                    inference_runner_sha256=None,
                    devices=[4, 5],
                )
            with self.assertRaisesRegex(MODULE.XCometScoringError, "64 hexadecimal"):
                MODULE.resolve_recovery_config(
                    prediction_gather_dir=gather_dir,
                    inference_runner_sha256="not-a-digest",
                    devices=[4, 5],
                )

            recovery = MODULE.resolve_recovery_config(
                prediction_gather_dir=gather_dir,
                inference_runner_sha256="A" * 64,
                devices=[4, 5],
            )
            assert recovery is not None
            self.assertEqual(recovery.inference_runner_sha256, "a" * 64)
            self.assertEqual(
                {path.name for path in recovery.gather_files},
                {
                    "pred_0.pt",
                    "pred_1.pt",
                    "batch_indices_0.pt",
                    "batch_indices_1.pt",
                },
            )
            with self.assertRaisesRegex(MODULE.XCometScoringError, "overwrite an input"):
                MODULE.validate_output_paths(
                    [recovery.gather_files[0], temp_dir / "summary.tsv"],
                    recovery.gather_files,
                )

            extra = gather_dir / "notes.txt"
            extra.write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.XCometScoringError, "exactly one"):
                MODULE.validate_prediction_gather_dir(gather_dir, device_count=2)
            extra.unlink()

            pred_zero = gather_dir / "pred_0.pt"
            pred_zero.unlink()
            symlink_target = temp_dir / "outside-pred.pt"
            symlink_target.write_bytes(b"outside")
            pred_zero.symlink_to(symlink_target)
            with self.assertRaisesRegex(MODULE.XCometScoringError, "regular file"):
                MODULE.validate_prediction_gather_dir(gather_dir, device_count=2)

    def test_recovery_gathers_without_cleanup_and_records_provenance(self) -> None:
        load_calls = []

        class StubTorch:
            def load(self, path, *args, **kwargs):
                load_calls.append((Path(path), args, kwargs))
                return object()

        predict_writer_module = types.ModuleType("comet.models.predict_writer")
        predict_writer_module.torch = StubTorch()

        class StubWriter:
            cleanup_called = False

            def gather_all_predictions(self):
                for entry in sorted(Path(self.output_dir).iterdir()):
                    predict_writer_module.torch.load(entry)
                return FakePrediction(
                    scores=[0.8, 0.9],
                    metadata=FakePrediction(
                        error_spans=[[], [{"start": 1, "end": 2, "severity": "minor"}]]
                    ),
                )

            def cleanup(self):
                StubWriter.cleanup_called = True
                raise AssertionError("recovery must retain its input gather directory")

        predict_writer_module.CustomWriter = StubWriter
        comet_module = types.ModuleType("comet")
        comet_models_module = types.ModuleType("comet.models")
        comet_models_module.predict_writer = predict_writer_module
        fake_modules = {
            "comet": comet_module,
            "comet.models": comet_models_module,
            "comet.models.predict_writer": predict_writer_module,
        }

        with tempfile.TemporaryDirectory() as temp_dir_raw:
            gather_dir = Path(temp_dir_raw) / "gather"
            gather_dir.mkdir()
            for rank in range(2):
                (gather_dir / f"pred_{rank}.pt").write_bytes(f"pred-{rank}".encode())
                (gather_dir / f"batch_indices_{rank}.pt").write_bytes(
                    f"indices-{rank}".encode()
                )
            recovery = MODULE.resolve_recovery_config(
                prediction_gather_dir=gather_dir,
                inference_runner_sha256="a" * 64,
                devices=[4, 5],
            )
            assert recovery is not None
            manifest = MODULE.ManifestRow(
                "acl6060",
                "RASST",
                "de",
                "1",
                Path("instances.log"),
                Path("source.txt"),
                Path("reference.txt"),
                Path("audio.yaml"),
                "word",
                2,
            )
            system = MODULE.PreparedSystem(
                manifest,
                1,
                [{}, {}],
                {"runner_sha256": recovery.inference_runner_sha256},
            )
            output = io.StringIO()
            with mock.patch.dict(sys.modules, fake_modules):
                with contextlib.redirect_stdout(output):
                    MODULE.recover_prepared_systems(
                        [system],
                        recovery=recovery,
                        devices=[4, 5],
                        recovery_runner_sha256="b" * 64,
                    )

            self.assertIn("[RECOVER]", output.getvalue())
            self.assertEqual([row["xcomet_score"] for row in system.segments], [0.8, 0.9])
            self.assertEqual(system.provenance_hashes["runner_sha256"], "a" * 64)
            self.assertEqual(system.provenance_hashes["recovery_runner_sha256"], "b" * 64)
            for name in (
                "pred_0_sha256",
                "pred_1_sha256",
                "batch_indices_0_sha256",
                "batch_indices_1_sha256",
            ):
                self.assertRegex(system.provenance_hashes[name], r"^[0-9a-f]{64}$")
            self.assertFalse(StubWriter.cleanup_called)
            self.assertEqual(len(list(gather_dir.iterdir())), 4)
            self.assertEqual(len(load_calls), 4)
            self.assertTrue(all(call[2] == {"weights_only": False} for call in load_calls))

    def test_comet_prediction_gather_loads_only_restricted_temp_files(self) -> None:
        load_calls = []

        class StubTorch:
            def load(self, path, *args, **kwargs):
                load_calls.append((Path(path), args, kwargs))
                return "loaded"

        predict_writer_module = types.ModuleType("comet.models.predict_writer")
        predict_writer_module.torch = StubTorch()

        class StubWriter:
            def __init__(self, output_dir: Path, requested_path: Path) -> None:
                self.output_dir = output_dir
                self.requested_path = requested_path

            def gather_all_predictions(self):
                return predict_writer_module.torch.load(self.requested_path)

        predict_writer_module.CustomWriter = StubWriter
        comet_module = types.ModuleType("comet")
        comet_models_module = types.ModuleType("comet.models")
        comet_models_module.predict_writer = predict_writer_module
        fake_modules = {
            "comet": comet_module,
            "comet.models": comet_models_module,
            "comet.models.predict_writer": predict_writer_module,
        }

        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            allowed = temp_dir / "pred_0.pt"
            disallowed_name = temp_dir / "checkpoint.pt"
            outside_dir = temp_dir / "outside"
            outside_dir.mkdir()
            outside = outside_dir / "pred_0.pt"
            for path in (allowed, disallowed_name, outside):
                path.touch()

            original_gather = StubWriter.gather_all_predictions
            with mock.patch.dict(sys.modules, fake_modules):
                with MODULE.restricted_comet_prediction_gather_loads():
                    writer = StubWriter(temp_dir, allowed)
                    self.assertEqual(writer.gather_all_predictions(), "loaded")
                    with self.assertRaisesRegex(
                        MODULE.XCometScoringError,
                        "Refusing non-COMET",
                    ):
                        StubWriter(temp_dir, disallowed_name).gather_all_predictions()
                    with self.assertRaisesRegex(
                        MODULE.XCometScoringError,
                        "Refusing non-COMET",
                    ):
                        StubWriter(temp_dir, outside).gather_all_predictions()

            self.assertIs(StubWriter.gather_all_predictions, original_gather)
            self.assertIs(predict_writer_module.torch.__class__, StubTorch)
            self.assertEqual(len(load_calls), 1)
            self.assertEqual(load_calls[0][0], allowed.resolve())
            self.assertEqual(load_calls[0][2], {"weights_only": False})

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
