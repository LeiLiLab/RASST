from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code/rasst/analysis/rebuttal/analyze_term_failure_chain.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_term_failure_chain", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AnalyzeTermFailureChainTest(unittest.TestCase):
    def test_acl_talk_scope_rejects_non_acl_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "ACL-talk-only"):
            MODULE.analyze(
                term_adoption_path=Path("unused"),
                runtime_log_path=Path("unused"),
                audio_yaml_path=Path("unused"),
                xcomet_segments_path=Path("unused"),
                dataset="medicine_hardraw",
                lang="ja",
                lm=2,
                output_dir=Path("unused"),
                require_acl_talks=True,
            )

    def test_load_retrieval_noise_audit_filters_language_and_validates_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            path = Path(temp_dir_raw) / "noise.tsv"
            header = (
                "lang\tsentence_index\tterm\ttranslation\taudit_label\taudit_note\n"
            )
            path.write_text(
                header
                + "de\t3\toracle\tOracle\tharmful_unsupported_adoption\twrong entity\n"
                + "zh\t3\toracle\t甲骨文\tharmful_unsupported_adoption\twrong entity\n",
                encoding="utf-8",
            )
            rows = MODULE.load_retrieval_noise_audit(path, "de")
            self.assertEqual(set(rows), {(3, "oracle", "Oracle")})

            path.write_text(
                header + "de\t3\toracle\tOracle\tnot_a_label\tbad\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsupported retrieval-noise"):
                MODULE.load_retrieval_noise_audit(path, "de")

    def test_load_flat_audio_yaml_and_morphology_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            path = Path(temp_dir_raw) / "audio.yaml"
            path.write_text(
                "- duration: 1.5\n"
                "  offset: 2.0\n"
                "  wav: data/talk-a.wav\n",
                encoding="utf-8",
            )
            rows = MODULE.load_audio_sentences(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].paper_id, "talk-a")
        self.assertEqual(rows[0].start_sec, 2.0)
        self.assertEqual(rows[0].end_sec, 3.5)
        self.assertEqual(
            MODULE.morphology_candidate("Informationen", "mit informationen")[
                "candidate_kind"
            ],
            "casefold_or_compound_substring",
        )
        self.assertEqual(
            MODULE.morphology_candidate("Softmax", "eine Soft max Funktion")[
                "candidate_kind"
            ],
            "spacing_or_hyphen_variant",
        )
        self.assertEqual(
            MODULE.morphology_candidate("Wörtern", "mehrere Wörter")[
                "candidate_kind"
            ],
            "fuzzy_inflection_candidate",
        )

    def test_end_to_end_timing_and_quality_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            root = Path(temp_dir_raw)
            audio = root / "audio.yaml"
            runtime = root / "runtime.jsonl"
            adoption = root / "term_adoption.json"
            xcomet = root / "segments.jsonl"
            output = root / "output"
            audio.write_text(
                "- duration: 1.0\n"
                "  offset: 0.0\n"
                "  wav: data/talk-a.wav\n",
                encoding="utf-8",
            )
            runtime_rows = [
                {
                    "type": "rag_window",
                    "segment_idx": 0,
                    "current_start_sec": 0.0,
                    "current_end_sec": 0.8,
                },
                {
                    "type": "llm_input",
                    "segment_idx": 0,
                    "references": [
                        {
                            "term": "alpha",
                            "translation": "Alpha",
                            "time_start": 0.2,
                            "time_end": 0.4,
                        }
                    ],
                },
                {
                    "type": "rag_window",
                    "segment_idx": 1,
                    "current_start_sec": 0.8,
                    "current_end_sec": 1.2,
                },
                {
                    "type": "llm_input",
                    "segment_idx": 1,
                    "references": [
                        {
                            "term": "beta",
                            "translation": "Beta",
                            "time_start": 0.7,
                            "time_end": 0.9,
                        }
                    ],
                },
            ]
            runtime.write_text(
                "".join(json.dumps(row) + "\n" for row in runtime_rows),
                encoding="utf-8",
            )
            sentence = {
                "index": 0,
                "source": "alpha beta gamma",
                "reference": "Alpha Beta Gamma",
                "hypothesis": "Alpha",
                "terms": [
                    {"term": "alpha", "translation": "Alpha", "adopted": True},
                    {"term": "beta", "translation": "Beta", "adopted": False},
                    {"term": "gamma", "translation": "Gamma", "adopted": False},
                ],
                "term_map_negative_terms": [{"term": "noise", "translation": "Noise"}],
                "term_map_false_copy_terms": [],
            }
            adoption.write_text(
                json.dumps({"sentences": [sentence]}),
                encoding="utf-8",
            )
            xcomet_rows = [
                {
                    "dataset": "toy",
                    "lang": "de",
                    "lm": "2",
                    "method": "RASST",
                    "sentence_index": 0,
                    "source": sentence["source"],
                    "reference": sentence["reference"],
                    "hypothesis": sentence["hypothesis"],
                    "xcomet_score": 0.7,
                    "error_spans": [],
                },
                {
                    "dataset": "toy",
                    "lang": "de",
                    "lm": "2",
                    "method": "InfiniSST",
                    "sentence_index": 0,
                    "source": sentence["source"],
                    "reference": sentence["reference"],
                    "hypothesis": "Alpha Gamma",
                    "xcomet_score": 0.8,
                    "error_spans": [],
                },
            ]
            xcomet.write_text(
                "".join(json.dumps(row) + "\n" for row in xcomet_rows),
                encoding="utf-8",
            )

            result = MODULE.analyze(
                term_adoption_path=adoption,
                runtime_log_path=runtime,
                audio_yaml_path=audio,
                xcomet_segments_path=xcomet,
                dataset="toy",
                lang="de",
                lm=2,
                output_dir=output,
            )

            self.assertEqual(result["gold_occurrence_count"], 3)
            self.assertEqual(result["exact_correct_count"], 1)
            self.assertEqual(
                result["retrieval_conditionals"]["retrieved_on_time"]["occurrence_count"],
                1,
            )
            self.assertEqual(
                result["retrieval_conditionals"]["retrieved_late"]["occurrence_count"],
                1,
            )
            self.assertEqual(
                result["retrieval_conditionals"]["never_retrieved"]["occurrence_count"],
                1,
            )
            self.assertEqual(
                result["failure_chain"]["retrieved_late_not_exact"][
                    "occurrence_count"
                ],
                1,
            )
            self.assertEqual(result["failure_chain"]["retriever_miss"]["occurrence_count"], 1)
            self.assertAlmostEqual(
                result["quality_by_sentence_group"]["all"]["mean_xcomet_delta"],
                -0.1,
            )
            self.assertTrue((output / "occurrences.tsv").is_file())
            self.assertTrue((output / "german_exact_miss_candidates.tsv").is_file())


if __name__ == "__main__":
    unittest.main()
