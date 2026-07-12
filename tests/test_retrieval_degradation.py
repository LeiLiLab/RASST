import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "rasst" / "eval"))
sys.path.insert(0, str(ROOT / "code" / "rasst" / "tools"))

from agents.retrieval_degradation import RetrievalDegrader  # noqa: E402
from score_retrieval_degradation import score_runtime_log  # noqa: E402


class RetrievalDegradationTest(unittest.TestCase):
    def _write_plan(self, directory: Path) -> Path:
        plan = {
            "schema_version": 1,
            "target_lang": "de",
            "glossary": [
                {"term": "alpha", "translation": "Alpha"},
                {"term": "beta", "translation": "Beta"},
                {"term": "gamma", "translation": "Gamma"},
                {"term": "delta", "translation": "Delta"},
            ],
            "instances": [
                {
                    "instance_index": 0,
                    "paper_id": "paper0",
                    "sentences": [
                        {
                            "start_sec": 0.0,
                            "end_sec": 4.0,
                            "references": [
                                {"term": "alpha", "translation": "Alpha"},
                                {"term": "beta", "translation": "Beta"},
                            ],
                        }
                    ],
                }
            ],
        }
        path = directory / "plan.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        return path

    def test_full_corruption_preserves_count_and_retrieval_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = self._write_plan(Path(tmp))
            degrader = RetrievalDegrader(str(plan_path), rate=1.0, seed=17)
            original = [
                {"term": "alpha", "translation": "Alpha", "score": 0.9},
                {"term": "beta", "translation": "Beta", "score": 0.8},
            ]
            degraded, audit = degrader.degrade(
                original,
                instance_index=0,
                segment_idx=2,
                current_start_sec=1.0,
                current_end_sec=3.0,
                lookback_sec=1.0,
            )
            self.assertEqual(len(degraded), len(original))
            self.assertEqual([row["score"] for row in degraded], [0.9, 0.8])
            self.assertTrue(all(row["term"] in {"gamma", "delta"} for row in degraded))
            self.assertEqual(len({row["term"] for row in degraded}), 2)
            self.assertEqual(audit["replaced_relevant_hint_count"], 2)
            self.assertEqual(audit["retrieval_precision_original"], 1.0)
            self.assertEqual(audit["retrieval_precision_final"], 0.0)

    def test_zero_corruption_is_identity_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = self._write_plan(Path(tmp))
            original = [{"term": "alpha", "translation": "Alpha", "score": 0.9}]
            degrader = RetrievalDegrader(str(plan_path), rate=0.0, seed=17)
            first, first_audit = degrader.degrade(
                original,
                instance_index=0,
                segment_idx=0,
                current_start_sec=0.0,
                current_end_sec=2.0,
                lookback_sec=1.92,
            )
            second, second_audit = degrader.degrade(
                original,
                instance_index=0,
                segment_idx=0,
                current_start_sec=0.0,
                current_end_sec=2.0,
                lookback_sec=1.92,
            )
            self.assertEqual(first, original)
            self.assertEqual(first, second)
            self.assertEqual(first_audit, second_audit)

    def test_runtime_aggregation_uses_only_llm_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.jsonl"
            audit = {
                "configured_rate": 0.5,
                "seed": 37,
                "hint_count_original": 4,
                "hint_count_final": 4,
                "relevant_gold_count": 2,
                "relevant_hint_count_original": 2,
                "relevant_hint_count_final": 1,
                "replaced_relevant_hint_count": 1,
            }
            rows = [
                {"type": "rag", "retrieval_degradation": audit},
                {"type": "llm_input", "retrieval_degradation": audit},
                {"type": "llm_input", "retrieval_degradation": audit},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            result = score_runtime_log(path)
            self.assertEqual(result["retrieval_events"], 2)
            self.assertEqual(result["hint_count_original"], 8)
            self.assertEqual(result["achieved_replacement_rate"], 0.5)
            self.assertEqual(result["retrieval_precision_final"], 0.25)


if __name__ == "__main__":
    unittest.main()
