from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code/rasst/analysis/rebuttal/run_gemini_llm_judge_pilot.py"
)
SPEC = importlib.util.spec_from_file_location("run_gemini_llm_judge_pilot", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GeminiLlmJudgePilotTest(unittest.TestCase):
    def test_prompt_matches_wmt_shape_and_is_blind(self) -> None:
        prompt = MODULE.format_prompt("de", "source sentence", "translated sentence")
        self.assertTrue(prompt.startswith("Score the following translation from English to\nGerman"))
        self.assertTrue(prompt.endswith("German translation:\ntranslated sentence"))
        self.assertNotIn("reference", prompt.casefold())
        self.assertNotIn("RASST", prompt)
        self.assertNotIn("InfiniSST", prompt)

    def test_score_parser_accepts_only_canonical_integer(self) -> None:
        for text, expected in (("0", 0), (" 33\n", 33), ("99", 99), ("100", 100)):
            self.assertEqual(MODULE.parse_score(text), expected)
        for text in ("-1", "101", "085", "+85", "85.0", "85%", "Score: 85", '"85"'):
            with self.assertRaises(MODULE.PilotError, msg=text):
                MODULE.parse_score(text)

    def test_pair_allocation_is_proportional_and_deterministic(self) -> None:
        counts = {
            ("acl_tagged_raw", lang, str(lm)): 468
            for lang in ("de", "ja", "zh")
            for lm in range(1, 5)
        }
        counts.update(
            {
                ("medicine_hardraw", "de", str(lm)): 1437
                for lm in range(1, 5)
            }
        )
        allocation = MODULE._allocate_samples(counts, 50)
        self.assertEqual(sum(allocation.values()), 50)
        self.assertEqual(
            [allocation[("acl_tagged_raw", "de", str(lm))] for lm in range(1, 5)],
            [2, 2, 2, 2],
        )
        self.assertEqual(
            [allocation[("medicine_hardraw", "de", str(lm))] for lm in range(1, 5)],
            [7, 7, 6, 6],
        )

    def test_api_key_file_must_be_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "key.txt"
            path.write_text("not-a-real-key\n", encoding="utf-8")
            path.chmod(0o644)
            with self.assertRaises(MODULE.PilotError):
                MODULE.read_api_key(path)
            path.chmod(0o600)
            self.assertEqual(MODULE.read_api_key(path), "not-a-real-key")


if __name__ == "__main__":
    unittest.main()
