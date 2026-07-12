from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "code"
    / "rasst"
    / "analysis"
    / "rebuttal"
    / "target_term_masked_bleu.py"
)
SPEC = importlib.util.spec_from_file_location("target_term_masked_bleu", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TargetTermMaskingTest(unittest.TestCase):
    def test_longest_match_prevents_overlap_double_count(self) -> None:
        patterns = MODULE._compile_term_mask_patterns(["York", "New York"])
        masked, removed = MODULE._mask_target_terms(
            "New York and yorkshire meet York",
            patterns,
        )
        self.assertEqual(masked, "and yorkshire meet")
        self.assertEqual(removed, 2)

    def test_cjk_longest_match(self) -> None:
        patterns = MODULE._compile_term_mask_patterns(["机器", "机器学习"])
        masked, removed = MODULE._mask_target_terms("机器学习和机器", patterns)
        self.assertEqual(masked, "和")
        self.assertEqual(removed, 2)

    def test_glossary_target_language_and_casefold_deduplication(self) -> None:
        glossary = [
            {"target_translations": {"de": "New York"}},
            {"translation": "new york"},
            {"de": "York"},
            {"target_translations": {"ja": "無視"}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "glossary.json"
            path.write_text(json.dumps(glossary), encoding="utf-8")
            terms = MODULE._load_target_terms(path, "de")
        self.assertEqual(terms, ["New York", "York"])

    def test_duplicate_talk_predictions_fail_fast(self) -> None:
        rows = [
            {"source": ["/a/talk.wav"], "prediction": "first"},
            {"source": ["/b/talk.wav"], "prediction": "second"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instances.log"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate talk prediction"):
                MODULE._read_talk_predictions(path)


if __name__ == "__main__":
    unittest.main()
