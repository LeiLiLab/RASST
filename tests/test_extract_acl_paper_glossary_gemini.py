from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code/rasst/analysis/rebuttal/extract_acl_paper_glossary_gemini.py"
)
SPEC = importlib.util.spec_from_file_location("extract_acl_paper_glossary_gemini", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ExtractAclPaperGlossaryGeminiTest(unittest.TestCase):
    def test_split_text_is_bounded_and_deterministic(self) -> None:
        text = "\n\n".join(f"paragraph {index} " + "x" * 600 for index in range(12))
        chunks = MODULE.split_text(text, chunk_chars=1_400, max_chunks=3)
        self.assertEqual(len(chunks), 3)
        self.assertIn("paragraph 0", chunks[0])
        self.assertIn("paragraph 11", chunks[-1])

    def test_parse_and_merge_terms(self) -> None:
        parsed = MODULE.parse_json_array(
            '```json\n[{"term":"BERT","target_translations":{"zh":"BERT","de":"BERT","ja":"BERT"}}]\n```'
        )
        merged = MODULE.validate_and_merge_terms([*parsed, *parsed])
        self.assertEqual(list(merged), ["bert"])
        self.assertEqual(merged["bert"]["target_translations"]["de"], "BERT")

    def test_missing_translation_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.validate_and_merge_terms(
                [{"term": "BERT", "target_translations": {"zh": "BERT", "de": "BERT"}}]
            )


if __name__ == "__main__":
    unittest.main()
