from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code/rasst/analysis/rebuttal/compute_term_prevalence.py"
)
SPEC = importlib.util.spec_from_file_location("compute_term_prevalence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _space_tokenize(text: str) -> list[str]:
    return text.split()


class ComputeTermPrevalenceTest(unittest.TestCase):
    def test_find_token_spans_is_case_insensitive(self) -> None:
        self.assertEqual(
            MODULE.find_token_spans(["A", "Neural", "Model"], ["neural", "model"]),
            [(1, 3)],
        )
        self.assertEqual(
            MODULE.find_token_spans(
                ["Sprachmodelle"],
                ["Sprachmodell"],
                allow_single_token_substring=True,
            ),
            [(0, 1)],
        )

    def test_prevalence_uses_position_union_for_overlapping_terms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source = tmp_path / "source.txt"
            reference = tmp_path / "ref.txt"
            glossary = tmp_path / "glossary.json"
            source.write_text("neural language model works\nordinary words\n", encoding="utf-8")
            reference.write_text(
                "neuronales Sprachmodell funktioniert\nnormale Woerter\n",
                encoding="utf-8",
            )
            glossary.write_text(
                json.dumps(
                    [
                        {
                            "term": "language model",
                            "target_translations": {"de": "Sprachmodell"},
                        },
                        {
                            "term": "neural language model",
                            "target_translations": {"de": "neuronales Sprachmodell"},
                        },
                    ]
                ),
                encoding="utf-8",
            )

            result = MODULE.compute_prevalence(
                dataset="toy",
                lang="de",
                tokenizer_name="space",
                source_path=source,
                reference_path=reference,
                glossary_path=glossary,
                tokenize_source=_space_tokenize,
                tokenize_target=_space_tokenize,
            )

            self.assertEqual(result.source_glossary_term_tokens, 3)
            self.assertEqual(result.aligned_gold_term_tokens, 2)
            self.assertEqual(result.aligned_term_pair_occurrences, 2)
            self.assertEqual(result.source_glossary_term_sentences, 1)
            self.assertEqual(result.aligned_gold_term_sentences, 1)


if __name__ == "__main__":
    unittest.main()
