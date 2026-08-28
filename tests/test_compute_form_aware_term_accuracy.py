from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code/rasst/eval/offline_sst_eval/compute_form_aware_term_accuracy.py"
)
SPEC = importlib.util.spec_from_file_location("compute_form_aware_term_accuracy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Token:
    def __init__(self, text: str, lemma: str) -> None:
        self.text = text
        self.lemma_ = lemma
        self.is_space = False
        self.is_punct = False


def fake_german_nlp(text: str):
    lemmas = {"Wörtern": "Wort", "Wörter": "Wort"}
    return [Token(piece, lemmas.get(piece, piece.casefold())) for piece in text.split()]


class FormAwareTermAccuracyTest(unittest.TestCase):
    def test_surface_and_lemma_matches_are_conservative(self) -> None:
        self.assertTrue(MODULE.orthographic_match("eine Soft max Funktion", "Softmax").matched)
        self.assertTrue(MODULE.orthographic_match("transformer-basiert", "Transformer").matched)
        self.assertFalse(MODULE.orthographic_match("Training", "AI").matched)
        self.assertTrue(MODULE.lemma_match("mehrere Wörter", "Wörtern", fake_german_nlp).matched)

    def test_scoring_preserves_exact_denominator(self) -> None:
        sentences = [
            MODULE.AlignedSentence(
                index=0,
                wav="talk.wav",
                source="Softmax words",
                reference="Softmax Wörtern",
                hypothesis="Soft max mehrere Wörter",
            )
        ]
        glossary = [
            MODULE.GlossaryTerm(source="Softmax", target="Softmax"),
            MODULE.GlossaryTerm(source="words", target="Wörtern"),
            MODULE.GlossaryTerm(source="missing", target="Fehlt"),
        ]
        summary, rows = MODULE.score_sentences(
            sentences,
            glossary,
            "de",
            nlp=fake_german_nlp,
        )
        self.assertEqual(summary["total_terms"], 2)
        self.assertEqual(summary["exact_correct"], 0)
        self.assertEqual(summary["form_aware_correct"], 2)
        self.assertEqual(len(rows), 2)

    def test_glossary_deduplicates_target_forms(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            path = Path(temp_dir_raw) / "glossary.json"
            path.write_text(
                json.dumps(
                    {
                        "first": {
                            "term": "model",
                            "target_translations": {"de": "Modell"},
                        },
                        "second": {
                            "term": "models",
                            "target_translations": {"de": "Modell"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            terms = MODULE.load_glossary(path, "de")
        self.assertEqual(terms, [MODULE.GlossaryTerm(source="model", target="Modell")])


if __name__ == "__main__":
    unittest.main()
