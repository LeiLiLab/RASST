from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code/rasst/analysis/rebuttal/score_merged_realistic_glossary.py"
)
SPEC = importlib.util.spec_from_file_location("score_merged_realistic_glossary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _fake_mwer(path: Path) -> Path:
    _write(
        path,
        """#!/bin/sh
set -eu
reference=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -mref) reference="$2"; shift 2 ;;
    *) shift ;;
  esac
done
cp "$reference" __segments
""",
    )
    path.chmod(0o755)
    return path


class ScoreMergedRealisticGlossaryTest(unittest.TestCase):
    def test_strip_output_tags_matches_word_and_character_modes(self) -> None:
        word, word_removed = MODULE.strip_output_tags(
            "Das <t>Fachwort</t> ist korrekt",
            mode="term_t",
            latency_unit="word",
        )
        char, char_removed = MODULE.strip_output_tags(
            "这是<t>术语</t>。",
            mode="term_t",
            latency_unit="char",
        )
        self.assertEqual(word, "Das Fachwort ist korrekt")
        self.assertEqual(char, "这是术语。")
        self.assertEqual(word_removed, 2)
        self.assertEqual(char_removed, 2)

    def test_exact_term_accuracy_uses_source_and_target_raw_gold_gate(self) -> None:
        rows = [
            {
                "wav": "talk.wav",
                "source": "We evaluate a terminology system.",
                "reference": "我们评估术语系统。",
                "prediction": "我们评估术语系统。",
            },
            {
                "wav": "talk.wav",
                "source": "No relevant item appears here.",
                "reference": "这里没有相关项目。",
                "prediction": "术语",
            },
        ]
        terms = [{"source": "terminology", "target": "术语"}]
        result = MODULE.compute_exact_term_accuracy(rows, terms)
        self.assertEqual(result["term_correct"], 1)
        self.assertEqual(result["term_total"], 1)
        self.assertEqual(result["term_acc"], 1.0)

    def test_resegment_prediction_uses_explicit_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mwer = _fake_mwer(root / "mwerSegmenter")
            word_segments = MODULE.resegment_prediction(
                prediction="ignored",
                references=["first sentence", "second sentence"],
                mwer_segmenter=mwer,
                character_level=False,
            )
            char_segments = MODULE.resegment_prediction(
                prediction="忽略",
                references=["第一句", "第二句"],
                mwer_segmenter=mwer,
                character_level=True,
            )
            self.assertEqual(word_segments, ["first sentence", "second sentence"])
            self.assertEqual(char_segments, ["第一句", "第二句"])

    def test_cli_writes_bleu_and_raw_gold_term_counts_without_environment_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mwer = _fake_mwer(root / "mwerSegmenter")
            instances = _write(
                root / "instances.log",
                json.dumps(
                    {
                        "index": 0,
                        "source": ["/audio/talk.wav"],
                        "prediction": "<t>术语</t>",
                    },
                    ensure_ascii=False,
                )
                + "\n",
            )
            source = _write(root / "source.txt", "technical term\nother sentence\n")
            reference = _write(root / "reference.txt", "技术术语\n其他句子\n")
            audio = _write(
                root / "audio.json",
                json.dumps(
                    [
                        {"wav": "/audio/talk.wav", "duration": 1.0, "offset": 0.0},
                        {"wav": "/audio/talk.wav", "duration": 1.0, "offset": 1.0},
                    ]
                ),
            )
            glossary = _write(
                root / "gold.json",
                json.dumps(
                    {
                        "term": {
                            "term": "term",
                            "target_translations": {"zh": "术语"},
                        }
                    },
                    ensure_ascii=False,
                ),
            )
            output_tsv = root / "eval.tsv"
            output_json = root / "eval.json"
            resegmented = root / "resegmented.jsonl"

            class FakeBleu:
                score = 100.0

                def __str__(self) -> str:
                    return "BLEU = 100.00"

            fake_sacrebleu = types.SimpleNamespace(
                corpus_bleu=lambda hypotheses, references, tokenize: FakeBleu()
            )
            previous_sacrebleu = sys.modules.get("sacrebleu")
            previous_argv = sys.argv
            sys.modules["sacrebleu"] = fake_sacrebleu
            sys.argv = [
                str(MODULE_PATH),
                "--instances-log",
                str(instances),
                "--source-file",
                str(source),
                "--reference-file",
                str(reference),
                "--audio-manifest",
                str(audio),
                "--glossary",
                str(glossary),
                "--target-language",
                "zh",
                "--latency-unit",
                "char",
                "--sacrebleu-tokenizer",
                "zh",
                "--mwer-segmenter",
                str(mwer),
                "--strip-output-tags",
                "term_t",
                "--output-tsv",
                str(output_tsv),
                "--output-json",
                str(output_json),
                "--resegmented-jsonl",
                str(resegmented),
            ]
            try:
                self.assertEqual(MODULE.main(), 0)
            finally:
                sys.argv = previous_argv
                if previous_sacrebleu is None:
                    del sys.modules["sacrebleu"]
                else:
                    sys.modules["sacrebleu"] = previous_sacrebleu

            report = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(report["BLEU"], 100.0)
            self.assertEqual(report["TERM_CORRECT"], 1)
            self.assertEqual(report["TERM_TOTAL"], 1)
            self.assertNotIn("StreamLAAL", report)
            self.assertEqual(len(resegmented.read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
