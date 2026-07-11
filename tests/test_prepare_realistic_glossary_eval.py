from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code/rasst/analysis/rebuttal/prepare_realistic_glossary_eval.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_realistic_glossary_eval", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class PrepareRealisticGlossaryEvalTest(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        paper_ids = ["paper.one", "paper.two"]
        extraction_root = root / "extraction"
        paper_rows = []
        for paper_id in paper_ids:
            glossary = {
                f"{paper_id} term": {
                    "term": f"{paper_id} term",
                    "target_translations": {
                        "zh": f"{paper_id} 中文",
                        "de": f"{paper_id} Deutsch",
                        "ja": f"{paper_id} 日本語",
                    },
                    "source": "gemini_paper_extracted",
                }
            }
            glossary_path = _write(
                extraction_root / "glossaries" / f"{paper_id}.json",
                json.dumps(glossary, ensure_ascii=False),
            )
            responses_path = _write(
                extraction_root / "raw_responses" / f"{paper_id}.jsonl",
                json.dumps({"paper_id": paper_id, "response_text": "[]"}) + "\n",
            )
            paper_rows.append(
                {
                    "paper_id": paper_id,
                    "glossary_path": str(glossary_path),
                    "glossary_sha256": MODULE.sha256_file(glossary_path),
                    "raw_responses_path": str(responses_path),
                    "raw_responses_sha256": MODULE.sha256_file(responses_path),
                }
            )
        extraction_manifest = _write(
            extraction_root / "manifest.json",
            json.dumps(
                {
                    "model": "gemini-2.5-flash",
                    "model_metadata": {
                        "lookup_status": "ok",
                        "requested_model": "gemini-2.5-flash",
                        "version": "2.5",
                    },
                    "prompt_sha256": "prompt-hash",
                    "sdk": "google-genai",
                    "sdk_version": "1.0",
                    "data_access_policy": {
                        "excluded": ["gold evaluation glossary"],
                        "manual_filtering": False,
                    },
                    "papers": paper_rows,
                }
            ),
        )

        release_root = root / "release"
        audio_root = release_root / "main_result/audio/acl6060"
        for paper_id in paper_ids:
            _write(audio_root / f"{paper_id}.wav", f"fake audio {paper_id}")
        yaml_text = """- duration: 1.0
  offset: 0.0
  wav: data/main_result/audio/acl6060/paper.one.wav
- duration: 2.0
  offset: 1.0
  wav: data/main_result/audio/acl6060/paper.one.wav
- duration: 3.0
  offset: 0.0
  wav: data/main_result/audio/acl6060/paper.two.wav
"""
        for language in ("zh", "de", "ja"):
            input_root = release_root / f"main_result/inputs/acl_{language}"
            source_name = MODULE.SOURCE_LIST_NAME[language]
            _write(
                input_root / source_name,
                "data/main_result/audio/acl6060/paper.one.wav\n"
                "data/main_result/audio/acl6060/paper.two.wav\n",
            )
            _write(input_root / "target.list", f"{language} talk one\n{language} talk two\n")
            _write(input_root / "source_text.txt", "source 0\nsource 1\nsource 2\n")
            _write(input_root / "ref.txt", f"{language} ref 0\n{language} ref 1\n{language} ref 2\n")
            _write(input_root / "audio.yaml", yaml_text)

        gold = {
            "gold only": {
                "term": "gold only",
                "target_translations": {"zh": "金", "de": "Gold", "ja": "金"},
            }
        }
        gold_path = _write(
            release_root / "glossaries/acl6060_tagged_gt_raw_min_norm2.json",
            json.dumps(gold, ensure_ascii=False),
        )
        return {
            "extraction_manifest": extraction_manifest,
            "release_root": release_root,
            "gold": gold_path,
            "paper_ids": paper_ids,
        }

    def test_prepare_separates_runtime_glossary_from_gold_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            manifest_path = MODULE.prepare(
                extraction_manifest_path=fixture["extraction_manifest"],
                release_data_root=fixture["release_root"],
                gold_glossary_path=fixture["gold"],
                output_dir=root / "prepared",
                paper_ids=fixture["paper_ids"],
                languages=("zh", "de", "ja"),
                expected_model="gemini-2.5-flash",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["separation_policy"]["gold_used_to_build_runtime_glossary"])
            self.assertEqual(len(manifest["shards"]), 6)
            self.assertEqual(
                manifest["fixed_raw_gold_eval_glossary"]["sha256"],
                MODULE.sha256_file(fixture["gold"]),
            )

            zh_one = next(
                shard
                for shard in manifest["shards"]
                if shard["language"] == "zh" and shard["paper_id"] == "paper.one"
            )
            runtime = json.loads(
                Path(zh_one["runtime_glossary"]["path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(list(runtime), ["paper.one term"])
            self.assertEqual(
                set(runtime["paper.one term"]["target_translations"]),
                {"zh"},
            )
            self.assertNotIn("gold only", runtime)
            self.assertEqual(zh_one["sentence_count"], 2)
            shard_audio = json.loads(
                Path(zh_one["files"]["audio_yaml"]["path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(len(shard_audio), 2)
            self.assertTrue(Path(shard_audio[0]["wav"]).is_absolute())

    def test_extraction_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            manifest = json.loads(fixture["extraction_manifest"].read_text(encoding="utf-8"))
            manifest["papers"][0]["glossary_sha256"] = "0" * 64
            fixture["extraction_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.prepare(
                    extraction_manifest_path=fixture["extraction_manifest"],
                    release_data_root=fixture["release_root"],
                    gold_glossary_path=fixture["gold"],
                    output_dir=root / "prepared",
                    paper_ids=fixture["paper_ids"],
                    languages=("zh", "de", "ja"),
                    expected_model="gemini-2.5-flash",
                )

    def test_flat_audio_yaml_parser_preserves_scalar_types(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = _write(
                Path(temporary) / "audio.yaml",
                "- duration: 1.25\n  offset: 0\n  speaker_id: NA\n  wav: paper.wav\n",
            )
            rows = MODULE.load_flat_audio_yaml(path)
            self.assertEqual(rows[0]["duration"], 1.25)
            self.assertEqual(rows[0]["offset"], 0)
            self.assertEqual(rows[0]["speaker_id"], "NA")


if __name__ == "__main__":
    unittest.main()
