from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code/rasst/analysis/rebuttal/run_realistic_glossary_eval.py"
)
SPEC = importlib.util.spec_from_file_location("run_realistic_glossary_eval", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": MODULE.sha256_file(path), "bytes": path.stat().st_size}


class RunRealisticGlossaryEvalTest(unittest.TestCase):
    def _prepared_fixture(self, root: Path) -> tuple[Path, dict[str, object]]:
        files = {
            "source_list": _write(root / "prepared/inputs/zh/paper.one/source.list", "/audio/paper.one.wav\n"),
            "target_list": _write(root / "prepared/inputs/zh/paper.one/target.list", "target\n"),
            "source_text": _write(root / "prepared/inputs/zh/paper.one/source_text.txt", "source\n"),
            "ref": _write(root / "prepared/inputs/zh/paper.one/ref.txt", "reference\n"),
            "audio_yaml": _write(
                root / "prepared/inputs/zh/paper.one/audio.yaml",
                '[{"duration": 1.0, "offset": 0.0, "wav": "/audio/paper.one.wav"}]\n',
            ),
        }
        glossary = _write(
            root / "prepared/runtime_glossaries/zh/paper.one.json",
            json.dumps(
                {
                    "term": {
                        "term": "term",
                        "target_translations": {"zh": "术语"},
                        "source_paper": "paper.one",
                    }
                }
            ),
        )
        gold = _write(
            root / "release/glossaries/gold.json",
            json.dumps({"gold": {"term": "gold", "target_translations": {"zh": "金"}}}),
        )
        source_records = {name: _record(path) for name, path in files.items()}
        prepared = {
            "schema_version": 1,
            "kind": "rasst_acl_realistic_paper_glossary_prepared",
            "gemini_model": "gemini-2.5-flash",
            "paper_ids": ["paper.one"],
            "languages": ["zh"],
            "separation_policy": {"gold_used_to_build_runtime_glossary": False},
            "fixed_raw_gold_eval_glossary": _record(gold),
            "release_inputs": {"zh": {"source_files": source_records}},
            "shards": [
                {
                    "language": "zh",
                    "paper_id": "paper.one",
                    "files": source_records,
                    "runtime_glossary": {**_record(glossary), "term_count": 1},
                }
            ],
        }
        manifest = _write(
            root / "prepared/prepared_manifest.json",
            json.dumps(prepared),
        )
        return manifest, prepared

    def _fake_repo(self, root: Path) -> Path:
        repo = root / "repo"
        _write(
            repo / "code/rasst/retriever/build_maxsim_index.py",
            'TEXT_MODEL_ID = "BAAI/bge-m3"\n',
        )
        cache_tool = _write(
            repo / "code/rasst/eval/tools/maxsim_index_cache_key.py",
            """import json, sys
def value(flag):
    return sys.argv[sys.argv.index(flag) + 1]
cache = value('--cache-dir')
tag = value('--glossary-tag')
path = cache + '/maxsim_' + tag + '_fake.pt'
print(json.dumps({'index_path': path, 'manifest_path': path + '.manifest.json', 'cache_key': 'fake-key', 'short_key': 'fake', 'exists': False}))
""",
        )
        self.assertTrue(cache_tool.is_file())
        _write(repo / "code/rasst/eval/src/batched_vllm_rag_eval.py", "# evaluator\n")
        _write(
            repo / "code/rasst/analysis/rebuttal/score_merged_realistic_glossary.py",
            "# offline evaluator\n",
        )
        return repo

    def test_plan_freezes_paper_specific_runtime_and_raw_gold_eval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path, prepared = self._prepared_fixture(root)
            repo = self._fake_repo(root)
            model = root / "model"
            _write(model / "config.json", "{}\n")
            _write(model / "model.safetensors.index.json", "{}\n")
            _write(model / "model-00001-of-00001.safetensors", "weights\n")
            retriever = _write(root / "retriever.pt", "retriever\n")
            mwer = _write(root / "mwer/mwerSegmenter", "#!/bin/sh\n")
            mwer.chmod(0o755)
            args = argparse.Namespace(
                prepared_manifest=prepared_path,
                repo_root=repo,
                python_bin=Path(sys.executable),
                model=[f"zh={model}"],
                retriever_checkpoint=retriever,
                index_cache_dir=root / "index_cache",
                output_dir=root / "outputs",
                mwer_segmenter_bin=mwer,
                lms=[1, 2, 3, 4],
                density_tag="realistic_gemini25flash",
                index_device="cuda:1",
                rag_device="cuda:1",
                rag_top_k=10,
                rag_score_threshold=0.78,
                rag_timeline_lookback_sec=1.92,
                rag_lora_rank=128,
                text_lora_rank=128,
                text_lora_alpha=256,
                vllm_tp_size=2,
                gpu_memory_utilization=0.72,
                max_model_len=32768,
                max_num_seqs=8,
                scheduler_batch_size=8,
                schedule_mode="round_robin",
                vllm_limit_audio=128,
                vllm_enforce_eager=0,
                disable_custom_all_reduce=0,
                safetensors_load_strategy="lazy",
                temperature=0.6,
                top_p=0.95,
                top_k_decode=20,
                max_new_tokens=40,
                seed=998244353,
            )
            run = MODULE.build_run_manifest(args)
            self.assertEqual(len(run["index_tasks"]), 1)
            self.assertEqual(len(run["eval_tasks"]), 2)
            self.assertEqual(len(run["aggregate_tasks"]), 4)
            command = run["eval_tasks"][0]["command"]
            glossary_index = command.index("--glossary") + 1
            eval_glossary_index = command.index("--eval-glossary") + 1
            self.assertEqual(
                command[glossary_index],
                prepared["shards"][0]["runtime_glossary"]["path"],
            )
            self.assertEqual(
                command[eval_glossary_index],
                prepared["fixed_raw_gold_eval_glossary"]["path"],
            )
            self.assertIn("--skip-offline-eval", command)
            self.assertEqual(run["eval_tasks"][0]["lms"], [1, 2])
            self.assertEqual(run["eval_tasks"][0]["cache_chunks"], 30)
            self.assertEqual(run["eval_tasks"][1]["lms"], [3, 4])
            self.assertEqual(run["eval_tasks"][1]["cache_chunks"], 20)
            offline_command = run["aggregate_tasks"][0]["offline_command"]
            self.assertIn("--mwer-segmenter", offline_command)
            self.assertNotIn("--fbk-fairseq-root", offline_command)
            self.assertEqual(
                offline_command[offline_command.index("--mwer-segmenter") + 1],
                str(mwer.resolve()),
            )

    def test_plan_can_select_only_default_lm2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path, prepared = self._prepared_fixture(root)
            prepared["runtime_glossary_policy"] = "paper-derived plus NLP/AI/CS 10k"
            prepared["runtime_glossary_tag"] = "paper_plus_nlp_ai_cs_10k"
            prepared_path.write_text(json.dumps(prepared), encoding="utf-8")
            repo = self._fake_repo(root)
            model = root / "model"
            _write(model / "config.json", "{}\n")
            _write(model / "model.safetensors.index.json", "{}\n")
            _write(model / "model-00001-of-00001.safetensors", "weights\n")
            retriever = _write(root / "retriever.pt", "retriever\n")
            mwer = _write(root / "mwer/mwerSegmenter", "#!/bin/sh\n")
            mwer.chmod(0o755)
            args = argparse.Namespace(
                prepared_manifest=prepared_path,
                repo_root=repo,
                python_bin=Path(sys.executable),
                model=[f"zh={model}"],
                retriever_checkpoint=retriever,
                index_cache_dir=root / "index_cache",
                output_dir=root / "outputs",
                mwer_segmenter_bin=mwer,
                lms=[2],
                density_tag="realistic_paper_plus_10k",
                index_device="cuda:1",
                rag_device="cuda:1",
                rag_top_k=10,
                rag_score_threshold=0.78,
                rag_timeline_lookback_sec=1.92,
                rag_lora_rank=128,
                text_lora_rank=128,
                text_lora_alpha=256,
                vllm_tp_size=2,
                gpu_memory_utilization=0.72,
                max_model_len=32768,
                max_num_seqs=8,
                scheduler_batch_size=8,
                schedule_mode="round_robin",
                vllm_limit_audio=128,
                vllm_enforce_eager=0,
                disable_custom_all_reduce=0,
                safetensors_load_strategy="lazy",
                temperature=0.6,
                top_p=0.95,
                top_k_decode=20,
                max_new_tokens=40,
                seed=998244353,
            )
            run = MODULE.build_run_manifest(args)
            self.assertEqual(len(run["eval_tasks"]), 1)
            self.assertEqual(run["eval_tasks"][0]["lms"], [2])
            self.assertEqual(run["eval_tasks"][0]["cache_chunks"], 30)
            self.assertEqual(len(run["aggregate_tasks"]), 1)
            self.assertEqual(run["aggregate_tasks"][0]["lm"], 2)
            self.assertEqual(run["parameters"]["lms"], [2])
            self.assertEqual(
                run["parameters"]["runtime_glossary"],
                "paper-derived plus NLP/AI/CS 10k",
            )

    def test_merge_aggregate_inputs_preserves_canonical_paper_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paper_ids = ["paper.one", "paper.two"]
            eval_tasks = []
            prepared_shards = []
            full_sources = []
            full_references = []
            for paper_id in paper_ids:
                out = root / paper_id
                instances = _write(
                    out / "instances.log",
                    json.dumps({"index": 0, "source": [f"/audio/{paper_id}.wav"], "prediction": paper_id})
                    + "\n",
                )
                runtime = _write(
                    out / "runtime.jsonl",
                    json.dumps({"instance_index": 0, "segment_idx": 0, "paper": paper_id}) + "\n",
                )
                source_text = _write(out / "source_text.txt", f"source {paper_id}\n")
                reference = _write(out / "reference.txt", f"reference {paper_id}\n")
                audio = _write(
                    out / "audio.json",
                    json.dumps([{"wav": f"/audio/{paper_id}.wav", "duration": 1.0, "offset": 0.0}])
                    + "\n",
                )
                full_sources.append(f"source {paper_id}")
                full_references.append(f"reference {paper_id}")
                prepared_shards.append(
                    {
                        "language": "zh",
                        "paper_id": paper_id,
                        "files": {
                            "source_text": {"path": str(source_text)},
                            "ref": {"path": str(reference)},
                            "audio_yaml": {"path": str(audio)},
                        },
                    }
                )
                eval_tasks.append(
                    {
                        "task_id": f"eval__{paper_id}",
                        "paper_id": paper_id,
                        "language": "zh",
                        "expected_outputs": {
                            "1": {"instances_log": str(instances), "runtime_log": str(runtime)}
                        },
                    }
                )
            full_source_path = _write(
                root / "full_source.txt", "\n".join(full_sources) + "\n"
            )
            full_reference_path = _write(
                root / "full_reference.txt", "\n".join(full_references) + "\n"
            )
            prepared = {
                "paper_ids": paper_ids,
                "shards": prepared_shards,
                "release_inputs": {
                    "zh": {
                        "source_files": {
                            "source_text": {"path": str(full_source_path)},
                            "ref": {"path": str(full_reference_path)},
                        }
                    }
                },
            }
            prepared_path = _write(root / "prepared.json", json.dumps(prepared))
            aggregate_dir = root / "aggregate"
            aggregate_task = {
                "task_id": "aggregate__zh__lm1",
                "language": "zh",
                "lm": 1,
                "source_eval_task_ids": [task["task_id"] for task in eval_tasks],
                "expected_outputs": {
                    "instances_log": str(aggregate_dir / "instances.log"),
                    "runtime_log": str(aggregate_dir / "runtime.jsonl"),
                    "source_text": str(aggregate_dir / "source_text.txt"),
                    "reference": str(aggregate_dir / "reference.txt"),
                    "audio_manifest": str(aggregate_dir / "audio.json"),
                },
            }
            run = {"prepared_manifest": str(prepared_path), "eval_tasks": eval_tasks}
            instances_path, runtime_path = MODULE.merge_aggregate_inputs(
                run_manifest=run,
                aggregate_task=aggregate_task,
            )
            instances = MODULE._read_jsonl(instances_path)
            runtime = MODULE._read_jsonl(runtime_path)
            self.assertEqual([row["index"] for row in instances], [0, 1])
            self.assertEqual([row["prediction"] for row in instances], ["paper.one", "paper.two"])
            self.assertEqual([row["instance_index"] for row in runtime], [0, 1])
            self.assertEqual(
                (aggregate_dir / "source_text.txt").read_text(encoding="utf-8"),
                full_source_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(len(json.loads((aggregate_dir / "audio.json").read_text())), 2)


if __name__ == "__main__":
    unittest.main()
