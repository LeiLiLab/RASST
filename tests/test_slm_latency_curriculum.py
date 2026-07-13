from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SELECT = load_module(
    "select_latency_multiplier_rows",
    "code/rasst/slm/data_prep/select_latency_multiplier_rows.py",
)
ASSEMBLE = load_module(
    "assemble_latency_curriculum",
    "code/rasst/slm/data_prep/assemble_latency_curriculum.py",
)


def make_row(multiplier: int, suffix: str) -> dict:
    return {
        "audios": [f"{suffix}.wav"],
        "chunk_metadata": [{"multiplier": multiplier, "duration_sec": multiplier * 0.96}],
        "gt_terms_by_chunk": [[]],
        "messages": [
            {"role": "system", "content": "translate"},
            {"role": "user", "content": "<audio>"},
            {"role": "assistant", "content": suffix},
        ],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class SlmLatencyCurriculumTest(unittest.TestCase):
    def test_selects_only_all_lm1_rows_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            root = Path(temp_dir_raw)
            source = root / "source.jsonl"
            output = root / "lm1.jsonl"
            stats = root / "stats.json"
            mixed = make_row(1, "mixed")
            mixed["audios"].append("mixed2.wav")
            mixed["chunk_metadata"].append({"multiplier": 2, "duration_sec": 1.92})
            mixed["gt_terms_by_chunk"].append([])
            mixed["messages"].extend(
                [
                    {"role": "user", "content": "<audio>"},
                    {"role": "assistant", "content": "mixed2"},
                ]
            )
            write_jsonl(source, [make_row(1, "one"), make_row(2, "two"), mixed])

            summary = SELECT.run(
                Namespace(
                    input_jsonl=source,
                    output_jsonl=output,
                    stats_json=stats,
                    focus_multiplier=1,
                    match_policy="all",
                    expected_rows=1,
                )
            )

            self.assertEqual(summary["selected_rows"], 1)
            selected = json.loads(output.read_text(encoding="utf-8"))
            metadata = selected["latency_multiplier_selection"]
            self.assertEqual(metadata["source_line_number"], 1)
            self.assertEqual(metadata["row_multipliers"], [1])

    def test_assembles_base_and_verified_supplement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            root = Path(temp_dir_raw)
            base = root / "base.jsonl"
            supplement = root / "supplement.jsonl"
            output = root / "train.jsonl"
            stats = root / "stats.json"
            base_rows = [make_row(1, "base-one"), make_row(2, "base-two")]
            supplement_row = make_row(1, "base-one")
            supplement_row["messages"][-1]["content"] = "alternate target"
            supplement_row["latency_multiplier_selection"] = {
                "focus_multiplier": 1,
                "source_line_number": 1,
            }
            write_jsonl(base, base_rows)
            write_jsonl(supplement, [supplement_row])

            summary = ASSEMBLE.run(
                Namespace(
                    base_jsonl=base,
                    supplement_jsonl=[supplement],
                    output_jsonl=output,
                    stats_json=stats,
                    focus_multiplier=1,
                    base_focus_rows=1,
                    expected_base_rows=2,
                    expected_supplement_rows=1,
                )
            )

            self.assertEqual(summary["total_rows"], 3)
            self.assertAlmostEqual(summary["focus_row_rate_before"], 0.5)
            self.assertAlmostEqual(summary["focus_row_rate_after"], 2 / 3)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(rows[-1]["latency_curriculum"]["role"], "supplement")

    def test_rejects_supplement_without_selection_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            root = Path(temp_dir_raw)
            base = root / "base.jsonl"
            supplement = root / "supplement.jsonl"
            write_jsonl(base, [make_row(1, "base")])
            write_jsonl(supplement, [make_row(1, "supplement")])
            with self.assertRaisesRegex(ASSEMBLE.CurriculumError, "lacks latency"):
                ASSEMBLE.run(
                    Namespace(
                        base_jsonl=base,
                        supplement_jsonl=[supplement],
                        output_jsonl=root / "out.jsonl",
                        stats_json=root / "stats.json",
                        focus_multiplier=1,
                        base_focus_rows=1,
                        expected_base_rows=1,
                        expected_supplement_rows=1,
                    )
                )


if __name__ == "__main__":
    unittest.main()
