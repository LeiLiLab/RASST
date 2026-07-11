from __future__ import annotations

import csv
import unittest
from collections import Counter
from pathlib import Path


MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "docs/results/rebuttal_2026/xcomet_input_manifest.taurus.tsv"
)


class XCometRebuttalManifestTest(unittest.TestCase):
    def test_manifest_is_complete_and_host_qualified(self) -> None:
        with MANIFEST.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(len(rows), 32)
        pairs = Counter((row["dataset"], row["lang"], row["lm"]) for row in rows)
        self.assertEqual(len(pairs), 16)
        self.assertTrue(all(count == 2 for count in pairs.values()))
        self.assertTrue(
            all(
                row[field].startswith("/mnt/taurus/")
                for row in rows
                for field in ("instances_log", "source_text", "reference", "audio_yaml")
            )
        )


if __name__ == "__main__":
    unittest.main()
