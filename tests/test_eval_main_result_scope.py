from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "code/rasst/tools/eval_main_result.py"
SPEC = importlib.util.spec_from_file_location("eval_main_result", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EVAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVAL
SPEC.loader.exec_module(EVAL)


def manifest(scope: str | None) -> dict:
    metadata = {
        "common_eval_config": {},
        "cells": [
            {"domain": "acl_tagged_raw", "lang": "ja", "lm": 1},
            {"domain": "acl_tagged_raw", "lang": "ja", "lm": 2},
        ],
    }
    if scope is not None:
        metadata["asset_validation_scope"] = scope
    return {"metadata": metadata}


class EvalMainResultScopeTest(unittest.TestCase):
    def test_defaults_to_all_cells(self) -> None:
        data = manifest(None)
        selected = [data["metadata"]["cells"][0]]
        self.assertEqual(EVAL.asset_validation_cells(data, selected), data["metadata"]["cells"])

    def test_selected_scope_limits_validation(self) -> None:
        data = manifest("selected")
        selected = [data["metadata"]["cells"][0]]
        self.assertEqual(EVAL.asset_validation_cells(data, selected), selected)

    def test_rejects_unknown_scope(self) -> None:
        data = manifest("partial")
        with self.assertRaisesRegex(EVAL.RasstError, "asset_validation_scope"):
            EVAL.asset_validation_cells(data, [])


if __name__ == "__main__":
    unittest.main()
