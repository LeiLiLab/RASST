#!/usr/bin/env python3
"""RASST main-result eval manifest validator and launcher."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


RAW_SHELL_PREFIX = "__RASST_RAW_SHELL__:"
METRIC_FIELDS = ("BLEU", "StreamLAAL", "StreamLAAL_CA", "TERM_ACC")
TERM_FIELDS = ("TERM_CORRECT", "TERM_TOTAL")
DOMAINS = {"acl_tagged_raw", "medicine_hardraw"}
LANGS = {"zh", "de", "ja"}
LMS = {"1", "2", "3", "4"}
CONFIG_DIFF_FIELDS = (
    "runner",
    "model_asset",
    "input_asset",
    "glossary_asset",
    "RAG_TOP_K_OVERRIDE",
    "RAG_SCORE_THRESHOLD_OVERRIDE",
    "RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE",
    "TERM_MAP_FORMAT_OVERRIDE",
    "EMPTY_TERM_MAP_POLICY_OVERRIDE",
    "SYSTEM_PROMPT_STYLE_OVERRIDE",
    "STRIP_OUTPUT_TAGS_OVERRIDE",
    "TERM_FCR_POLICY_OVERRIDE",
    "MAX_CACHE_SECONDS_OVERRIDE",
    "KEEP_CACHE_SECONDS_OVERRIDE",
    "MAX_CACHE_CHUNKS_OVERRIDE",
    "KEEP_CACHE_CHUNKS_OVERRIDE",
    "CACHE_POLICY_NOTE_OVERRIDE",
    "MAX_NEW_TOKENS_OVERRIDE",
    "MAX_NEW_TOKENS_POLICY_OVERRIDE",
    "VLLM_LIMIT_AUDIO_OVERRIDE",
    "VLLM_MAX_MODEL_LEN_OVERRIDE",
    "VLLM_DISABLE_CUSTOM_ALL_REDUCE",
    "VLLM_MOE_USE_DEEP_GEMM",
    "VLLM_USE_FUSED_MOE_GROUPED_TOPK",
    "GPU_MEMORY_UTILIZATION_OVERRIDE",
    "VLLM_TP_SIZE_OVERRIDE",
)


class RasstError(RuntimeError):
    pass


def repo_root() -> Path:
    root_text = os.environ.get("RASST_ROOT")
    if root_text:
        root = Path(root_text).expanduser()
        return root if root.is_absolute() else Path.cwd() / root
    return Path(__file__).resolve().parents[3]


def default_manifest_path(root: Path) -> Path:
    return root / "code/rasst/manifests/main_result_eval.global_cache30_30_20_20.json"


def output_root(root: Path) -> Path:
    return rel_or_abs(root, os.environ.get("RASST_OUTPUT_ROOT", "outputs"))


def log_root(root: Path) -> Path:
    return rel_or_abs(root, os.environ.get("RASST_LOG_ROOT", "logs"))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RasstError(f"JSON root must be an object: {path}")
    return data


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel_or_abs(root: Path, path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return root / path


def exists_nonempty(path: Path) -> bool:
    return path.exists() and (path.is_dir() or path.stat().st_size > 0)


def same_existing_path(path_a: Path, path_b: Path) -> bool:
    if str(path_a) == str(path_b):
        return True
    try:
        return path_a.exists() and path_b.exists() and path_a.resolve() == path_b.resolve()
    except OSError:
        return False


def legacy_allowed() -> bool:
    if os.environ.get("RASST_REQUIRE_LOCAL_ASSETS", "0") == "1":
        return False
    return os.environ.get("RASST_USE_LEGACY_PATHS", "1") == "1"


def hf_download_allowed() -> bool:
    return os.environ.get("RASST_AUTO_DOWNLOAD_ASSETS", "0") == "1"


def download_hf_asset(obj: Mapping[str, Any], *, root: Path, label: str) -> Path:
    repo_id = str(obj.get("hf_repo_id") or "")
    local_path = str(obj.get("local_path") or "")
    if not repo_id or not local_path:
        raise RasstError(f"{label} does not define both hf_repo_id and local_path.")
    target = rel_or_abs(root, local_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RasstError("Install huggingface_hub or run code/rasst/scripts/download_release_assets.sh first.") from exc

    revision = str(obj.get("hf_revision") or "main")
    asset_type = obj.get("type")
    print(f"[HF_DOWNLOAD] {label}: {repo_id}@{revision} -> {target}", file=sys.stderr)
    if asset_type == "file":
        filename = str(obj.get("hf_filename") or target.name)
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                filename=filename,
                local_dir=str(target.parent),
            )
        )
        if downloaded != target:
            downloaded.replace(target)
    elif asset_type == "hf_model_dir":
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            local_dir=str(target),
            ignore_patterns=[".git/*"],
        )
    else:
        raise RasstError(f"HF download is unsupported for {label} type={asset_type!r}.")
    return target


def resolve_dual_path(obj: Mapping[str, Any], *, root: Path, label: str, must_exist: bool = True) -> Path:
    env_name = str(obj.get("env") or "")
    candidates: List[Tuple[str, Path]] = []
    if env_name and os.environ.get(env_name):
        candidates.append((f"env:{env_name}", rel_or_abs(root, os.environ[env_name])))
    if obj.get("local_path"):
        candidates.append(("local_path", rel_or_abs(root, str(obj["local_path"]))))
    if legacy_allowed() and obj.get("legacy_path"):
        candidates.append(("legacy_path", rel_or_abs(root, str(obj["legacy_path"]))))

    checked: List[str] = []
    for source, path in candidates:
        if not must_exist or exists_nonempty(path):
            return path
        checked.append(f"{source}={path}")

    if must_exist and obj.get("hf_repo_id") and hf_download_allowed():
        path = download_hf_asset(obj, root=root, label=label)
        if exists_nonempty(path):
            return path
        checked.append(f"hf_repo_id={obj['hf_repo_id']} -> {path}")

    hint = "RASST_REQUIRE_LOCAL_ASSETS=1 disables legacy fallback." if not legacy_allowed() else ""
    if obj.get("hf_repo_id") and not hf_download_allowed():
        hint = f"{hint} Set RASST_AUTO_DOWNLOAD_ASSETS=1 to download this asset from Hugging Face.".strip()
    raise RasstError(f"Cannot resolve {label}; checked {checked}. {hint}".strip())


def artifact_map(manifest: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for item in manifest.get("artifacts", []):
        if not isinstance(item, dict) or not item.get("key"):
            raise RasstError("Every artifact must be an object with a key.")
        key = str(item["key"])
        if key in out:
            raise RasstError(f"Duplicate artifact key: {key}")
        out[key] = item
    return out


def metadata(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = manifest.get("metadata")
    if not isinstance(meta, dict):
        raise RasstError("Manifest metadata must be an object.")
    return meta


def all_cells(manifest: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    cells = metadata(manifest).get("cells")
    if not isinstance(cells, list):
        raise RasstError("metadata.cells must be a list.")
    return cells


def cell_id(cell: Mapping[str, Any]) -> str:
    return f"{cell['domain']}__{cell['lang']}__lm{cell['lm']}"


def selected_cells(
    manifest: Mapping[str, Any], *, domain: str, lang: str, lm: str
) -> List[Mapping[str, Any]]:
    cells = []
    for cell in all_cells(manifest):
        if domain != "all" and cell.get("domain") != domain:
            continue
        if lang != "all" and cell.get("lang") != lang:
            continue
        if lm != "all" and str(cell.get("lm")) != lm:
            continue
        cells.append(cell)
    if not cells:
        raise RasstError(f"No cells selected for domain={domain} lang={lang} lm={lm}")
    return cells


def parse_cell_overrides(items: Optional[Sequence[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise RasstError(f"Invalid --cell-override {item!r}; expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise RasstError(f"Invalid --cell-override {item!r}; key is empty.")
        overrides[key] = value
    return overrides


def parse_cache_seconds(value: Optional[str]) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 2 or not all(parts):
        raise RasstError(f"Invalid --cache-seconds {value!r}; expected MAX,KEEP.")
    try:
        max_seconds = float(parts[0])
        keep_seconds = float(parts[1])
    except ValueError as exc:
        raise RasstError(f"Invalid --cache-seconds {value!r}; values must be numeric.") from exc
    if max_seconds <= 0 or keep_seconds <= 0:
        raise RasstError("--cache-seconds values must be positive.")
    if keep_seconds > max_seconds:
        raise RasstError("--cache-seconds KEEP must be <= MAX.")
    return max_seconds, keep_seconds


def parse_cache_chunks_by_lm(value: Optional[str]) -> Optional[Dict[str, Tuple[int, int]]]:
    if value is None:
        return None
    parsed: Dict[str, Tuple[int, int]] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise RasstError(f"Invalid --cache-chunks-by-lm item {item!r}; expected lm:max[/keep].")
        lm_text, chunk_text = item.split(":", 1)
        lm_text = lm_text.strip()
        if lm_text not in LMS:
            raise RasstError(f"Invalid --cache-chunks-by-lm lm {lm_text!r}; allowed={sorted(LMS)}")
        chunk_parts = [part.strip() for part in chunk_text.split("/") if part.strip()]
        if len(chunk_parts) == 1:
            chunk_parts.append(chunk_parts[0])
        if len(chunk_parts) != 2:
            raise RasstError(f"Invalid --cache-chunks-by-lm chunks {chunk_text!r}; expected max or max/keep.")
        try:
            max_chunks = int(chunk_parts[0])
            keep_chunks = int(chunk_parts[1])
        except ValueError as exc:
            raise RasstError(f"Invalid --cache-chunks-by-lm chunks {chunk_text!r}; values must be integers.") from exc
        if max_chunks <= 0 or keep_chunks <= 0:
            raise RasstError("--cache-chunks-by-lm values must be positive.")
        if keep_chunks > max_chunks:
            raise RasstError("--cache-chunks-by-lm keep must be <= max.")
        parsed[lm_text] = (max_chunks, keep_chunks)
    if not parsed:
        raise RasstError("--cache-chunks-by-lm did not contain any mappings.")
    return parsed


def seconds_to_chunks(seconds: float, lm: int, rounding: str) -> int:
    raw = seconds / (0.96 * lm)
    if rounding == "ceil":
        chunks = math.ceil(raw)
    elif rounding == "floor":
        chunks = math.floor(raw)
    else:
        raise RasstError(f"Unsupported cache-seconds rounding: {rounding}")
    return max(1, chunks)


def apply_runtime_cell_overrides(
    cells: Sequence[Mapping[str, Any]],
    *,
    force_runner: Optional[str],
    cell_overrides: Mapping[str, str],
    fixed_cache_window_sec: Optional[float] = None,
    cache_seconds: Optional[Tuple[float, float]] = None,
    cache_chunks_by_lm: Optional[Mapping[str, Tuple[int, int]]] = None,
    cache_seconds_rounding: str = "floor",
    max_new_tokens_per_lm: Optional[int] = None,
) -> List[Mapping[str, Any]]:
    cache_policy_count = sum(
        item is not None for item in (fixed_cache_window_sec, cache_seconds, cache_chunks_by_lm)
    )
    if cache_policy_count > 1:
        raise RasstError("Use only one cache policy: --fixed-cache-window-sec, --cache-seconds, or --cache-chunks-by-lm.")
    out: List[Mapping[str, Any]] = []
    for cell in cells:
        copied = deepcopy(cell)
        if force_runner:
            copied["runner"] = force_runner
        derived_overrides: Dict[str, str] = {}
        if fixed_cache_window_sec is not None:
            lm = int(str(copied["lm"]))
            chunks = math.ceil(fixed_cache_window_sec / (0.96 * lm))
            derived_overrides["MAX_CACHE_CHUNKS_OVERRIDE"] = str(chunks)
            derived_overrides["KEEP_CACHE_CHUNKS_OVERRIDE"] = str(chunks)
            derived_overrides["CACHE_POLICY_NOTE_OVERRIDE"] = f"fixed_{fixed_cache_window_sec:g}s_window"
        if cache_seconds is not None:
            lm = int(str(copied["lm"]))
            max_seconds, keep_seconds = cache_seconds
            max_chunks = seconds_to_chunks(max_seconds, lm, cache_seconds_rounding)
            keep_chunks = seconds_to_chunks(keep_seconds, lm, cache_seconds_rounding)
            derived_overrides["MAX_CACHE_SECONDS_OVERRIDE"] = f"{max_seconds:g}"
            derived_overrides["KEEP_CACHE_SECONDS_OVERRIDE"] = f"{keep_seconds:g}"
            derived_overrides["MAX_CACHE_CHUNKS_OVERRIDE"] = str(max_chunks)
            derived_overrides["KEEP_CACHE_CHUNKS_OVERRIDE"] = str(keep_chunks)
            derived_overrides["CACHE_POLICY_NOTE_OVERRIDE"] = (
                f"seconds_{max_seconds:g}_{keep_seconds:g}_{cache_seconds_rounding}"
            )
        if cache_chunks_by_lm is not None:
            lm_text = str(copied["lm"])
            if lm_text not in cache_chunks_by_lm:
                raise RasstError(f"--cache-chunks-by-lm missing lm={lm_text} for selected cell {cell_id(copied)}")
            max_chunks, keep_chunks = cache_chunks_by_lm[lm_text]
            derived_overrides["MAX_CACHE_SECONDS_OVERRIDE"] = "0"
            derived_overrides["KEEP_CACHE_SECONDS_OVERRIDE"] = "0"
            derived_overrides["MAX_CACHE_CHUNKS_OVERRIDE"] = str(max_chunks)
            derived_overrides["KEEP_CACHE_CHUNKS_OVERRIDE"] = str(keep_chunks)
            derived_overrides["CACHE_POLICY_NOTE_OVERRIDE"] = (
                "chunks_by_lm_" + "_".join(
                    f"lm{lm_key}-{chunks[0]}-{chunks[1]}"
                    for lm_key, chunks in sorted(cache_chunks_by_lm.items())
                )
            )
        if max_new_tokens_per_lm is not None:
            lm = int(str(copied["lm"]))
            derived_overrides["MAX_NEW_TOKENS_OVERRIDE"] = str(max_new_tokens_per_lm * lm)
            derived_overrides["MAX_NEW_TOKENS_POLICY_OVERRIDE"] = "fixed"
        if cell_overrides:
            overrides = dict(copied.get("overrides") or {})
            overrides.update(derived_overrides)
            overrides.update(cell_overrides)
            copied["overrides"] = overrides
        elif derived_overrides:
            overrides = dict(copied.get("overrides") or {})
            overrides.update(derived_overrides)
            copied["overrides"] = overrides
        out.append(copied)
    return out


def parse_lm_list(value: Optional[str]) -> Optional[set[str]]:
    if not value:
        return None
    parsed = {item.strip() for item in value.split(",") if item.strip()}
    invalid = sorted(parsed - LMS)
    if invalid:
        raise RasstError(f"Invalid --lm-list values: {invalid}; allowed={sorted(LMS)}")
    if not parsed:
        raise RasstError("--lm-list did not contain any valid values.")
    return parsed


def read_tsv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise RasstError(f"Empty TSV: {path}")
    return rows


def read_single_result(path: Path) -> Dict[str, str]:
    rows = read_tsv_rows(path)
    return rows[-1]


def write_tsv(path: Path, rows: Sequence[Mapping[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def validate_result_tsv(path: Path) -> None:
    row = read_single_result(path)
    for key in METRIC_FIELDS:
        if key not in row or row[key] in {"", "NA"}:
            raise RasstError(f"Missing metric {key} in {path}")
        float(row[key])
    for key in TERM_FIELDS:
        if key in row and row[key] not in {"", "NA"}:
            int(float(row[key]))


def canonical_table_rows(manifest: Mapping[str, Any], root: Path) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    table_obj = metadata(manifest).get("canonical_table")
    if not isinstance(table_obj, dict):
        raise RasstError("metadata.canonical_table must be an object.")
    path = resolve_dual_path(table_obj, root=root, label="canonical_table")
    rows = read_tsv_rows(path)
    out: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in rows:
        if row.get("method") != "RASST":
            continue
        if row.get("dataset") not in DOMAINS:
            continue
        key = (row["dataset"], row["lang"], row["lm"])
        out[key] = row
    return out


def validate_manifest_shape(manifest: Mapping[str, Any], root: Path) -> None:
    cells = all_cells(manifest)
    if len(cells) != 24:
        raise RasstError(f"Expected exactly 24 cells, found {len(cells)}")
    seen = set()
    canonical = canonical_table_rows(manifest, root)
    for cell in cells:
        for key in ("domain", "lang", "lm", "legacy_event_id", "canonical_eval_results", "model_asset", "input_asset", "glossary_asset"):
            if key not in cell:
                raise RasstError(f"Cell missing {key}: {cell}")
        domain = str(cell["domain"])
        lang = str(cell["lang"])
        lm = str(cell["lm"])
        if domain not in DOMAINS or lang not in LANGS or lm not in LMS:
            raise RasstError(f"Invalid cell key: {cell}")
        key = (domain, lang, lm)
        if key in seen:
            raise RasstError(f"Duplicate cell key: {key}")
        seen.add(key)
        if key not in canonical:
            raise RasstError(f"Canonical table missing RASST row for {key}")
    expected = {(d, l, lm) for d in DOMAINS for l in LANGS for lm in LMS}
    missing = sorted(expected - seen)
    if missing:
        raise RasstError(f"Missing cells: {missing}")


def manifest_path_for_event(root: Path, event_id: str) -> Path:
    return root / "code/legacy/documents/code/simuleval/manifests/2026/05" / f"{event_id}.json"


def required_asset_path(assets: Mapping[str, Mapping[str, Any]], root: Path, key: str) -> Path:
    if key not in assets:
        raise RasstError(f"Unknown artifact key: {key}")
    return resolve_dual_path(assets[key], root=root, label=f"artifact:{key}")


def validate_model_dir(path: Path, key: str) -> None:
    required = ("config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer_config.json")
    for name in required:
        if not exists_nonempty(path / name):
            raise RasstError(f"Model asset {key} missing {name}: {path}")
    shards = sorted(path.glob("model-*.safetensors"))
    if len(shards) != 15:
        raise RasstError(f"Model asset {key} expected 15 safetensor shards, found {len(shards)}: {path}")


def input_files(input_root: Path, asset: Mapping[str, Any]) -> Dict[str, Path]:
    files = asset.get("files")
    if not isinstance(files, dict):
        raise RasstError(f"Input artifact {asset.get('key')} must define files.")
    return {str(name): input_root / str(rel) for name, rel in files.items()}


def validate_input_files(input_root: Path, asset: Mapping[str, Any]) -> None:
    for name, path in input_files(input_root, asset).items():
        if not exists_nonempty(path):
            raise RasstError(f"Input artifact {asset.get('key')} missing {name}: {path}")


def validate_assets(manifest: Mapping[str, Any], root: Path) -> None:
    assets = artifact_map(manifest)
    used_assets = {"eval_density_driver", "batch_vllm_driver", "retriever_hn1024"}
    canonical = canonical_table_rows(manifest, root)
    for cell in all_cells(manifest):
        key_tuple = (str(cell["domain"]), str(cell["lang"]), str(cell["lm"]))
        used_assets.add(str(cell["model_asset"]))
        used_assets.add(str(cell["input_asset"]))
        used_assets.add(str(cell["glossary_asset"]))
        eval_obj = cell["canonical_eval_results"]
        if not isinstance(eval_obj, dict):
            raise RasstError(f"canonical_eval_results must be an object for {cell_id(cell)}")
        source_path_text = canonical[key_tuple].get("source_path", "")
        if not source_path_text:
            raise RasstError(f"Frozen canonical table row lacks source_path for {key_tuple}")
        table_eval_path = rel_or_abs(root, source_path_text)
        if not exists_nonempty(table_eval_path):
            raise RasstError(f"Frozen canonical source_path missing for {key_tuple}: {table_eval_path}")
        validate_result_tsv(table_eval_path)
        if eval_obj.get("legacy_path") and not same_existing_path(
            rel_or_abs(root, str(eval_obj["legacy_path"])),
            table_eval_path,
        ):
            raise RasstError(
                f"Manifest canonical legacy_path does not match frozen TSV source_path for {key_tuple}: "
                f"{eval_obj['legacy_path']} != {table_eval_path}"
            )
        eval_path = resolve_dual_path(eval_obj, root=root, label=f"canonical_eval_results:{cell_id(cell)}")
        validate_result_tsv(eval_path)
        legacy_manifest = manifest_path_for_event(root, str(cell["legacy_event_id"]))
        if not exists_nonempty(legacy_manifest):
            raise RasstError(f"Missing legacy manifest for {cell_id(cell)}: {legacy_manifest}")

    for key in sorted(used_assets):
        path = required_asset_path(assets, root, key)
        asset = assets[key]
        if asset.get("type") == "hf_model_dir":
            validate_model_dir(path, key)
        if asset.get("type") == "input_dir":
            validate_input_files(path, asset)


def shell_env_assignments(env: Mapping[str, str]) -> str:
    parts = []
    for key, value in sorted(env.items()):
        if value.startswith(RAW_SHELL_PREFIX):
            raw_value = value[len(RAW_SHELL_PREFIX):]
            if not raw_value:
                raise RasstError(f"Raw shell assignment for {key} is empty.")
            parts.append(f"{key}={raw_value}")
        else:
            parts.append(f"{key}={shlex.quote(value)}")
    return " ".join(parts)


def shell_command(env: Mapping[str, str], argv: Sequence[str]) -> str:
    return f"{shell_env_assignments(env)} {' '.join(shlex.quote(a) for a in argv)}"


def merged_env_for_cell(
    manifest: Mapping[str, Any],
    assets: Mapping[str, Mapping[str, Any]],
    root: Path,
    cell: Mapping[str, Any],
    run_root: Path,
) -> Tuple[Dict[str, str], Path]:
    common = metadata(manifest).get("common_eval_config")
    if not isinstance(common, dict):
        raise RasstError("metadata.common_eval_config must be an object.")

    model = required_asset_path(assets, root, str(cell["model_asset"]))
    retriever = required_asset_path(assets, root, "retriever_hn1024")
    glossary = required_asset_path(assets, root, str(cell["glossary_asset"]))
    input_asset = assets[str(cell["input_asset"])]
    input_root = required_asset_path(assets, root, str(cell["input_asset"]))
    files = input_files(input_root, input_asset)
    lang = str(cell["lang"])
    lm = str(cell["lm"])
    cid = cell_id(cell)
    cell_output = run_root / "cells" / cid
    overrides = dict(cell.get("overrides") or {})
    tmp_stamp = "".join(ch for ch in run_root.parent.name if ch.isalnum())[-10:]
    tmp_profile = "".join(ch for ch in run_root.name if ch.isalnum())[:8]
    domain_short = "acl" if str(cell["domain"]).startswith("acl") else "med"
    eval_tmpdir = Path("/tmp") / f"rst_{tmp_stamp}_{tmp_profile}_{domain_short}_{lang}_lm{lm}"

    active_code_root = root / "code/rasst"
    env: Dict[str, str] = {
        "ROOT_DIR": str(active_code_root),
        "ROOT_DIR_OVERRIDE": str(active_code_root),
        "RASST_ROOT": str(root),
        "RASST_ACTIVE_CODE_ROOT": str(active_code_root),
        "PYTHONPATH": os.pathsep.join([str(active_code_root / "eval"), str(active_code_root), os.environ.get("PYTHONPATH", "")]),
        "MODEL_NAME_OVERRIDE": str(model),
        "RAG_MODEL_PATH_OVERRIDE": str(retriever),
        "LANG_CODE_OVERRIDE": lang,
        "LATENCY_MULTIPLIER_OVERRIDE": lm,
        "OUTPUT_BASE_OVERRIDE": str(cell_output),
        "CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE": os.environ.get("RASST_GPU_PAIR", "0,1"),
        "VLLM_TP_SIZE_OVERRIDE": os.environ.get("RASST_VLLM_TP_SIZE", "2"),
        "GPU_MEMORY_UTILIZATION_OVERRIDE": os.environ.get("RASST_GPU_MEMORY_UTILIZATION", "0.72"),
        "RAG_GPU_OVERRIDE": os.environ.get("RASST_RAG_GPU", "cuda:1"),
        "RAG_DEVICE_OVERRIDE": os.environ.get("RASST_RAG_DEVICE", os.environ.get("RASST_RAG_GPU", "cuda:1")),
        "GLOSSARY_PATH_OVERRIDE": str(glossary),
        "EVAL_GLOSSARY_PATH_OVERRIDE": str(files.get("eval_glossary", glossary)),
        "SRC_LIST_OVERRIDE": str(files["source"]),
        "TGT_LIST_OVERRIDE": str(files["target"]),
        "SOURCE_TEXT_FILE_OVERRIDE": str(files["source_text"]),
        "REF_FILE_OVERRIDE": str(files["ref"]),
        "AUDIO_YAML_OVERRIDE": str(files["audio"]),
        "EVAL_MODE_OVERRIDE": "acl6060",
        "INDEX_CACHE_DIR_OVERRIDE": str(run_root / "index_cache" / cid),
        "DENSITY_TAG": f"rasst_main_result_{cell['domain']}_{lang}_lm{lm}",
        "DENSITY_TAG_OVERRIDE": f"rasst_main_result_{cell['domain']}_{lang}_lm{lm}",
        "TERM_MAP_FORMAT_OVERRIDE": str(common["TERM_MAP_FORMAT"]),
        "EMPTY_TERM_MAP_POLICY_OVERRIDE": str(common["EMPTY_TERM_MAP_POLICY"]),
        "SYSTEM_PROMPT_STYLE_OVERRIDE": str(common["SYSTEM_PROMPT_STYLE"]),
        "RAG_PROMPT_POLICY_OVERRIDE": str(common["SYSTEM_PROMPT_STYLE"]),
        "STRIP_OUTPUT_TAGS_OVERRIDE": str(common["STRIP_OUTPUT_TAGS"]),
        "TERM_FCR_POLICY": str(common["TERM_FCR_POLICY"]),
        "TERM_FCR_POLICY_OVERRIDE": str(common["TERM_FCR_POLICY"]),
        "RAG_TOP_K_OVERRIDE": str(common["RAG_TOP_K"]),
        "RAG_SCORE_THRESHOLD_OVERRIDE": str(common["RAG_SCORE_THRESHOLD"]),
        "RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE": str(common["RAG_TIMELINE_LOOKBACK_SEC"]),
        "RAG_STREAMING_MODE_OVERRIDE": "timeline",
        "MAX_CACHE_SECONDS_OVERRIDE": "0",
        "KEEP_CACHE_SECONDS_OVERRIDE": "0",
        "MAX_CACHE_CHUNKS_OVERRIDE": str(overrides.get("MAX_CACHE_CHUNKS_OVERRIDE", overrides.get("MAX_CACHE_CHUNKS", 30))),
        "KEEP_CACHE_CHUNKS_OVERRIDE": str(overrides.get("KEEP_CACHE_CHUNKS_OVERRIDE", overrides.get("KEEP_CACHE_CHUNKS", 30))),
        "MAX_NEW_TOKENS_OVERRIDE": str(overrides.get("MAX_NEW_TOKENS_OVERRIDE", int(lm) * 40)),
        "CLEAN_OUTPUT_DIR_OVERRIDE": "1",
        "EVAL_TMPDIR_OVERRIDE": str(eval_tmpdir),
        "VLLM_LIMIT_AUDIO_OVERRIDE": str(overrides.get("VLLM_LIMIT_AUDIO_OVERRIDE", overrides.get("VLLM_LIMIT_AUDIO", 128))),
        "VLLM_LIMIT_AUDIO": str(overrides.get("VLLM_LIMIT_AUDIO_OVERRIDE", overrides.get("VLLM_LIMIT_AUDIO", 128))),
        "VLLM_MAX_MODEL_LEN_OVERRIDE": str(overrides.get("VLLM_MAX_MODEL_LEN_OVERRIDE", 12288)),
        "LMS_OVERRIDE": lm,
        "RUN_TAG_OVERRIDE": cid,
        "MAX_NEW_TOKENS_POLICY_OVERRIDE": str(overrides.get("MAX_NEW_TOKENS_POLICY_OVERRIDE", "fixed")),
        "GLOSSARY_TAG_OVERRIDE": str(files.get("glossary_tag", glossary.stem)),
        "DRY_RUN_OVERRIDE": "0",
        "WANDB_LOG_OVERRIDE": "0",
    }
    for key, value in overrides.items():
        env[str(key)] = str(value)
    if str(cell.get("runner", "serial_simuleval")) == "batch_vllm":
        env.update({
            "MAX_NUM_SEQS_OVERRIDE": str(overrides.get("MAX_NUM_SEQS_OVERRIDE", 5)),
            "SCHEDULER_BATCH_SIZE_OVERRIDE": str(overrides.get("SCHEDULER_BATCH_SIZE_OVERRIDE", 5)),
            "SCHEDULE_MODE_OVERRIDE": str(overrides.get("SCHEDULE_MODE_OVERRIDE", "round_robin")),
            "VLLM_ENFORCE_EAGER_OVERRIDE": str(overrides.get("VLLM_ENFORCE_EAGER_OVERRIDE", 1)),
            "VLLM_ENABLE_PREFIX_CACHING": str(overrides.get("VLLM_ENABLE_PREFIX_CACHING", 1)),
            "SAFETENSORS_LOAD_STRATEGY_OVERRIDE": str(overrides.get("SAFETENSORS_LOAD_STRATEGY_OVERRIDE", "lazy")),
            "MAX_MODEL_LEN_OVERRIDE": str(overrides.get("MAX_MODEL_LEN_OVERRIDE", env["VLLM_MAX_MODEL_LEN_OVERRIDE"])),
            "VLLM_DISABLE_CUSTOM_ALL_REDUCE": str(overrides.get("VLLM_DISABLE_CUSTOM_ALL_REDUCE", 1)),
            "RAG_DEVICE_OVERRIDE": str(overrides.get("RAG_DEVICE_OVERRIDE", "cuda:0")),
            "RAG_GPU_OVERRIDE": str(overrides.get("RAG_GPU_OVERRIDE", "cuda:0")),
            "RAG_BATCH_RETRIEVAL_OVERRIDE": str(overrides.get("RAG_BATCH_RETRIEVAL_OVERRIDE", 1)),
            "INDEX_BUILD_DEVICE_OVERRIDE": str(overrides.get("INDEX_BUILD_DEVICE_OVERRIDE", "cuda:0")),
            "LOG_ROOT_OVERRIDE": str(run_root / "batch_logs" / cid),
        })
    return env, cell_output


def command_for_cell(
    manifest: Mapping[str, Any],
    assets: Mapping[str, Mapping[str, Any]],
    root: Path,
    cell: Mapping[str, Any],
    run_root: Path,
) -> str:
    env, _ = merged_env_for_cell(manifest, assets, root, cell, run_root)
    runner = str(cell.get("runner", "serial_simuleval"))
    if runner == "batch_vllm":
        driver = required_asset_path(assets, root, "batch_vllm_driver")
    elif runner == "serial_simuleval":
        driver = required_asset_path(assets, root, "eval_density_driver")
    else:
        raise RasstError(f"Unsupported runner={runner} for {cell_id(cell)}")
    return shell_command(env, ["bash", str(driver)])


def command_list(
    manifest: Mapping[str, Any],
    root: Path,
    cells: Sequence[Mapping[str, Any]],
    run_root: Path,
) -> List[str]:
    assets = artifact_map(manifest)
    return [command_for_cell(manifest, assets, root, cell, run_root) for cell in cells]


def print_dry_run(
    manifest: Mapping[str, Any],
    root: Path,
    cells: Sequence[Mapping[str, Any]],
    *,
    run_root: Optional[Path] = None,
) -> None:
    if run_root is None:
        run_root = rel_or_abs(
            root,
            os.environ.get("RASST_DRY_RUN_OUTPUT_ROOT", str(output_root(root) / "main_result_eval/DRY_RUN")),
        )
    for cell, command in zip(cells, command_list(manifest, root, cells, run_root)):
        print(f"[DRY-RUN] {cell_id(cell)}")
        print(command)


def output_eval_path(run_root: Path, cell: Mapping[str, Any]) -> Path:
    paths = sorted((run_root / "cells" / cell_id(cell)).glob("**/eval_results.tsv"))
    if len(paths) != 1:
        raise RasstError(f"Expected one eval_results.tsv for {cell_id(cell)}, found {len(paths)}: {paths}")
    return paths[0]


def validate_completed_cell(run_root: Path, cell: Mapping[str, Any]) -> Path:
    eval_path = output_eval_path(run_root, cell)
    validate_result_tsv(eval_path)
    parent = eval_path.parent
    expected_rows = int(cell.get("expected_instance_rows", 5))
    for name in ("instances.log", "instances.strip_term.log"):
        path = parent / name
        if not exists_nonempty(path):
            raise RasstError(f"Missing output artifact for {cell_id(cell)}: {path}")
        count = sum(1 for line in path.open("r", encoding="utf-8", errors="replace") if line.strip())
        if count != expected_rows:
            raise RasstError(f"Expected {expected_rows} rows in {path}, found {count}")
    return eval_path


def float_or_none(value: str) -> Optional[float]:
    if value in {"", "NA", "N/A"}:
        return None
    return float(value)


def write_comparison(
    manifest: Mapping[str, Any],
    root: Path,
    cells: Sequence[Mapping[str, Any]],
    run_root: Path,
    *,
    strict_metrics: bool,
) -> None:
    canonical = canonical_table_rows(manifest, root)
    summary_rows: List[Dict[str, str]] = []
    compare_rows: List[Dict[str, str]] = []
    failures: List[str] = []
    metric_failures: List[str] = []
    for cell in cells:
        key = (str(cell["domain"]), str(cell["lang"]), str(cell["lm"]))
        expected = canonical[key]
        summary = {
            "domain": key[0],
            "lang": key[1],
            "lm": key[2],
            "eval_results": "",
            "status": "missing",
        }
        for field in METRIC_FIELDS + TERM_FIELDS:
            summary[field] = ""
        row = {
            "domain": key[0],
            "lang": key[1],
            "lm": key[2],
            "eval_results": "",
            "status": "missing",
            "error": "",
        }
        actual: Dict[str, str] = {}
        try:
            eval_path = validate_completed_cell(run_root, cell)
            actual = read_single_result(eval_path)
            summary["eval_results"] = str(eval_path)
            summary["status"] = "verified"
            row["eval_results"] = str(eval_path)
            row["status"] = "verified"
            for field in METRIC_FIELDS + TERM_FIELDS:
                summary[field] = actual.get(field, "")
        except Exception as exc:  # noqa: BLE001 - report malformed/missing artifacts, then fail after writing TSVs.
            msg = str(exc)
            row["error"] = msg
            summary["status"] = "failed_artifact_validation"
            failures.append(f"{key}: {msg}")
        summary_rows.append(summary)

        for field in METRIC_FIELDS + TERM_FIELDS:
            ev = expected.get(field, "")
            av = actual.get(field, "")
            row[f"expected_{field}"] = ev
            row[f"actual_{field}"] = av
            try:
                e = float_or_none(ev)
                a = float_or_none(av)
                row[f"delta_{field}"] = "" if e is None or a is None else f"{a - e:.6f}"
                if strict_metrics and e is not None and a is not None and abs(a - e) > 1e-4:
                    metric_failures.append(f"{key} {field} delta={a - e:.6f}")
            except ValueError:
                row[f"delta_{field}"] = "non_numeric"
        compare_rows.append(row)

    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "summary_all.tsv"
    comparison_path = run_root / "comparison_report.tsv"
    write_tsv(summary_path, summary_rows, list(summary_rows[0].keys()))
    write_tsv(comparison_path, compare_rows, list(compare_rows[0].keys()))
    print(f"summary_all={summary_path}")
    print(f"comparison_report={comparison_path}")
    if failures:
        raise RasstError("Output artifact validation failed: " + "; ".join(failures[:10]))
    if metric_failures:
        raise RasstError("Strict metric comparison failed: " + "; ".join(metric_failures[:10]))


def effective_config_for_cell(
    manifest: Mapping[str, Any],
    assets: Mapping[str, Mapping[str, Any]],
    root: Path,
    cell: Mapping[str, Any],
    run_root: Path,
) -> Dict[str, str]:
    env, _ = merged_env_for_cell(manifest, assets, root, cell, run_root)
    out = {
        "domain": str(cell["domain"]),
        "lang": str(cell["lang"]),
        "lm": str(cell["lm"]),
        "cell_id": cell_id(cell),
        "legacy_event_id": str(cell["legacy_event_id"]),
        "runner": str(cell.get("runner", "serial_simuleval")),
        "model_asset": str(cell["model_asset"]),
        "input_asset": str(cell["input_asset"]),
        "glossary_asset": str(cell["glossary_asset"]),
    }
    for field in CONFIG_DIFF_FIELDS:
        if field in out:
            continue
        out[field] = str(env.get(field, ""))
    return out


def write_config_report(
    manifest: Mapping[str, Any],
    root: Path,
    cells: Sequence[Mapping[str, Any]],
    run_root: Path,
) -> None:
    assets = artifact_map(manifest)
    rows = [effective_config_for_cell(manifest, assets, root, cell, run_root) for cell in cells]
    cell_fields = list(rows[0].keys())
    config_cells_path = run_root / "config_cells.tsv"
    write_tsv(config_cells_path, rows, cell_fields)

    diff_rows: List[Dict[str, str]] = []
    for field in CONFIG_DIFF_FIELDS:
        by_value: Dict[str, List[str]] = {}
        for row in rows:
            by_value.setdefault(row.get(field, ""), []).append(row["cell_id"])
        if len(by_value) <= 1:
            continue
        values = []
        for value, cell_ids in sorted(by_value.items()):
            values.append(f"{value} :: {','.join(cell_ids)}")
        diff_rows.append({
            "field": field,
            "unique_values": str(len(by_value)),
            "values_and_cells": " | ".join(values),
        })
    diff_path = run_root / "config_differences.tsv"
    write_tsv(diff_path, diff_rows, ("field", "unique_values", "values_and_cells"))

    md_path = run_root / "config_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# RASST Main Result Eval Config Report\n\n")
        f.write(f"- manifest: {default_manifest_path(root)}\n")
        f.write(f"- selected_cells: {len(rows)}\n")
        f.write(f"- config_cells_tsv: {config_cells_path}\n")
        f.write(f"- config_differences_tsv: {diff_path}\n\n")
        f.write("## Non-uniform Fields\n\n")
        if not diff_rows:
            f.write("All tracked config fields are uniform across selected cells.\n")
        else:
            for row in diff_rows:
                f.write(f"- `{row['field']}`: {row['unique_values']} values\n")
    print(f"config_cells={config_cells_path}")
    print(f"config_differences={diff_path}")
    print(f"config_report={md_path}")


def launch_detached(
    manifest: Mapping[str, Any],
    root: Path,
    cells: Sequence[Mapping[str, Any]],
    run_root: Path,
    *,
    domain: str,
    lang: str,
    lm: str,
    strict_metrics: bool,
    force_runner: Optional[str],
    cell_overrides: Mapping[str, str],
    lm_list: Optional[str],
    fixed_cache_window_sec: Optional[float],
    cache_seconds_text: Optional[str],
    cache_chunks_by_lm_text: Optional[str],
    cache_seconds_rounding: str,
    max_new_tokens_per_lm: Optional[int],
) -> None:
    if os.environ.get("RASST_ALLOW_LAUNCH", "0") != "1":
        raise RasstError("Set RASST_ALLOW_LAUNCH=1 to launch. Use --dry-run first.")
    log_dir = log_root(root) / "curated"
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_config_report(manifest, root, cells, run_root)
    stamp = run_root.name
    script_path = log_dir / f"{stamp}__eval_main_result.sh"
    out_log = log_dir / f"{stamp}__eval_main_result.out"
    err_log = log_dir / f"{stamp}__eval_main_result.err"
    pid_file = log_dir / f"{stamp}__eval_main_result.pid"
    compare_args = compare_args_for_run(
        root,
        run_root,
        domain=domain,
        lang=lang,
        lm=lm,
        strict_metrics=strict_metrics,
        force_runner=force_runner,
        cell_overrides=cell_overrides,
        lm_list=lm_list,
        fixed_cache_window_sec=fixed_cache_window_sec,
        cache_seconds_text=cache_seconds_text,
        cache_chunks_by_lm_text=cache_chunks_by_lm_text,
        cache_seconds_rounding=cache_seconds_rounding,
        max_new_tokens_per_lm=max_new_tokens_per_lm,
    )
    commands = command_list(manifest, root, cells, run_root)
    run_meta = {
        "event_id": manifest.get("event_id"),
        "launched_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "cells": [cell_id(cell) for cell in cells],
        "commands": commands,
    }
    (run_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    body = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(root))}",
        f"mkdir -p {shlex.quote(str(run_root))}",
    ]
    for cell, command in zip(cells, commands):
        body.append(f"echo '[CELL] {cell_id(cell)}'")
        body.append(command)
    body.append("echo '[COMPARE] writing summary_all.tsv and comparison_report.tsv'")
    body.append(" ".join(shlex.quote(arg) for arg in compare_args))
    body.append("echo '[DONE] main result eval complete'")
    script_path.write_text("\n".join(body) + "\n", encoding="utf-8")
    script_path.chmod(0o755)
    proc = subprocess.Popen(
        ["setsid", "bash", str(script_path)],
        cwd=str(root),
        stdin=subprocess.DEVNULL,
        stdout=out_log.open("wb"),
        stderr=err_log.open("wb"),
        start_new_session=False,
    )
    pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")
    print("status=launched_detached")
    print(f"run_root={run_root}")
    print(f"pid_file={pid_file}")
    print(f"stdout={out_log}")
    print(f"stderr={err_log}")
    print(f"script={script_path}")


def compare_args_for_run(
    root: Path,
    run_root: Path,
    *,
    domain: str,
    lang: str,
    lm: str,
    strict_metrics: bool,
    force_runner: Optional[str],
    cell_overrides: Mapping[str, str],
    lm_list: Optional[str],
    fixed_cache_window_sec: Optional[float],
    cache_seconds_text: Optional[str],
    cache_chunks_by_lm_text: Optional[str],
    cache_seconds_rounding: str,
    max_new_tokens_per_lm: Optional[int],
) -> List[str]:
    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--manifest",
        str(Path(os.environ.get("RASST_MAIN_RESULT_MANIFEST", default_manifest_path(root)))),
        "--compare-output",
        str(run_root),
    ]
    if domain != "all":
        args.extend(["--domain", domain])
    if lang != "all":
        args.extend(["--lang", lang])
    if lm != "all":
        args.extend(["--lm", lm])
    if strict_metrics:
        args.append("--strict-metrics")
    if lm_list:
        args.extend(["--lm-list", lm_list])
    if force_runner:
        args.extend(["--force-runner", force_runner])
    if fixed_cache_window_sec is not None:
        args.extend(["--fixed-cache-window-sec", f"{fixed_cache_window_sec:g}"])
    if cache_seconds_text is not None:
        args.extend(["--cache-seconds", cache_seconds_text])
        args.extend(["--cache-seconds-rounding", cache_seconds_rounding])
    if cache_chunks_by_lm_text is not None:
        args.extend(["--cache-chunks-by-lm", cache_chunks_by_lm_text])
    if max_new_tokens_per_lm is not None:
        args.extend(["--max-new-tokens-per-lm", str(max_new_tokens_per_lm)])
    for key, value in sorted(cell_overrides.items()):
        args.extend(["--cell-override", f"{key}={value}"])
    return args


def launch_sbatch(
    manifest: Mapping[str, Any],
    root: Path,
    cells: Sequence[Mapping[str, Any]],
    run_root: Path,
    *,
    domain: str,
    lang: str,
    lm: str,
    strict_metrics: bool,
    force_runner: Optional[str],
    cell_overrides: Mapping[str, str],
    lm_list: Optional[str],
    fixed_cache_window_sec: Optional[float],
    cache_seconds_text: Optional[str],
    cache_chunks_by_lm_text: Optional[str],
    cache_seconds_rounding: str,
    max_new_tokens_per_lm: Optional[int],
    prepare_only: bool = False,
) -> None:
    if os.environ.get("RASST_ALLOW_LAUNCH", "0") != "1":
        raise RasstError("Set RASST_ALLOW_LAUNCH=1 to launch. Use --dry-run first.")
    if not shutil_which("sbatch"):
        raise RasstError("sbatch is not available on PATH.")
    log_dir = log_root(root) / "curated"
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = run_root.name
    script_path = log_dir / f"{stamp}__eval_main_result_sbatch_array.sh"
    post_script_path = log_dir / f"{stamp}__eval_main_result_sbatch_post.sh"
    submit_stdout = log_dir / f"{stamp}__eval_main_result_sbatch_submit.out"
    submit_stderr = log_dir / f"{stamp}__eval_main_result_sbatch_submit.err"
    jobid_file = log_dir / f"{stamp}__eval_main_result.sbatch_jobid"
    cell_status_path = run_root / "cell_status.tsv"
    cell_scripts_dir = run_root / "cell_scripts"
    task_status_dir = run_root / "task_status"
    cell_scripts_dir.mkdir(parents=True, exist_ok=True)
    task_status_dir.mkdir(parents=True, exist_ok=True)
    partition = os.environ.get("RASST_SBATCH_PARTITION", "taurus")
    gres = os.environ.get("RASST_SBATCH_GRES", "gpu:2")
    cpus = os.environ.get("RASST_SBATCH_CPUS", "16")
    mem = os.environ.get("RASST_SBATCH_MEM", "128G")
    time_limit = os.environ.get("RASST_SBATCH_TIME", "08:00:00")
    job_name = os.environ.get("RASST_SBATCH_JOB_NAME", "rasst24_eval")[:64]
    array_limit = os.environ.get("RASST_SBATCH_ARRAY_LIMIT")
    if array_limit is None:
        array_limit = "4" if partition == "taurus" else "3" if partition == "aries" else "1"
    if len(cells) == 1:
        array_spec = "0-0"
    elif array_limit and array_limit != "0":
        array_spec = f"0-{len(cells) - 1}%{array_limit}"
    else:
        array_spec = f"0-{len(cells) - 1}"

    old_gpu_pair = os.environ.get("RASST_GPU_PAIR")
    os.environ["RASST_GPU_PAIR"] = RAW_SHELL_PREFIX + "${CUDA_VISIBLE_DEVICES:-0,1}"
    try:
        write_config_report(manifest, root, cells, run_root)
        commands = command_list(manifest, root, cells, run_root)
    finally:
        if old_gpu_pair is None:
            os.environ.pop("RASST_GPU_PAIR", None)
        else:
            os.environ["RASST_GPU_PAIR"] = old_gpu_pair

    compare_args = compare_args_for_run(
        root,
        run_root,
        domain=domain,
        lang=lang,
        lm=lm,
        strict_metrics=strict_metrics,
        force_runner=force_runner,
        cell_overrides=cell_overrides,
        lm_list=lm_list,
        fixed_cache_window_sec=fixed_cache_window_sec,
        cache_seconds_text=cache_seconds_text,
        cache_chunks_by_lm_text=cache_chunks_by_lm_text,
        cache_seconds_rounding=cache_seconds_rounding,
        max_new_tokens_per_lm=max_new_tokens_per_lm,
    )
    task_rows: List[Dict[str, str]] = []
    cell_script_paths: List[Path] = []
    for index, (cell, command) in enumerate(zip(cells, commands)):
        cid = cell_id(cell)
        cell_script = cell_scripts_dir / f"{index:03d}__{cid}.sh"
        cell_script.write_text(
            "\n".join([
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(str(root))}",
                f"echo '[COMMAND] {cid}'",
                command,
            ]) + "\n",
            encoding="utf-8",
        )
        cell_script.chmod(0o755)
        cell_script_paths.append(cell_script)
        task_rows.append({
            "task_index": str(index),
            "cell_id": cid,
            "cell_script": str(cell_script),
        })
    write_tsv(run_root / "task_manifest.tsv", task_rows, ("task_index", "cell_id", "cell_script"))

    run_meta = {
        "event_id": manifest.get("event_id"),
        "launch_backend": "sbatch_array",
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "log_dir": str(log_dir),
        "sbatch": {
            "partition": partition,
            "gres": gres,
            "cpus_per_task": cpus,
            "mem": mem,
            "time": time_limit,
            "job_name": job_name,
            "array": array_spec,
            "array_limit": array_limit,
        },
        "runtime_overrides": {
            "force_runner": force_runner or "",
            "cell_overrides": dict(cell_overrides),
            "lm_list": lm_list or "",
            "fixed_cache_window_sec": "" if fixed_cache_window_sec is None else f"{fixed_cache_window_sec:g}",
            "cache_seconds": cache_seconds_text or "",
            "cache_chunks_by_lm": cache_chunks_by_lm_text or "",
            "cache_seconds_rounding": cache_seconds_rounding,
            "max_new_tokens_per_lm": "" if max_new_tokens_per_lm is None else str(max_new_tokens_per_lm),
        },
        "cells": [cell_id(cell) for cell in cells],
        "commands": commands,
    }
    run_meta_path = run_root / "run_meta.json"
    run_meta_path.write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cell_id_array = " ".join(shlex.quote(cell_id(cell)) for cell in cells)
    cell_script_array = " ".join(shlex.quote(str(path)) for path in cell_script_paths)
    body = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --array={array_spec}",
        f"#SBATCH --chdir={root}",
        f"#SBATCH --output={log_dir}/%A_%a__eval_main_result.out",
        f"#SBATCH --error={log_dir}/%A_%a__eval_main_result.err",
        "set -uo pipefail",
        f"cd {shlex.quote(str(root))}",
        f"mkdir -p {shlex.quote(str(run_root))}",
        f"mkdir -p {shlex.quote(str(task_status_dir))}",
        f"CELL_IDS=({cell_id_array})",
        f"CELL_SCRIPTS=({cell_script_array})",
        "task_index=${SLURM_ARRAY_TASK_ID:-0}",
        "cid=${CELL_IDS[$task_index]}",
        "cell_script=${CELL_SCRIPTS[$task_index]}",
        "status_file=" + shlex.quote(str(task_status_dir)) + "/$(printf '%03d' \"$task_index\")__${cid}.tsv",
        "echo '[SLURM] host='$(hostname)' job='${SLURM_JOB_ID:-unknown}' task='$task_index' started='$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "echo '[SLURM] CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-unset}",
        "nvidia-smi || true",
        "df -h /mnt/taurus/data2 /mnt/taurus/data1 /mnt/gemini/data1 2>/dev/null || true",
        "echo '[CELL_START] '$cid' '$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "cell_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "set +e",
        "bash \"$cell_script\"",
        "cell_status=$?",
        "set -u",
        "cell_ended_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "if [[ $cell_status -eq 0 ]]; then cell_label=success; else cell_label=failed; fi",
        "printf '%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' \"$cid\" \"$cell_label\" \"$cell_started_at\" \"$cell_ended_at\" \"$cell_status\" \"$task_index\" \"$(hostname)\" > \"$status_file\"",
        "echo '[CELL_DONE] '$cid' status='$cell_status' '$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "exit \"$cell_status\"",
    ]
    script_path.write_text("\n".join(body) + "\n", encoding="utf-8")
    script_path.chmod(0o755)

    post_body = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={(job_name + '_post')[:64]}",
        f"#SBATCH --partition={partition}",
        "#SBATCH --cpus-per-task=2",
        "#SBATCH --mem=8G",
        "#SBATCH --time=00:30:00",
        f"#SBATCH --chdir={root}",
        f"#SBATCH --output={log_dir}/%j__eval_main_result_post.out",
        f"#SBATCH --error={log_dir}/%j__eval_main_result_post.err",
        "set -uo pipefail",
        f"cd {shlex.quote(str(root))}",
        "echo '[POSTPROCESS] job='${SLURM_JOB_ID:-unknown}' dependency complete at '$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        f"printf 'cell_id\\tstatus\\tstarted_at_utc\\tended_at_utc\\texit_code\\ttask_index\\thost\\n' > {shlex.quote(str(cell_status_path))}",
        f"if compgen -G {shlex.quote(str(task_status_dir / '*.tsv'))} > /dev/null; then cat {shlex.quote(str(task_status_dir))}/*.tsv | sort >> {shlex.quote(str(cell_status_path))}; fi",
        "set +e",
        " ".join(shlex.quote(arg) for arg in compare_args),
        "compare_status=$?",
        "set -u",
        f"echo \"$compare_status\" > {shlex.quote(str(run_root / 'compare_exit_code.txt'))}",
        "failed_cells=$(awk 'NR>1 && $2 != \"success\" {n++} END {print n+0}' " + shlex.quote(str(cell_status_path)) + ")",
        "missing_cells=$(( " + str(len(cells)) + " - $(awk 'NR>1 {n++} END {print n+0}' " + shlex.quote(str(cell_status_path)) + ") ))",
        "echo '[POSTPROCESS_DONE] compare_status='$compare_status' failed_cells='$failed_cells' missing_cells='$missing_cells",
        "if [[ -x \"$HOME/bin/codex-notify\" ]]; then",
        f"  \"$HOME/bin/codex-notify\" --delay 8 --detach --workspace {shlex.quote(str(root))} \"RASST eval finished: array=${{SLURM_ARRAY_JOB_ID:-unknown}} post=${{SLURM_JOB_ID:-unknown}} failed_cells=${{failed_cells}} missing_cells=${{missing_cells}} compare_status=${{compare_status}} run_root={run_root}\" || true",
        "fi",
        "if [[ \"$failed_cells\" != \"0\" || \"$missing_cells\" != \"0\" || \"$compare_status\" != \"0\" ]]; then exit 2; fi",
        "echo '[DONE] main result eval complete'",
    ]
    post_script_path.write_text("\n".join(post_body) + "\n", encoding="utf-8")
    post_script_path.chmod(0o755)

    if prepare_only:
        print("status=prepared_sbatch")
        print(f"run_root={run_root}")
        print(f"script={script_path}")
        print(f"post_script={post_script_path}")
        print(f"task_manifest={run_root / 'task_manifest.tsv'}")
        return

    proc = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    submit_stdout.write_text(proc.stdout, encoding="utf-8")
    submit_stderr.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RasstError(f"sbatch failed rc={proc.returncode}: {proc.stderr.strip()}")
    job_id = proc.stdout.strip().split(";")[0]
    post_proc = subprocess.run(
        ["sbatch", "--parsable", f"--dependency=afterany:{job_id}", str(post_script_path)],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    submit_stdout.write_text(proc.stdout + post_proc.stdout, encoding="utf-8")
    submit_stderr.write_text(proc.stderr + post_proc.stderr, encoding="utf-8")
    if post_proc.returncode != 0:
        raise RasstError(f"postprocess sbatch failed rc={post_proc.returncode}: {post_proc.stderr.strip()}")
    post_job_id = post_proc.stdout.strip().split(";")[0]
    jobid_file.write_text(f"array_job_id={job_id}\npost_job_id={post_job_id}\n", encoding="utf-8")
    run_meta["slurm_array_job_id"] = job_id
    run_meta["slurm_post_job_id"] = post_job_id
    run_meta["submitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_meta_path.write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("status=submitted_sbatch")
    print(f"run_root={run_root}")
    print(f"slurm_array_job_id={job_id}")
    print(f"slurm_post_job_id={post_job_id}")
    print(f"jobid_file={jobid_file}")
    print(f"script={script_path}")
    print(f"post_script={post_script_path}")
    print(f"slurm_stdout={log_dir}/%A_%a__eval_main_result.out")
    print(f"slurm_stderr={log_dir}/%A_%a__eval_main_result.err")
    print(f"post_stdout={log_dir}/%j__eval_main_result_post.out")
    print(f"post_stderr={log_dir}/%j__eval_main_result_post.err")


def shutil_which(name: str) -> Optional[str]:
    for item in os.environ.get("PATH", "").split(os.pathsep):
        path = Path(item) / name
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=None, help="Path to main-result manifest JSON.")
    p.add_argument("--domain", default="all", choices=sorted(DOMAINS | {"all"}))
    p.add_argument("--lang", default="all", choices=sorted(LANGS | {"all"}))
    p.add_argument("--lm", default="all", choices=sorted(LMS | {"all"}))
    p.add_argument("--lm-list", default=None, help="Comma-separated lm subset, e.g. 1,2,3. Use with --lm all.")
    p.add_argument("--dry-run", action="store_true", help="Print concrete commands without launching.")
    p.add_argument("--validate-only", action="store_true", help="Validate manifest, canonical artifacts, and launch inputs.")
    p.add_argument("--compare-output", default=None, help="Validate and compare a completed output root.")
    p.add_argument("--strict-metrics", action="store_true", help="Fail comparison when metrics differ from frozen canonical rows.")
    p.add_argument("--launch-backend", default="local", choices=("local", "sbatch"), help="Launch backend for real runs.")
    p.add_argument("--sbatch", action="store_true", help="Shortcut for --launch-backend sbatch.")
    p.add_argument("--prepare-only", action="store_true", help="Prepare generated launch scripts and metadata without submitting.")
    p.add_argument("--run-root", default=None, help="Exact output root. Relative paths are resolved under RASST_ROOT.")
    p.add_argument("--force-runner", default=None, choices=("serial_simuleval", "batch_vllm"), help="Temporarily override selected cell runner without editing the manifest.")
    p.add_argument("--cell-override", action="append", default=[], help="Temporarily override selected cell config, as KEY=VALUE. Repeatable.")
    p.add_argument("--fixed-cache-window-sec", type=float, default=None, help="Set MAX/KEEP cache chunks per cell as ceil(seconds / (0.96 * lm)).")
    p.add_argument("--cache-seconds", default=None, help="Set MAX/KEEP cache chunks from separate seconds windows, as MAX,KEEP.")
    p.add_argument("--cache-chunks-by-lm", default=None, help="Set MAX/KEEP cache chunks by lm, e.g. 1:30,2:15,3:10,4:8 or 1:30/30.")
    p.add_argument("--cache-seconds-rounding", default="floor", choices=("floor", "ceil"), help="Rounding policy for --cache-seconds conversion to chunks.")
    p.add_argument("--max-new-tokens-per-lm", type=int, default=None, help="Set MAX_NEW_TOKENS_OVERRIDE per cell as this value times lm.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    manifest_path = Path(args.manifest) if args.manifest else default_manifest_path(root)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    os.environ.setdefault("RASST_MAIN_RESULT_MANIFEST", str(manifest_path))
    manifest = load_json(manifest_path)
    validate_manifest_shape(manifest, root)
    validate_assets(manifest, root)
    cell_overrides = parse_cell_overrides(args.cell_override)
    cache_seconds = parse_cache_seconds(args.cache_seconds)
    cache_chunks_by_lm = parse_cache_chunks_by_lm(args.cache_chunks_by_lm)
    lm_list = parse_lm_list(args.lm_list)
    if lm_list and args.lm != "all":
        raise RasstError("Use either --lm or --lm-list, not both.")
    cells = apply_runtime_cell_overrides(
        [
            cell
            for cell in selected_cells(manifest, domain=args.domain, lang=args.lang, lm=args.lm)
            if lm_list is None or str(cell["lm"]) in lm_list
        ],
        force_runner=args.force_runner,
        cell_overrides=cell_overrides,
        fixed_cache_window_sec=args.fixed_cache_window_sec,
        cache_seconds=cache_seconds,
        cache_chunks_by_lm=cache_chunks_by_lm,
        cache_seconds_rounding=args.cache_seconds_rounding,
        max_new_tokens_per_lm=args.max_new_tokens_per_lm,
    )
    if not cells:
        raise RasstError(f"No cells selected after --lm-list filtering: {args.lm_list}")
    run_root_arg = rel_or_abs(root, args.run_root) if args.run_root else None

    if args.validate_only:
        print(f"status=validated cells=24 selected={len(cells)} manifest={manifest_path}")
        return 0
    if args.compare_output:
        compare_root = rel_or_abs(root, args.compare_output)
        write_config_report(manifest, root, cells, compare_root)
        write_comparison(manifest, root, cells, compare_root, strict_metrics=args.strict_metrics)
        return 0
    if args.dry_run:
        print(f"status=dry_run cells=24 selected={len(cells)} manifest={manifest_path}")
        print_dry_run(manifest, root, cells, run_root=run_root_arg)
        return 0

    run_root = run_root_arg or output_root(root) / "main_result_eval" / utc_stamp()
    backend = "sbatch" if args.sbatch else args.launch_backend
    if backend == "sbatch":
        launch_sbatch(
            manifest,
            root,
            cells,
            run_root,
            domain=args.domain,
            lang=args.lang,
            lm=args.lm,
            strict_metrics=args.strict_metrics,
            force_runner=args.force_runner,
            cell_overrides=cell_overrides,
            lm_list=args.lm_list,
            fixed_cache_window_sec=args.fixed_cache_window_sec,
            cache_seconds_text=args.cache_seconds,
            cache_chunks_by_lm_text=args.cache_chunks_by_lm,
            cache_seconds_rounding=args.cache_seconds_rounding,
            max_new_tokens_per_lm=args.max_new_tokens_per_lm,
            prepare_only=args.prepare_only,
        )
    else:
        launch_detached(
            manifest,
            root,
            cells,
            run_root,
            domain=args.domain,
            lang=args.lang,
            lm=args.lm,
            strict_metrics=args.strict_metrics,
            force_runner=args.force_runner,
            cell_overrides=cell_overrides,
            lm_list=args.lm_list,
            fixed_cache_window_sec=args.fixed_cache_window_sec,
            cache_seconds_text=args.cache_seconds,
            cache_chunks_by_lm_text=args.cache_chunks_by_lm,
            cache_seconds_rounding=args.cache_seconds_rounding,
            max_new_tokens_per_lm=args.max_new_tokens_per_lm,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RasstError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
