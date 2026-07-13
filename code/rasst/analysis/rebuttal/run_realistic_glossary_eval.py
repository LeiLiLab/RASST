#!/usr/bin/env python3
"""Plan, launch, and aggregate the ACL realistic-glossary RASST evaluation.

All experiment inputs and parameters are supplied as command-line arguments and
frozen in a run manifest.  ``start`` creates a detached worker with persistent
logs and a PID file; the long-running worker is not intended for foreground use.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


CACHE_GROUPS = (((1, 2), 30), ((3, 4), 20))
TEXT_MODEL_ID = "BAAI/bge-m3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty JSONL: {path}")
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"No JSON rows in {path}")
    return rows


def _require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty file: {path}")
    return path.absolute()


def _require_dir(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing directory: {path}")
    return path.absolute()


def _safe_tag(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError(f"Cannot build a safe tag from {value!r}")
    return cleaned


def _parse_model_assignments(values: Sequence[str]) -> Dict[str, Path]:
    output: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LANG=PATH for --model: {value!r}")
        language, path_text = value.split("=", 1)
        language = language.strip()
        if language in output:
            raise ValueError(f"Duplicate model language: {language}")
        output[language] = _require_dir(Path(path_text))
    return output


def _directory_signature(path: Path) -> Dict[str, Any]:
    config = _require_file(path / "config.json")
    index_candidates = sorted(path.glob("*.safetensors.index.json"))
    shard_paths = sorted(path.glob("*.safetensors"))
    if not index_candidates or not shard_paths:
        raise FileNotFoundError(f"Model directory lacks safetensor index or shards: {path}")
    return {
        "path": str(path.absolute()),
        "config_sha256": sha256_file(config),
        "safetensors_index": str(index_candidates[0].absolute()),
        "safetensors_index_sha256": sha256_file(index_candidates[0]),
        "shards": [
            {"name": shard.name, "bytes": shard.stat().st_size}
            for shard in shard_paths
        ],
    }


def _python_constant(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Python constant {name} not found in {path}")


def _validate_prepared_manifest(path: Path) -> Dict[str, Any]:
    manifest = _load_json(path)
    if not isinstance(manifest, dict) or manifest.get("kind") != (
        "rasst_acl_realistic_paper_glossary_prepared"
    ):
        raise ValueError(f"Unsupported prepared manifest: {path}")
    runtime_policy = manifest.get("runtime_glossary_policy")
    if runtime_policy is None:
        if manifest.get("gemini_model") != "gemini-2.5-flash":
            raise ValueError("Legacy prepared manifest is not pinned to gemini-2.5-flash")
    elif not isinstance(runtime_policy, str) or not runtime_policy.strip():
        raise ValueError("Prepared manifest has an invalid runtime_glossary_policy")
    runtime_tag = manifest.get("runtime_glossary_tag")
    if runtime_tag is not None and (
        not isinstance(runtime_tag, str) or _safe_tag(runtime_tag) != runtime_tag
    ):
        raise ValueError("Prepared manifest has an invalid runtime_glossary_tag")
    separation = manifest.get("separation_policy")
    if not isinstance(separation, dict) or separation.get(
        "gold_used_to_build_runtime_glossary"
    ) is not False:
        raise ValueError("Prepared manifest does not guarantee glossary/eval separation")
    gold = manifest.get("fixed_raw_gold_eval_glossary")
    if not isinstance(gold, dict):
        raise ValueError("Prepared manifest has no fixed raw-gold glossary")
    if sha256_file(Path(gold["path"])) != gold.get("sha256"):
        raise ValueError("Fixed raw-gold glossary hash mismatch")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Prepared manifest has no shards")
    for shard in shards:
        glossary = shard.get("runtime_glossary")
        files = shard.get("files")
        if not isinstance(glossary, dict) or not isinstance(files, dict):
            raise ValueError(f"Malformed prepared shard: {shard!r}")
        for record in [glossary, *files.values()]:
            artifact = _require_file(Path(record["path"]))
            if sha256_file(artifact) != record.get("sha256"):
                raise ValueError(f"Prepared artifact hash mismatch: {artifact}")
    return manifest


def _runtime_glossary_identity(prepared: Mapping[str, Any]) -> Tuple[str, str]:
    policy = prepared.get("runtime_glossary_policy")
    if isinstance(policy, str) and policy.strip():
        tag = str(prepared.get("runtime_glossary_tag") or policy)
        return policy, _safe_tag(tag)
    return "paper-specific Gemini-2.5-flash extraction", "gemini25"


def _selected_cache_groups(lms: Sequence[int]) -> Tuple[Tuple[Tuple[int, ...], int], ...]:
    requested = tuple(int(lm) for lm in lms)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("--lms must contain one or more unique values")
    unsupported = sorted(set(requested) - {1, 2, 3, 4})
    if unsupported:
        raise ValueError(f"Unsupported latency multipliers: {unsupported}")
    groups: List[Tuple[Tuple[int, ...], int]] = []
    for group_lms, cache_chunks in CACHE_GROUPS:
        selected = tuple(lm for lm in group_lms if lm in requested)
        if selected:
            groups.append((selected, cache_chunks))
    return tuple(groups)


def _run_checked(command: Sequence[str], *, cwd: Path, capture: bool = False) -> str:
    print("[RUN] " + " ".join(command), flush=True)
    result = subprocess.run(
        list(command),
        cwd=str(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def _resolve_index(
    *,
    python_bin: Path,
    cache_tool: Path,
    builder: Path,
    retriever_checkpoint: Path,
    glossary: Path,
    glossary_tag: str,
    index_cache_dir: Path,
    text_lora_rank: int,
    text_lora_alpha: int,
) -> Tuple[Dict[str, Any], List[str]]:
    command = [
        str(python_bin),
        str(cache_tool),
        "resolve",
        "--model-path",
        str(retriever_checkpoint),
        "--glossary-path",
        str(glossary),
        "--builder-script",
        str(builder),
        "--glossary-tag",
        glossary_tag,
        "--text-lora-rank",
        str(text_lora_rank),
        "--text-lora-alpha",
        str(text_lora_alpha),
        "--text-model-id",
        TEXT_MODEL_ID,
        "--checkpoint-hash-mode",
        "stat",
        "--glossary-hash-mode",
        "sha256",
        "--builder-hash-mode",
        "sha256",
        "--cache-dir",
        str(index_cache_dir),
        "--output-format",
        "json",
    ]
    output = _run_checked(command, cwd=builder.parents[3], capture=True)
    record = json.loads(output)
    return record, command


def _expected_eval_dir(
    *,
    output_base: Path,
    language: str,
    density_tag: str,
    lm: int,
    rag_top_k: int,
    score_threshold: float,
    glossary_tag: str,
) -> Path:
    name = (
        f"d{density_tag}_lm{lm}_k{rag_top_k}_th{score_threshold}_g{glossary_tag}"
    )
    return output_base / language / name


def build_run_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    prepared_path = _require_file(args.prepared_manifest)
    prepared = _validate_prepared_manifest(prepared_path)
    runtime_glossary_policy, runtime_glossary_tag = _runtime_glossary_identity(prepared)
    selected_cache_groups = _selected_cache_groups(
        getattr(args, "lms", (1, 2, 3, 4))
    )
    selected_lms = tuple(
        lm for group_lms, _ in selected_cache_groups for lm in group_lms
    )
    repo_root = _require_dir(args.repo_root)
    python_bin = _require_file(args.python_bin)
    retriever = _require_file(args.retriever_checkpoint)
    mwer_bin = _require_file(args.mwer_segmenter_bin)
    if not os.access(mwer_bin, os.X_OK):
        raise PermissionError(f"mwerSegmenter is not executable: {mwer_bin}")
    models = _parse_model_assignments(args.model)
    languages = list(prepared["languages"])
    if set(models) != set(languages):
        raise ValueError(f"Model language set mismatch: models={sorted(models)}, data={languages}")

    analysis_root = repo_root / "code/rasst"
    builder = _require_file(analysis_root / "retriever/build_maxsim_index.py")
    if _python_constant(builder, "TEXT_MODEL_ID") != TEXT_MODEL_ID:
        raise ValueError(f"Unexpected text model in index builder: {builder}")
    cache_tool = _require_file(analysis_root / "eval/tools/maxsim_index_cache_key.py")
    evaluator = _require_file(analysis_root / "eval/src/batched_vllm_rag_eval.py")
    offline_evaluator = _require_file(
        analysis_root / "analysis/rebuttal/score_merged_realistic_glossary.py"
    )
    index_cache_dir = args.index_cache_dir.absolute()
    output_dir = args.output_dir.absolute()
    index_cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for shard in prepared["shards"]:
        key = (shard["language"], shard["paper_id"])
        if key in shard_by_key:
            raise ValueError(f"Duplicate prepared shard: {key}")
        shard_by_key[key] = shard

    index_tasks: List[Dict[str, Any]] = []
    index_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for language in languages:
        for paper_id in prepared["paper_ids"]:
            shard = shard_by_key[(language, paper_id)]
            glossary = Path(shard["runtime_glossary"]["path"])
            glossary_tag = _safe_tag(
                f"{runtime_glossary_tag}_{paper_id}_{language}"
            )
            resolved, resolve_command = _resolve_index(
                python_bin=python_bin,
                cache_tool=cache_tool,
                builder=builder,
                retriever_checkpoint=retriever,
                glossary=glossary,
                glossary_tag=glossary_tag,
                index_cache_dir=index_cache_dir,
                text_lora_rank=args.text_lora_rank,
                text_lora_alpha=args.text_lora_alpha,
            )
            index_path = Path(resolved["index_path"])
            sidecar_path = Path(resolved["manifest_path"])
            build_command = [
                str(python_bin),
                str(builder),
                "--model-path",
                str(retriever),
                "--glossary-path",
                str(glossary),
                "--output-path",
                str(index_path),
                "--device",
                args.index_device,
                "--text-lora-rank",
                str(args.text_lora_rank),
                "--text-lora-alpha",
                str(args.text_lora_alpha),
            ]
            finalize_command = [
                str(python_bin),
                str(cache_tool),
                "finalize",
                "--model-path",
                str(retriever),
                "--glossary-path",
                str(glossary),
                "--builder-script",
                str(builder),
                "--glossary-tag",
                glossary_tag,
                "--text-model-id",
                TEXT_MODEL_ID,
                "--text-lora-rank",
                str(args.text_lora_rank),
                "--text-lora-alpha",
                str(args.text_lora_alpha),
                "--checkpoint-hash-mode",
                "stat",
                "--glossary-hash-mode",
                "sha256",
                "--builder-hash-mode",
                "sha256",
                "--index-path",
                str(index_path),
                "--manifest-path",
                str(sidecar_path),
            ]
            task = {
                "task_id": f"index__{language}__{paper_id}",
                "language": language,
                "paper_id": paper_id,
                "glossary": str(glossary),
                "glossary_sha256": sha256_file(glossary),
                "glossary_tag": glossary_tag,
                "index_path": str(index_path),
                "index_manifest_path": str(sidecar_path),
                "cache_key": resolved["cache_key"],
                "resolve_command": resolve_command,
                "build_command": build_command,
                "finalize_command": finalize_command,
            }
            index_tasks.append(task)
            index_by_key[(language, paper_id)] = task

    eval_tasks: List[Dict[str, Any]] = []
    for language in languages:
        for paper_id in prepared["paper_ids"]:
            shard = shard_by_key[(language, paper_id)]
            index_task = index_by_key[(language, paper_id)]
            for lms, cache_chunks in selected_cache_groups:
                group_tag = "lm" + "_".join(str(lm) for lm in lms)
                task_output_base = output_dir / "shards" / language / paper_id / group_tag
                command = [
                    str(python_bin),
                    str(evaluator),
                    "--source-list",
                    shard["files"]["source_list"]["path"],
                    "--target-list",
                    shard["files"]["target_list"]["path"],
                    "--source-text-file",
                    shard["files"]["source_text"]["path"],
                    "--ref-file",
                    shard["files"]["ref"]["path"],
                    "--audio-yaml",
                    shard["files"]["audio_yaml"]["path"],
                    "--glossary",
                    index_task["glossary"],
                    "--eval-glossary",
                    prepared["fixed_raw_gold_eval_glossary"]["path"],
                    "--output-base",
                    str(task_output_base),
                    "--run-tag",
                    f"realistic_{runtime_glossary_tag}_{language}_{paper_id}_{group_tag}",
                    "--density-tag",
                    args.density_tag,
                    "--glossary-tag",
                    index_task["glossary_tag"],
                    "--lang-code",
                    language,
                    "--source-lang",
                    "English",
                    "--lms",
                    *[str(lm) for lm in lms],
                    "--model-name",
                    str(models[language]),
                    "--vllm-tp-size",
                    str(args.vllm_tp_size),
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization),
                    "--max-model-len",
                    str(args.max_model_len),
                    "--max-num-seqs",
                    str(args.max_num_seqs),
                    "--scheduler-batch-size",
                    str(args.scheduler_batch_size),
                    "--schedule-mode",
                    args.schedule_mode,
                    "--vllm-limit-audio",
                    str(args.vllm_limit_audio),
                    "--enable-prefix-caching",
                    "1",
                    "--vllm-enforce-eager",
                    str(args.vllm_enforce_eager),
                    "--safetensors-load-strategy",
                    args.safetensors_load_strategy,
                    "--disable-custom-all-reduce",
                    str(args.disable_custom_all_reduce),
                    "--max-cache-chunks",
                    str(cache_chunks),
                    "--keep-cache-chunks",
                    str(cache_chunks),
                    "--max-cache-seconds",
                    "0",
                    "--keep-cache-seconds",
                    "0",
                    "--min-cache-chunks",
                    "1",
                    "--temperature",
                    str(args.temperature),
                    "--top-p",
                    str(args.top_p),
                    "--top-k",
                    str(args.top_k_decode),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--max-new-tokens-policy",
                    "lm_scaled",
                    "--seed",
                    str(args.seed),
                    "--rag-model-path",
                    str(retriever),
                    "--rag-index-path",
                    index_task["index_path"],
                    "--rag-device",
                    args.rag_device,
                    "--rag-top-k",
                    str(args.rag_top_k),
                    "--rag-score-threshold",
                    str(args.rag_score_threshold),
                    "--rag-timeline-lookback-sec",
                    str(args.rag_timeline_lookback_sec),
                    "--rag-lora-r",
                    str(args.rag_lora_rank),
                    "--rag-text-lora-r",
                    str(args.text_lora_rank),
                    "--rag-batch-retrieval",
                    "1",
                    "--term-map-format",
                    "plain",
                    "--empty-term-map-policy",
                    "omit",
                    "--rag-prompt-policy",
                    "given_chunks",
                    "--norag-prompt-policy",
                    "term_map_if_available",
                    "--offline-eval-script",
                    str(offline_evaluator),
                    "--eval-mode",
                    "acl6060",
                    "--strip-output-tags",
                    "term_t",
                    "--term-fcr-policy",
                    "term_map_source_ref_negative_sentence",
                    "--skip-offline-eval",
                ]
                expected: Dict[str, Dict[str, str]] = {}
                for lm in lms:
                    out_dir = _expected_eval_dir(
                        output_base=task_output_base,
                        language=language,
                        density_tag=args.density_tag,
                        lm=lm,
                        rag_top_k=args.rag_top_k,
                        score_threshold=args.rag_score_threshold,
                        glossary_tag=index_task["glossary_tag"],
                    )
                    expected[str(lm)] = {
                        "output_dir": str(out_dir),
                        "instances_log": str(out_dir / "instances.log"),
                        "runtime_log": str(
                            out_dir / f"runtime_omni_vllm_maxsim_rag_batched_lm{lm}.jsonl"
                        ),
                    }
                eval_tasks.append(
                    {
                        "task_id": f"eval__{language}__{paper_id}__{group_tag}",
                        "language": language,
                        "paper_id": paper_id,
                        "lms": list(lms),
                        "cache_chunks": cache_chunks,
                        "index_task_id": index_task["task_id"],
                        "command": command,
                        "expected_outputs": expected,
                    }
                )

    aggregate_tasks: List[Dict[str, Any]] = []
    full_gold = prepared["fixed_raw_gold_eval_glossary"]["path"]
    for language in languages:
        for lm in selected_lms:
            aggregate_dir = output_dir / "aggregate" / language / f"lm{lm}"
            source_tasks = [
                task
                for paper_id in prepared["paper_ids"]
                for task in eval_tasks
                if task["language"] == language
                and task["paper_id"] == paper_id
                and lm in task["lms"]
            ]
            if len(source_tasks) != len(prepared["paper_ids"]):
                raise AssertionError(f"Expected one source task per paper for {language}/lm{lm}")
            offline_command = [
                str(python_bin),
                str(offline_evaluator),
                "--instances-log",
                str(aggregate_dir / "instances.log"),
                "--source-file",
                str(aggregate_dir / "source_text.txt"),
                "--reference-file",
                str(aggregate_dir / "reference.txt"),
                "--audio-manifest",
                str(aggregate_dir / "audio.json"),
                "--glossary",
                full_gold,
                "--target-language",
                language,
                "--latency-unit",
                "word" if language == "de" else "char",
                "--sacrebleu-tokenizer",
                {"zh": "zh", "de": "13a", "ja": "ja-mecab"}[language],
                "--mwer-segmenter",
                str(mwer_bin),
                "--strip-output-tags",
                "term_t",
                "--output-tsv",
                str(aggregate_dir / "eval_results.tsv"),
                "--output-json",
                str(aggregate_dir / "eval_results.json"),
                "--resegmented-jsonl",
                str(aggregate_dir / "resegmented.jsonl"),
            ]
            aggregate_tasks.append(
                {
                    "task_id": f"aggregate__{language}__lm{lm}",
                    "language": language,
                    "lm": lm,
                    "output_dir": str(aggregate_dir),
                    "source_eval_task_ids": [task["task_id"] for task in source_tasks],
                    "offline_command": offline_command,
                    "expected_outputs": {
                        "instances_log": str(aggregate_dir / "instances.log"),
                        "runtime_log": str(
                            aggregate_dir / f"runtime_omni_vllm_maxsim_rag_batched_lm{lm}.jsonl"
                        ),
                        "source_text": str(aggregate_dir / "source_text.txt"),
                        "reference": str(aggregate_dir / "reference.txt"),
                        "audio_manifest": str(aggregate_dir / "audio.json"),
                        "eval_results_tsv": str(aggregate_dir / "eval_results.tsv"),
                        "eval_results_json": str(aggregate_dir / "eval_results.json"),
                        "resegmented_jsonl": str(aggregate_dir / "resegmented.jsonl"),
                    },
                }
            )

    return {
        "schema_version": 1,
        "kind": "rasst_acl_realistic_paper_glossary_run",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_manifest": str(prepared_path),
        "prepared_manifest_sha256": sha256_file(prepared_path),
        "repo_root": str(repo_root),
        "python_bin": str(python_bin),
        "retriever_checkpoint": {
            "path": str(retriever),
            "sha256": sha256_file(retriever),
        },
        "models": {language: _directory_signature(path) for language, path in models.items()},
        "offline_scorer": {
            "path": str(offline_evaluator),
            "sha256": sha256_file(offline_evaluator),
        },
        "mwer_segmenter": {"path": str(mwer_bin), "sha256": sha256_file(mwer_bin)},
        "output_dir": str(output_dir),
        "index_cache_dir": str(index_cache_dir),
        "gpu_contract": (
            "Run inside a container exposing exactly two preflight-selected GPUs as cuda:0,1; "
            "vLLM uses TP=2 and the retriever uses the explicitly recorded logical device."
        ),
        "parameters": {
            "density_tag": args.density_tag,
            "cache_groups": [
                {"lms": list(lms), "max_keep_cache_chunks": cache}
                for lms, cache in selected_cache_groups
            ],
            "lms": list(selected_lms),
            "rag_top_k": args.rag_top_k,
            "rag_score_threshold": args.rag_score_threshold,
            "rag_timeline_lookback_sec": args.rag_timeline_lookback_sec,
            "rag_lora_rank": args.rag_lora_rank,
            "text_lora_rank": args.text_lora_rank,
            "text_lora_alpha": args.text_lora_alpha,
            "text_model_id": TEXT_MODEL_ID,
            "index_device": args.index_device,
            "rag_device": args.rag_device,
            "vllm_tp_size": args.vllm_tp_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "scheduler_batch_size": args.scheduler_batch_size,
            "schedule_mode": args.schedule_mode,
            "vllm_limit_audio": args.vllm_limit_audio,
            "vllm_enforce_eager": args.vllm_enforce_eager,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
            "safetensors_load_strategy": args.safetensors_load_strategy,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k_decode": args.top_k_decode,
            "max_new_tokens": args.max_new_tokens,
            "max_new_tokens_policy": "lm_scaled",
            "seed": args.seed,
            "runtime_glossary": runtime_glossary_policy,
            "term_acc_denominator": "fixed existing raw-gold ACL glossary",
        },
        "index_tasks": index_tasks,
        "eval_tasks": eval_tasks,
        "aggregate_tasks": aggregate_tasks,
    }


def _validate_run_manifest(path: Path) -> Dict[str, Any]:
    manifest = _load_json(path)
    if not isinstance(manifest, dict) or manifest.get("kind") != (
        "rasst_acl_realistic_paper_glossary_run"
    ):
        raise ValueError(f"Unsupported run manifest: {path}")
    prepared_path = _require_file(Path(manifest["prepared_manifest"]))
    if sha256_file(prepared_path) != manifest.get("prepared_manifest_sha256"):
        raise ValueError("Prepared manifest changed after run planning")
    _validate_prepared_manifest(prepared_path)
    retriever = manifest["retriever_checkpoint"]
    if sha256_file(Path(retriever["path"])) != retriever.get("sha256"):
        raise ValueError("Retriever checkpoint changed after run planning")
    for record_name in ("offline_scorer", "mwer_segmenter"):
        record = manifest[record_name]
        if sha256_file(Path(record["path"])) != record.get("sha256"):
            raise ValueError(f"{record_name} changed after run planning")
    return manifest


def _all_nonempty(paths: Sequence[Path]) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def _run_index_tasks(manifest: Dict[str, Any]) -> None:
    cwd = Path(manifest["repo_root"])
    for task in manifest["index_tasks"]:
        _run_checked(task["resolve_command"], cwd=cwd, capture=True)
        index_path = Path(task["index_path"])
        sidecar_path = Path(task["index_manifest_path"])
        if index_path.is_file():
            if not sidecar_path.is_file():
                raise RuntimeError(f"Index exists without sidecar: {index_path}")
            print(f"[SKIP] verified index {index_path}", flush=True)
            continue
        _run_checked(task["build_command"], cwd=cwd)
        _run_checked(task["finalize_command"], cwd=cwd)
        if not _all_nonempty([index_path, sidecar_path]):
            raise RuntimeError(f"Index task did not produce verified outputs: {task['task_id']}")


def _run_eval_tasks(manifest: Dict[str, Any], *, resume: bool) -> None:
    cwd = Path(manifest["repo_root"])
    for task in manifest["eval_tasks"]:
        expected = [
            Path(record[field])
            for record in task["expected_outputs"].values()
            for field in ("instances_log", "runtime_log")
        ]
        existing = [path.is_file() and path.stat().st_size > 0 for path in expected]
        if all(existing):
            if resume:
                print(f"[SKIP] completed eval {task['task_id']}", flush=True)
                continue
            raise FileExistsError(f"Eval outputs already exist: {task['task_id']}")
        if any(existing):
            raise RuntimeError(f"Partial eval outputs require manual inspection: {task['task_id']}")
        _run_checked(task["command"], cwd=cwd)
        if not _all_nonempty(expected):
            raise RuntimeError(f"Eval task did not produce all outputs: {task['task_id']}")


def merge_aggregate_inputs(
    *,
    run_manifest: Dict[str, Any],
    aggregate_task: Dict[str, Any],
) -> Tuple[Path, Path]:
    prepared = _load_json(Path(run_manifest["prepared_manifest"]))
    paper_ids = list(prepared["paper_ids"])
    language = aggregate_task["language"]
    eval_by_id = {task["task_id"]: task for task in run_manifest["eval_tasks"]}
    prepared_shards = {
        (shard["language"], shard["paper_id"]): shard
        for shard in prepared["shards"]
    }
    source_task_ids = aggregate_task["source_eval_task_ids"]
    if len(source_task_ids) != len(paper_ids):
        raise ValueError(f"Aggregate source count mismatch: {aggregate_task['task_id']}")

    merged_instances: List[Dict[str, Any]] = []
    merged_runtime: List[Dict[str, Any]] = []
    merged_sources: List[str] = []
    merged_references: List[str] = []
    merged_audio: List[Dict[str, Any]] = []
    for talk_index, (paper_id, source_task_id) in enumerate(zip(paper_ids, source_task_ids)):
        source_task = eval_by_id[source_task_id]
        if source_task["paper_id"] != paper_id or source_task["language"] != language:
            raise ValueError(
                f"Aggregate paper order mismatch: expected={paper_id}, "
                f"got={source_task['paper_id']}"
            )
        output = source_task["expected_outputs"][str(aggregate_task["lm"])]
        instances = _read_jsonl(Path(output["instances_log"]))
        if len(instances) != 1:
            raise ValueError(f"Expected one full-talk instance for {source_task_id}")
        instance = dict(instances[0])
        source_values = instance.get("source")
        if not isinstance(source_values, list) or not source_values:
            raise ValueError(f"Missing source path in {source_task_id}")
        if Path(str(source_values[0])).stem != paper_id:
            raise ValueError(f"Instance source/paper mismatch in {source_task_id}")
        instance["index"] = talk_index
        merged_instances.append(instance)

        runtime_rows = _read_jsonl(Path(output["runtime_log"]))
        for row in runtime_rows:
            copied = dict(row)
            copied["instance_index"] = talk_index
            merged_runtime.append(copied)

        prepared_shard = prepared_shards[(language, paper_id)]
        merged_sources.extend(
            Path(prepared_shard["files"]["source_text"]["path"])
            .read_text(encoding="utf-8")
            .splitlines()
        )
        merged_references.extend(
            Path(prepared_shard["files"]["ref"]["path"])
            .read_text(encoding="utf-8")
            .splitlines()
        )
        audio_rows = _load_json(Path(prepared_shard["files"]["audio_yaml"]["path"]))
        if not isinstance(audio_rows, list) or not all(isinstance(row, dict) for row in audio_rows):
            raise ValueError(f"Prepared shard audio is not a JSON array: {paper_id}/{language}")
        merged_audio.extend(audio_rows)

    full_inputs = prepared["release_inputs"][language]["source_files"]
    expected_sources = Path(full_inputs["source_text"]["path"]).read_text(
        encoding="utf-8"
    ).splitlines()
    expected_references = Path(full_inputs["ref"]["path"]).read_text(
        encoding="utf-8"
    ).splitlines()
    if merged_sources != expected_sources or merged_references != expected_references:
        raise ValueError(f"Merged sentence inputs do not reproduce full release inputs: {language}")
    if len(merged_audio) != len(merged_references):
        raise ValueError(f"Merged audio/reference count mismatch: {language}")

    instances_path = Path(aggregate_task["expected_outputs"]["instances_log"])
    runtime_path = Path(aggregate_task["expected_outputs"]["runtime_log"])
    _write_jsonl(instances_path, merged_instances)
    _write_jsonl(runtime_path, merged_runtime)
    _write_text(
        Path(aggregate_task["expected_outputs"]["source_text"]),
        "\n".join(merged_sources) + "\n",
    )
    _write_text(
        Path(aggregate_task["expected_outputs"]["reference"]),
        "\n".join(merged_references) + "\n",
    )
    _write_json(
        Path(aggregate_task["expected_outputs"]["audio_manifest"]),
        merged_audio,
    )
    return instances_path, runtime_path


def _run_aggregate_tasks(manifest: Dict[str, Any], *, resume: bool) -> None:
    cwd = Path(manifest["repo_root"])
    for task in manifest["aggregate_tasks"]:
        results = Path(task["expected_outputs"]["eval_results_tsv"])
        if results.is_file() and results.stat().st_size > 0:
            if resume:
                completed = [
                    Path(task["expected_outputs"][name])
                    for name in (
                        "instances_log",
                        "runtime_log",
                        "source_text",
                        "reference",
                        "audio_manifest",
                        "eval_results_tsv",
                        "eval_results_json",
                        "resegmented_jsonl",
                    )
                ]
                if not _all_nonempty(completed):
                    raise RuntimeError(
                        f"Partial aggregate outputs require manual inspection: {task['task_id']}"
                    )
                print(f"[SKIP] completed aggregate {task['task_id']}", flush=True)
                continue
            raise FileExistsError(f"Aggregate output exists: {results}")
        merge_aggregate_inputs(run_manifest=manifest, aggregate_task=task)
        _run_checked(task["offline_command"], cwd=cwd)
        required = [
            Path(task["expected_outputs"][name])
            for name in (
                "instances_log",
                "runtime_log",
                "source_text",
                "reference",
                "audio_manifest",
                "eval_results_tsv",
                "eval_results_json",
                "resegmented_jsonl",
            )
        ]
        if not _all_nonempty(required):
            raise RuntimeError(f"Aggregate task did not produce all outputs: {task['task_id']}")


def run_worker(run_manifest_path: Path, *, resume: bool) -> None:
    manifest = _validate_run_manifest(run_manifest_path)
    _run_index_tasks(manifest)
    _run_eval_tasks(manifest, resume=resume)
    _run_aggregate_tasks(manifest, resume=resume)
    completion_path = Path(manifest["output_dir"]) / "worker_complete.json"
    _write_json(
        completion_path,
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_manifest": str(run_manifest_path.absolute()),
            "run_manifest_sha256": sha256_file(run_manifest_path),
            "index_tasks": len(manifest["index_tasks"]),
            "eval_tasks": len(manifest["eval_tasks"]),
            "aggregate_tasks": len(manifest["aggregate_tasks"]),
        },
    )


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepared-manifest", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--python-bin", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True, help="LANG=/absolute/model/path")
    parser.add_argument("--retriever-checkpoint", required=True, type=Path)
    parser.add_argument("--index-cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mwer-segmenter-bin", required=True, type=Path)
    parser.add_argument("--lms", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--density-tag", default="realistic_paper_glossary")
    parser.add_argument("--index-device", default="cuda:1")
    parser.add_argument("--rag-device", default="cuda:1")
    parser.add_argument("--rag-top-k", type=int, default=10)
    parser.add_argument("--rag-score-threshold", type=float, default=0.78)
    parser.add_argument("--rag-timeline-lookback-sec", type=float, default=1.92)
    parser.add_argument("--rag-lora-rank", type=int, default=128)
    parser.add_argument("--text-lora-rank", type=int, default=128)
    parser.add_argument("--text-lora-alpha", type=int, default=256)
    parser.add_argument("--vllm-tp-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--scheduler-batch-size", type=int, default=8)
    parser.add_argument("--schedule-mode", choices=["round_robin", "serial_by_lm"], default="round_robin")
    parser.add_argument("--vllm-limit-audio", type=int, default=128)
    parser.add_argument("--vllm-enforce-eager", type=int, choices=[0, 1], default=0)
    parser.add_argument("--disable-custom-all-reduce", type=int, choices=[0, 1], default=0)
    parser.add_argument("--safetensors-load-strategy", default="lazy")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k-decode", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--seed", type=int, default=998244353)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", help="Freeze commands and provenance in a run manifest.")
    _add_plan_arguments(plan_parser)

    start_parser = subparsers.add_parser("start", help="Start a detached manifest worker.")
    start_parser.add_argument("--run-manifest", required=True, type=Path)
    start_parser.add_argument("--log-file", required=True, type=Path)
    start_parser.add_argument("--pid-file", required=True, type=Path)
    start_parser.add_argument("--resume", action="store_true")

    worker_parser = subparsers.add_parser("worker", help="Internal detached worker used by start.")
    worker_parser.add_argument("--run-manifest", required=True, type=Path)
    worker_parser.add_argument("--resume", action="store_true")

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Merge completed paper shards and run raw-gold offline evaluation."
    )
    aggregate_parser.add_argument("--run-manifest", required=True, type=Path)
    aggregate_parser.add_argument("--resume", action="store_true")

    index_parser = subparsers.add_parser(
        "index", help="Build or verify only the frozen runtime glossary indices."
    )
    index_parser.add_argument("--run-manifest", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "plan":
        if args.run_manifest.exists():
            raise FileExistsError(f"Run manifest already exists: {args.run_manifest}")
        manifest = build_run_manifest(args)
        _write_json(args.run_manifest, manifest)
        print(json.dumps({"run_manifest": str(args.run_manifest.absolute())}, sort_keys=True))
        return 0
    if args.command == "start":
        run_manifest = _require_file(args.run_manifest)
        if args.pid_file.exists() or args.log_file.exists():
            raise FileExistsError("Refusing to overwrite an existing PID or log file")
        args.pid_file.parent.mkdir(parents=True, exist_ok=True)
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        manifest = _validate_run_manifest(run_manifest)
        command = [
            manifest["python_bin"],
            str(Path(__file__).absolute()),
            "worker",
            "--run-manifest",
            str(run_manifest),
        ]
        if args.resume:
            command.append("--resume")
        log_handle = args.log_file.open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            cwd=manifest["repo_root"],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_handle.close()
        args.pid_file.write_text(str(process.pid) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {"pid": process.pid, "pid_file": str(args.pid_file), "log_file": str(args.log_file)},
                sort_keys=True,
            )
        )
        return 0
    if args.command == "worker":
        if os.getsid(0) != os.getpid():
            raise RuntimeError("worker must be launched detached through the start subcommand")
        run_worker(_require_file(args.run_manifest), resume=args.resume)
        return 0
    if args.command == "aggregate":
        manifest = _validate_run_manifest(_require_file(args.run_manifest))
        _run_aggregate_tasks(manifest, resume=args.resume)
        return 0
    if args.command == "index":
        manifest = _validate_run_manifest(_require_file(args.run_manifest))
        _run_index_tasks(manifest)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
