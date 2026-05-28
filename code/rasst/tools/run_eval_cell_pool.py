#!/usr/bin/env python3
"""Run generated RASST eval cell scripts across local GPU pairs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def abs_no_resolve(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: Sequence[Mapping[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def output_eval_exists(profile_root: Path, cell_id: str) -> bool:
    return len(list((profile_root / "cells" / cell_id).glob("**/eval_results.tsv"))) == 1


def load_tasks(
    profile_roots: Sequence[Path],
    *,
    skip_completed: bool,
    task_mod: int | None = None,
    task_offset: int = 0,
) -> List[Dict[str, str]]:
    grouped: List[List[Dict[str, str]]] = []
    for profile_root in profile_roots:
        manifest = profile_root / "task_manifest.tsv"
        if not manifest.exists():
            raise FileNotFoundError(f"Missing task manifest: {manifest}")
        rows: List[Dict[str, str]] = []
        for row in read_tsv(manifest):
            cell_id = row["cell_id"]
            if skip_completed and output_eval_exists(profile_root, cell_id):
                continue
            rows.append({
                "profile": profile_root.name,
                "profile_root": str(profile_root),
                "task_index": row["task_index"],
                "cell_id": cell_id,
                "cell_script": row["cell_script"],
            })
        grouped.append(rows)

    tasks: List[Dict[str, str]] = []
    max_len = max((len(rows) for rows in grouped), default=0)
    for index in range(max_len):
        for rows in grouped:
            if index < len(rows):
                tasks.append(rows[index])
    if task_mod is not None:
        if task_mod <= 0:
            raise ValueError("--task-mod must be positive.")
        if task_offset < 0 or task_offset >= task_mod:
            raise ValueError("--task-offset must satisfy 0 <= offset < mod.")
        tasks = [task for index, task in enumerate(tasks) if index % task_mod == task_offset]
    return tasks


def compare_args(root: Path, profile_root: Path, python: str, manifest: str) -> List[str]:
    run_meta_path = profile_root / "run_meta.json"
    if not run_meta_path.exists():
        raise FileNotFoundError(f"Missing run_meta.json: {run_meta_path}")
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    overrides = run_meta.get("runtime_overrides") or {}
    args = [
        python,
        str(root / "code/rasst/tools/eval_main_result.py"),
        "--manifest",
        manifest,
        "--compare-output",
        str(profile_root),
    ]
    if overrides.get("force_runner"):
        args.extend(["--force-runner", str(overrides["force_runner"])])
    if overrides.get("lm_list"):
        args.extend(["--lm-list", str(overrides["lm_list"])])
    if overrides.get("fixed_cache_window_sec"):
        args.extend(["--fixed-cache-window-sec", str(overrides["fixed_cache_window_sec"])])
    if overrides.get("cache_seconds"):
        args.extend(["--cache-seconds", str(overrides["cache_seconds"])])
        args.extend(["--cache-seconds-rounding", str(overrides.get("cache_seconds_rounding") or "floor")])
    if overrides.get("cache_chunks_by_lm"):
        args.extend(["--cache-chunks-by-lm", str(overrides["cache_chunks_by_lm"])])
    if overrides.get("max_new_tokens_per_lm"):
        args.extend(["--max-new-tokens-per-lm", str(overrides["max_new_tokens_per_lm"])])
    for key, value in sorted((overrides.get("cell_overrides") or {}).items()):
        args.extend(["--cell-override", f"{key}={value}"])
    return args


def run_task(
    task: Mapping[str, str],
    *,
    root: Path,
    log_root: Path,
    gpu_pair: str,
    env_base: Mapping[str, str],
) -> Dict[str, str]:
    profile_root = Path(task["profile_root"])
    cell_id = task["cell_id"]
    safe_cell = cell_id.replace("/", "_")
    log_dir = log_root / task["profile"] / "direct_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_log = log_dir / f"{int(task['task_index']):03d}__{safe_cell}.out"
    err_log = log_dir / f"{int(task['task_index']):03d}__{safe_cell}.err"
    status = {
        "profile": task["profile"],
        "cell_id": cell_id,
        "task_index": task["task_index"],
        "gpu_pair": gpu_pair,
        "host": os.uname().nodename,
        "started_at_utc": utc_now(),
        "ended_at_utc": "",
        "exit_code": "",
        "status": "running",
        "stdout": str(out_log),
        "stderr": str(err_log),
    }
    env = os.environ.copy()
    env.update(env_base)
    env["CUDA_VISIBLE_DEVICES"] = gpu_pair
    env["RASST_GPU_PAIR"] = f"__RASST_RAW_SHELL__:${{CUDA_VISIBLE_DEVICES:-{gpu_pair}}}"
    with out_log.open("wb") as out, err_log.open("wb") as err:
        out.write(f"[DIRECT] started={status['started_at_utc']} profile={task['profile']} cell={cell_id} gpu_pair={gpu_pair}\n".encode())
        out.flush()
        proc = subprocess.run(
            ["bash", task["cell_script"]],
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            env=env,
            check=False,
        )
    status["ended_at_utc"] = utc_now()
    status["exit_code"] = str(proc.returncode)
    status["status"] = "success" if proc.returncode == 0 else "failed"
    status_path = profile_root / "task_status_direct" / f"{int(task['task_index']):03d}__{safe_cell}.tsv"
    write_tsv(status_path, [status], list(status.keys()))
    return status


def worker_loop(
    task_queue: "queue.Queue[Mapping[str, str]]",
    results: List[Dict[str, str]],
    results_lock: threading.Lock,
    *,
    root: Path,
    log_root: Path,
    gpu_pair: str,
    env_base: Mapping[str, str],
) -> None:
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            return
        try:
            result = run_task(task, root=root, log_root=log_root, gpu_pair=gpu_pair, env_base=env_base)
        except Exception as exc:  # noqa: BLE001 - keep other cells running and record the failure.
            result = {
                "profile": task.get("profile", ""),
                "cell_id": task.get("cell_id", ""),
                "task_index": task.get("task_index", ""),
                "gpu_pair": gpu_pair,
                "host": os.uname().nodename,
                "started_at_utc": "",
                "ended_at_utc": utc_now(),
                "exit_code": "exception",
                "status": "failed",
                "stdout": "",
                "stderr": repr(exc),
            }
        with results_lock:
            results.append(result)
        task_queue.task_done()


def run_compares(root: Path, profile_roots: Iterable[Path], python: str, manifest: str, log_root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for profile_root in profile_roots:
        profile = profile_root.name
        out_log = log_root / profile / "direct_compare.out"
        err_log = log_root / profile / "direct_compare.err"
        out_log.parent.mkdir(parents=True, exist_ok=True)
        args = compare_args(root, profile_root, python, manifest)
        with out_log.open("wb") as out, err_log.open("wb") as err:
            proc = subprocess.run(args, cwd=str(root), stdin=subprocess.DEVNULL, stdout=out, stderr=err, check=False)
        rows.append({
            "profile": profile,
            "exit_code": str(proc.returncode),
            "stdout": str(out_log),
            "stderr": str(err_log),
        })
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--profile-root", action="append", required=True)
    parser.add_argument("--gpu-pairs", default="0,1;2,3;4,5;6,7")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--status-tsv", required=True)
    parser.add_argument("--no-skip-completed", action="store_true")
    parser.add_argument("--task-mod", type=int, default=None, help="Run only tasks where queue_index %% mod == offset.")
    parser.add_argument("--task-offset", type=int, default=0, help="Offset used with --task-mod.")
    parser.add_argument("--no-compare", action="store_true", help="Skip final full-profile comparison.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = abs_no_resolve(args.root)
    profile_roots = [abs_no_resolve(item) for item in args.profile_root]
    gpu_pairs = [item.strip() for item in args.gpu_pairs.split(";") if item.strip()]
    if not gpu_pairs:
        raise SystemExit("No GPU pairs provided.")
    tasks = load_tasks(
        profile_roots,
        skip_completed=not args.no_skip_completed,
        task_mod=args.task_mod,
        task_offset=args.task_offset,
    )
    log_root = abs_no_resolve(args.log_root)
    status_tsv = abs_no_resolve(args.status_tsv)
    log_root.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "started_at_utc": utc_now(),
        "root": str(root),
        "profile_roots": [str(path) for path in profile_roots],
        "gpu_pairs": gpu_pairs,
        "task_count": len(tasks),
        "task_mod": args.task_mod,
        "task_offset": args.task_offset,
        "python": args.python,
        "manifest": args.manifest,
    }
    (log_root / "direct_pool_meta.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env_base = {
        "RASST_ROOT": str(root),
        "PYTHON": args.python,
        "RASST_MAIN_RESULT_MANIFEST": args.manifest,
    }
    task_queue: "queue.Queue[Mapping[str, str]]" = queue.Queue()
    for task in tasks:
        task_queue.put(task)
    results: List[Dict[str, str]] = []
    results_lock = threading.Lock()
    threads = []
    for gpu_pair in gpu_pairs:
        thread = threading.Thread(
            target=worker_loop,
            kwargs={
                "task_queue": task_queue,
                "results": results,
                "results_lock": results_lock,
                "root": root,
                "log_root": log_root,
                "gpu_pair": gpu_pair,
                "env_base": env_base,
            },
            daemon=False,
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    rows = sorted(results, key=lambda row: (row["profile"], int(row["task_index"] or 0)))
    fields = ["profile", "cell_id", "task_index", "gpu_pair", "host", "started_at_utc", "ended_at_utc", "exit_code", "status", "stdout", "stderr"]
    write_tsv(status_tsv, rows, fields)
    compare_rows = [] if args.no_compare else run_compares(root, profile_roots, args.python, args.manifest, log_root)
    write_tsv(log_root / "direct_compare_status.tsv", compare_rows, ["profile", "exit_code", "stdout", "stderr"])
    failed = [row for row in rows if row["status"] != "success"]
    compare_failed = [row for row in compare_rows if row["exit_code"] != "0"]
    if os.access(Path.home() / "bin/codex-notify", os.X_OK):
        msg = (
            f"RASST direct eval pool finished: failed_cells={len(failed)} "
            f"compare_failed={len(compare_failed)} status={status_tsv}"
        )
        subprocess.run(
            [str(Path.home() / "bin/codex-notify"), "--delay", "8", "--detach", "--workspace", str(root), msg],
            check=False,
        )
    return 2 if failed or compare_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
