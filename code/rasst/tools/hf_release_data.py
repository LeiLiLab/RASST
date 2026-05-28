#!/usr/bin/env python3
"""Prepare, upload, and download the public RASST eval data package."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from huggingface_hub import HfApi, snapshot_download


class ReleaseDataError(RuntimeError):
    pass


WAV_RE = re.compile(r"/[^\s\"']+?\.wav")


def repo_root() -> Path:
    root_text = os.environ.get("RASST_ROOT")
    if root_text:
        root = Path(root_text).expanduser()
        return root if root.is_absolute() else Path.cwd() / root
    return Path(__file__).resolve().parents[3]


def default_manifest(root: Path) -> Path:
    return root / "code/rasst/manifests/main_result_eval.global_cache30_30_20_20.json"


def default_stage_root() -> Path:
    return Path("/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/hf_datasets/rasst-main-result-data")


def rel_or_abs(root: Path, path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else root / path


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ReleaseDataError(f"Manifest root must be an object: {path}")
    return data


def release_data_meta(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = manifest.get("metadata")
    if not isinstance(meta, dict):
        raise ReleaseDataError("Manifest metadata must be an object.")
    release_data = meta.get("release_data")
    if not isinstance(release_data, dict):
        raise ReleaseDataError("Manifest metadata.release_data must be an object.")
    return release_data


def repo_id(manifest: Mapping[str, Any]) -> str:
    rid = release_data_meta(manifest).get("hf_repo_id")
    if not rid:
        raise ReleaseDataError("metadata.release_data.hf_repo_id is missing.")
    return str(rid)


def release_data_local_root(manifest: Mapping[str, Any], root: Path) -> Path:
    local_path = str(release_data_meta(manifest).get("local_path") or "data")
    return rel_or_abs(root, local_path)


def artifacts_by_type(manifest: Mapping[str, Any], asset_type: str) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    for item in manifest.get("artifacts", []):
        if isinstance(item, dict) and item.get("type") == asset_type:
            out.append(item)
    return out


def source_path(asset: Mapping[str, Any], root: Path) -> Path:
    candidates = []
    if asset.get("legacy_path"):
        candidates.append(rel_or_abs(root, str(asset["legacy_path"])))
    if asset.get("local_path"):
        candidates.append(rel_or_abs(root, str(asset["local_path"])))
    for path in candidates:
        if path.exists():
            return path
    raise ReleaseDataError(f"No source path exists for {asset.get('key')}: {candidates}")


def package_path_for_local_path(local_path: str) -> Path:
    path = Path(local_path)
    parts = path.parts
    if parts and parts[0] == "data":
        return Path(*parts[1:])
    return path


def audio_family(path_text: str) -> str:
    if "acl6060" in path_text:
        return "acl6060"
    if "eso-dataset" in path_text or "rag-sst" in path_text:
        return "medicine"
    return "other"


def audio_package_rel(path_text: str) -> Path:
    source = Path(path_text)
    family = audio_family(path_text)
    if family == "medicine":
        return Path("main_result/audio/medicine") / source.parent.name / source.name
    if family == "acl6060":
        return Path("main_result/audio/acl6060") / source.name
    return Path("main_result/audio/other") / source.name


def audio_runtime_ref(package_rel: Path) -> str:
    return str(Path("data") / package_rel)


def replace_wav_paths(text: str, mapping: Mapping[str, str]) -> str:
    for old, new in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(old, new)
    return text


def copy_file_with_rewrites(src: Path, dst: Path, mapping: Mapping[str, str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = src.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        shutil.copy2(src, dst)
        return
    dst.write_text(replace_wav_paths(text, mapping), encoding="utf-8")


def collect_wavs(input_roots: Sequence[Path]) -> Dict[str, Path]:
    wavs: Dict[str, Path] = {}
    for root in input_roots:
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for match in WAV_RE.findall(text):
                path = Path(match)
                if not path.exists():
                    raise ReleaseDataError(f"Referenced WAV does not exist: {path} in {file_path}")
                wavs[match] = path
    return wavs


def write_dataset_card(stage_root: Path, manifest: Mapping[str, Any]) -> None:
    rid = repo_id(manifest)
    text = f"""---
license: other
pretty_name: RASST Main Result Eval Data
tags:
- rasst
- speech-translation
- streaming-translation
- research-artifact
---

# RASST Main Result Eval Data

This dataset contains the release-facing RASST main-result evaluation inputs,
glossaries, and referenced audio snippets for the ACL6060 tagged and medicine
hard/raw evaluation tracks.

It is intended to be downloaded into the RASST repository's ignored `data/`
directory:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
RASST_ALLOW_DOWNLOAD=1 bash code/rasst/scripts/download_release_data.sh --download
```

After download, the RASST eval manifest resolves these paths:

- `data/glossaries/`
- `data/main_result/inputs/`
- `data/main_result/audio/`

Source repository: https://github.com/luojiaxuan/RASST

HF dataset repo: `{rid}`

The dataset is released as a research artifact for reproducing the reported
RASST evaluation. Check upstream dataset and audio-source licenses before
redistributing derivative copies.
"""
    (stage_root / "README.md").write_text(text, encoding="utf-8")


def prepare_package(manifest: Mapping[str, Any], root: Path, stage_root: Path, *, force: bool) -> None:
    if stage_root.exists():
        if not force:
            raise ReleaseDataError(f"Stage root already exists. Use --force to rebuild: {stage_root}")
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True)

    input_assets = artifacts_by_type(manifest, "input_dir")
    glossary_assets = artifacts_by_type(manifest, "json")
    input_roots = [source_path(asset, root) for asset in input_assets]
    wav_sources = collect_wavs(input_roots)
    wav_mapping: Dict[str, str] = {}
    copied_audio: Dict[str, str] = {}
    for old_text, old_path in sorted(wav_sources.items()):
        rel = audio_package_rel(old_text)
        dst = stage_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(old_path, dst)
        wav_mapping[old_text] = audio_runtime_ref(rel)
        copied_audio[str(rel)] = old_text

    for asset in input_assets:
        src_root = source_path(asset, root)
        local_path = str(asset["local_path"])
        dst_root = stage_root / package_path_for_local_path(local_path)
        for src in src_root.rglob("*"):
            if src.is_dir():
                continue
            dst = dst_root / src.relative_to(src_root)
            copy_file_with_rewrites(src, dst, wav_mapping)

    for asset in glossary_assets:
        src = source_path(asset, root)
        dst = stage_root / package_path_for_local_path(str(asset["local_path"]))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    manifest_out = {
        "hf_repo_id": repo_id(manifest),
        "source_manifest_event_id": manifest.get("event_id"),
        "local_download_root": "data",
        "input_assets": [
            {"key": asset["key"], "local_path": asset["local_path"], "source_path": str(source_path(asset, root))}
            for asset in input_assets
        ],
        "glossary_assets": [
            {"key": asset["key"], "local_path": asset["local_path"], "source_path": str(source_path(asset, root))}
            for asset in glossary_assets
        ],
        "audio_files": copied_audio,
    }
    (stage_root / "dataset_manifest.json").write_text(
        json.dumps(manifest_out, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_dataset_card(stage_root, manifest)
    print(f"status=prepared stage_root={stage_root}")
    print(f"input_dirs={len(input_assets)} glossaries={len(glossary_assets)} audio_files={len(copied_audio)}")


def upload_package(manifest: Mapping[str, Any], stage_root: Path, *, dry_run: bool) -> None:
    rid = repo_id(manifest)
    if not stage_root.exists():
        raise ReleaseDataError(f"Stage root missing: {stage_root}")
    print(f"[UPLOAD_DATA] {stage_root} -> {rid}")
    if dry_run:
        return
    if os.environ.get("RASST_ALLOW_HF_UPLOAD") != "1":
        raise ReleaseDataError("Set RASST_ALLOW_HF_UPLOAD=1 to upload public HF data.")
    api = HfApi()
    api.create_repo(repo_id=rid, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=rid,
        repo_type="dataset",
        folder_path=str(stage_root),
        path_in_repo=".",
        commit_message="Upload RASST main-result eval data",
    )


def download_package(manifest: Mapping[str, Any], root: Path, *, dry_run: bool, force: bool) -> None:
    rid = repo_id(manifest)
    local_root = release_data_local_root(manifest, root)
    revision = str(release_data_meta(manifest).get("hf_revision") or "main")
    print(f"[DOWNLOAD_DATA] {rid}@{revision} -> {local_root}")
    non_placeholder_items = []
    if local_root.exists():
        non_placeholder_items = [item for item in local_root.iterdir() if item.name != ".gitkeep"]
    if non_placeholder_items and not force:
        print(f"[SKIP] target exists: {local_root}")
        return
    if dry_run:
        return
    if os.environ.get("RASST_ALLOW_DOWNLOAD") != "1":
        raise ReleaseDataError("Set RASST_ALLOW_DOWNLOAD=1 to download HF release data.")
    local_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=rid,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_root),
        force_download=force,
        ignore_patterns=[".git/*"],
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "upload", "download"))
    p.add_argument("--manifest", default=None)
    p.add_argument("--stage-root", default=str(default_stage_root()))
    p.add_argument("--execute", action="store_true", help="Perform upload/download. Default is dry-run for those actions.")
    p.add_argument("--force", action="store_true", help="Overwrite stage/download targets when supported.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    manifest_path = Path(args.manifest) if args.manifest else default_manifest(root)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = load_manifest(manifest_path)
    stage_root = Path(args.stage_root).expanduser()
    if not stage_root.is_absolute():
        stage_root = root / stage_root
    if args.action == "prepare":
        prepare_package(manifest, root, stage_root, force=args.force)
    elif args.action == "upload":
        upload_package(manifest, stage_root, dry_run=not args.execute)
    elif args.action == "download":
        download_package(manifest, root, dry_run=not args.execute, force=args.force)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseDataError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
