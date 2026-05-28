#!/usr/bin/env python3
"""Upload or download public Hugging Face assets for the RASST release."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


class ReleaseAssetError(RuntimeError):
    pass


def repo_root() -> Path:
    root_text = os.environ.get("RASST_ROOT")
    if root_text:
        root = Path(root_text).expanduser()
        return root if root.is_absolute() else Path.cwd() / root
    return Path(__file__).resolve().parents[3]


def default_manifest(root: Path) -> Path:
    return root / "code/rasst/manifests/main_result_eval.global_cache30_30_20_20.json"


def rel_or_abs(root: Path, path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else root / path


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ReleaseAssetError(f"Manifest root must be an object: {path}")
    return data


def release_assets(manifest: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    for item in manifest.get("artifacts", []):
        if isinstance(item, dict) and item.get("hf_repo_id"):
            out.append(item)
    return out


def select_assets(assets: Sequence[Mapping[str, Any]], selectors: Sequence[str]) -> List[Mapping[str, Any]]:
    if not selectors or "all" in selectors:
        return list(assets)
    by_key = {str(asset["key"]): asset for asset in assets}
    missing = [name for name in selectors if name not in by_key]
    if missing:
        raise ReleaseAssetError(f"Unknown HF release asset(s): {', '.join(missing)}")
    return [by_key[name] for name in selectors]


def existing_source_path(asset: Mapping[str, Any], root: Path) -> Path:
    env_name = str(asset.get("env") or "")
    candidates: List[Path] = []
    if env_name and os.environ.get(env_name):
        candidates.append(rel_or_abs(root, os.environ[env_name]))
    if asset.get("local_path"):
        candidates.append(rel_or_abs(root, str(asset["local_path"])))
    if asset.get("legacy_path"):
        candidates.append(rel_or_abs(root, str(asset["legacy_path"])))
    for path in candidates:
        if path.exists() and (path.is_dir() or path.stat().st_size > 0):
            return path
    checked = ", ".join(str(path) for path in candidates)
    raise ReleaseAssetError(f"No source path exists for {asset.get('key')}; checked: {checked}")


def target_path(asset: Mapping[str, Any], root: Path) -> Path:
    local_path = asset.get("local_path")
    if not local_path:
        raise ReleaseAssetError(f"Asset {asset.get('key')} has no local_path.")
    return rel_or_abs(root, str(local_path))


def model_card(asset: Mapping[str, Any], source_path: Optional[Path]) -> str:
    key = str(asset["key"])
    repo_id = str(asset["hf_repo_id"])
    meta = asset.get("metadata") if isinstance(asset.get("metadata"), dict) else {}
    lines = [
        "---",
        "library_name: transformers",
        "tags:",
        "- rasst",
        "- speech-translation",
        "- streaming-translation",
        "- research-artifact",
        "---",
        "",
        f"# {repo_id}",
        "",
        "Public RASST release artifact used by the global-cache 30/30/20/20 main result.",
        "",
        f"- Manifest asset key: `{key}`",
        f"- Artifact type: `{asset.get('type')}`",
    ]
    if source_path is not None:
        lines.append(f"- Original release source path: `{source_path}`")
    if meta:
        lines.append("- Manifest metadata:")
        for mkey, mvalue in sorted(meta.items()):
            lines.append(f"  - `{mkey}`: `{mvalue}`")
    if asset.get("type") == "file":
        lines.extend([
            "",
            "This repository stores the MaxSim retriever checkpoint file used by RASST eval.",
        ])
    else:
        lines.extend([
            "",
            "This repository stores a Hugging Face-format Speech-LLM checkpoint directory.",
        ])
    lines.extend([
        "",
        "See the RASST repository for manifests and launch wrappers:",
        "https://github.com/luojiaxuan/RASST",
        "",
    ])
    return "\n".join(lines)


def upload_asset(api: HfApi, asset: Mapping[str, Any], root: Path, *, dry_run: bool) -> None:
    repo_id = str(asset["hf_repo_id"])
    source = existing_source_path(asset, root)
    print(f"[UPLOAD] {asset['key']} {source} -> {repo_id}")
    if dry_run:
        return
    if os.environ.get("RASST_ALLOW_HF_UPLOAD") != "1":
        raise ReleaseAssetError("Set RASST_ALLOW_HF_UPLOAD=1 to upload public HF assets.")
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rasst_hf_card_") as tmpdir:
        card_path = Path(tmpdir) / "README.md"
        card_path.write_text(model_card(asset, source), encoding="utf-8")
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            commit_message=f"Add RASST release card for {asset['key']}",
        )
    if asset.get("type") == "file":
        filename = str(asset.get("hf_filename") or source.name)
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=str(source),
            path_in_repo=filename,
            commit_message=f"Upload {asset['key']}",
        )
    elif asset.get("type") == "hf_model_dir":
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(source),
            path_in_repo=".",
            ignore_patterns=[".stage*", "__pycache__/**"],
            commit_message=f"Upload {asset['key']}",
        )
    else:
        raise ReleaseAssetError(f"Unsupported upload asset type for {asset['key']}: {asset.get('type')}")


def download_asset(asset: Mapping[str, Any], root: Path, *, dry_run: bool, force: bool) -> None:
    repo_id = str(asset["hf_repo_id"])
    revision = str(asset.get("hf_revision") or "main")
    target = target_path(asset, root)
    print(f"[DOWNLOAD] {asset['key']} {repo_id}@{revision} -> {target}")
    if target.exists() and not force:
        print(f"[SKIP] target exists: {target}")
        return
    if dry_run:
        return
    if os.environ.get("RASST_ALLOW_DOWNLOAD") != "1":
        raise ReleaseAssetError("Set RASST_ALLOW_DOWNLOAD=1 to download HF release assets.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if asset.get("type") == "file":
        filename = str(asset.get("hf_filename") or target.name)
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            filename=filename,
            local_dir=str(target.parent),
            force_download=force,
        )
        downloaded = target.parent / filename
        if downloaded != target:
            downloaded.replace(target)
    elif asset.get("type") == "hf_model_dir":
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            local_dir=str(target),
            force_download=force,
            ignore_patterns=[".git/*"],
        )
    else:
        raise ReleaseAssetError(f"Unsupported download asset type for {asset['key']}: {asset.get('type')}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "upload"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--asset", action="append", default=[], help="Asset key to process. Repeatable. Default: all HF assets.")
    parser.add_argument("--execute", action="store_true", help="Perform the action. Default is dry-run.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing downloads or force HF download.")
    parser.add_argument("--list", action="store_true", help="List HF release assets and exit.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    manifest_path = Path(args.manifest) if args.manifest else default_manifest(root)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = load_manifest(manifest_path)
    assets = release_assets(manifest)
    selected = select_assets(assets, args.asset)
    if args.list:
        for asset in selected:
            print(f"{asset['key']}\t{asset['type']}\t{asset['hf_repo_id']}")
        return 0
    dry_run = not args.execute
    if dry_run:
        print("status=dry_run")
    if args.action == "upload":
        api = HfApi()
        for asset in selected:
            upload_asset(api, asset, root, dry_run=dry_run)
    else:
        for asset in selected:
            download_asset(asset, root, dry_run=dry_run, force=args.force)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseAssetError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
