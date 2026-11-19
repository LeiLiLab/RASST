#!/usr/bin/env python3
"""Prepare, upload, and download the public RASST Speech-LLM SFT dataset.

Only JSONL metadata, stats/recipe, and clip boundaries are published. Audio is
never redistributed: every JSONL ``audios`` entry is rewritten to a relative key
(``audio/<lang>/<gigaspeech_audio_id>/<utterance_index>/<clip_index>.wav``), and
any remaining internal absolute mount/home path is scrubbed from the JSONL and
stats files. A final leak guard fails fast if an internal path survives.

For each language, ``audio_clip_map.parquet`` maps every clip key to its
GigaSpeech opus file, start sample, and length (16 kHz), and
``aligned_utterances.tsv`` lists the utterance boundaries and target-side
trajectories the clips were cut from, so users can rebuild the audio with
``code/rasst/tools/reconstruct_slm_sft_audio.py``.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from huggingface_hub import HfApi, snapshot_download


class ReleaseSlmError(RuntimeError):
    pass


# Strip "/mnt/[host/]<data*|home>/<user>/" and "/home/<user>/" prefixes so that
# no internal absolute path survives in published JSONL/stats files.
INTERNAL_MOUNT_RE = re.compile(r"/mnt/(?:taurus/|aries/|gemini/)?(?:data\d*|home)/[^/]+/")
HOME_RE = re.compile(r"/home/[^/]+/")
# Any remaining internal path is treated as a leak.
LEAK_RE = re.compile(r"/mnt/[^\s\"']+|/home/[^\s\"']+")


def repo_root() -> Path:
    root_text = os.environ.get("RASST_ROOT")
    if root_text:
        root = Path(root_text).expanduser()
        return root if root.is_absolute() else Path.cwd() / root
    return Path(__file__).resolve().parents[3]


def default_manifest(root: Path) -> Path:
    return root / "code/rasst/manifests/slm_training_dataset.cap16_denoise_budget_ttag.json"


def default_stage_root() -> Path:
    # Repo-relative under the ignored outputs/ dir; override with --stage-root.
    return (
        Path(__file__).resolve().parents[3]
        / "outputs/hf_datasets/rasst-speech-llm-sft-cap16-denoise-ttag"
    )


def rel_or_abs(root: Path, path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else root / path


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ReleaseSlmError(f"Manifest root must be an object: {path}")
    return data


def metadata(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = manifest.get("metadata")
    if not isinstance(meta, dict):
        raise ReleaseSlmError("Manifest metadata must be an object.")
    return meta


def release_data_meta(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    release_data = metadata(manifest).get("release_data")
    if not isinstance(release_data, dict):
        raise ReleaseSlmError("metadata.release_data must be an object.")
    return release_data


def repo_id(manifest: Mapping[str, Any]) -> str:
    rid = release_data_meta(manifest).get("hf_repo_id")
    if not rid:
        raise ReleaseSlmError("metadata.release_data.hf_repo_id is missing.")
    return str(rid)


def languages(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    langs = metadata(manifest).get("languages")
    if not isinstance(langs, dict) or not langs:
        raise ReleaseSlmError("metadata.languages must be a non-empty object.")
    return langs


def recipe(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    rec = metadata(manifest).get("recipe")
    return rec if isinstance(rec, dict) else {}


def local_root(manifest: Mapping[str, Any], root: Path) -> Path:
    local_path = str(release_data_meta(manifest).get("local_path") or "data/slm_training")
    return rel_or_abs(root, local_path)


def scrub_text(value: str) -> str:
    value = INTERNAL_MOUNT_RE.sub("", value)
    value = HOME_RE.sub("", value)
    return value


def scrub_value(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, list):
        return [scrub_value(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub_value(item) for key, item in value.items()}
    return value


def assert_no_leak(text: str, context: str) -> None:
    match = LEAK_RE.search(text)
    if match:
        raise ReleaseSlmError(f"Internal path leak in {context}: {match.group(0)!r}")


class AudioStats:
    def __init__(self) -> None:
        self.audio_refs = 0
        self.uttids: Set[str] = set()
        self.clip_keys: Set[str] = set()


def rewrite_audios(record: Mapping[str, Any], *, lang: str, audio_root: str, stats: AudioStats) -> None:
    audios = record.get("audios")
    if audios is None:
        return
    if not isinstance(audios, list):
        raise ReleaseSlmError(f"'audios' must be a list (lang={lang}).")
    prefix = audio_root.rstrip("/") + "/"
    rewritten: List[str] = []
    for entry in audios:
        if not isinstance(entry, str):
            raise ReleaseSlmError(f"'audios' entries must be strings (lang={lang}).")
        if not entry.startswith(prefix):
            raise ReleaseSlmError(
                f"Audio path is not under declared audio_root for {lang}: {entry} (audio_root={audio_root})"
            )
        rel = entry[len(prefix):]
        if not rel or rel.startswith("/"):
            raise ReleaseSlmError(f"Unexpected audio path layout for {lang}: {entry}")
        key = f"audio/{lang}/{rel}"
        rewritten.append(key)
        stats.audio_refs += 1
        stats.uttids.add(rel.split("/", 1)[0])
        stats.clip_keys.add(key)
    record["audios"] = rewritten


def stage_jsonl(src: Path, dst: Path, *, lang: str, audio_root: str, max_rows: int, stats: AudioStats) -> int:
    if not src.is_file():
        raise ReleaseSlmError(f"Missing JSONL source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReleaseSlmError(f"Malformed JSON in {src} line {lineno}: {exc}") from exc
            rewrite_audios(record, lang=lang, audio_root=audio_root, stats=stats)
            record = scrub_value(record)
            out = json.dumps(record, ensure_ascii=False)
            assert_no_leak(out, f"{src.name} line {lineno}")
            fout.write(out + "\n")
            count += 1
            if max_rows and count >= max_rows:
                break
    return count


def stage_stats(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise ReleaseSlmError(f"Missing stats source: {src}")
    data = scrub_value(json.loads(src.read_text(encoding="utf-8")))
    out = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    assert_no_leak(out, src.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(out, encoding="utf-8")


SAMPLE_RATE = 16000
ALIGNMENT_TEXT_COLUMNS = ("tgt_text", "trajectory")


def load_alignment(path: Path, need: Set[str]) -> Dict[str, Dict[str, Any]]:
    """Read the aligned TSV rows for the needed utterances.

    The clips were cut by splitting each utterance span ``<opus>:<start>:<frames>``
    into ``len(trajectory)`` equal base chunks, so the target trajectory length is
    part of the boundary definition.
    """
    if not path.is_file():
        raise ReleaseSlmError(f"Missing alignment TSV: {path}")
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        for name in ("id", "audio", *ALIGNMENT_TEXT_COLUMNS):
            if name not in col:
                raise ReleaseSlmError(f"{path} lacks column {name!r}")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] not in need:
                continue
            opus, start, frames = parts[col["audio"]].rsplit(":", 2)
            trajectory = ast.literal_eval(parts[col["trajectory"]].replace("\x00", ""))
            out[parts[0]] = {
                "opus": opus,
                "start": int(start),
                "frames": int(frames),
                "trajectory": trajectory,
                "tgt_text": parts[col["tgt_text"]],
            }
    missing = need - set(out)
    if missing:
        raise ReleaseSlmError(f"{len(missing)} utterances missing from {path}, e.g. {sorted(missing)[:3]}")
    return out


def build_clip_map(
    lang: str,
    spec: Mapping[str, Any],
    gigaspeech_root: str,
    lang_dir: Path,
    *,
    max_rows: int,
) -> Dict[str, Any]:
    """Write ``audio_clip_map.parquet`` and ``aligned_utterances.tsv`` for one language.

    Clip ``i`` of an utterance with ``T`` base chunks of ``b`` samples covers
    ``[round(i*m*b), round(min((i+1)*m, T)*b))`` relative to the utterance start,
    where ``m`` is the number of base chunks per clip. ``m`` is the row's
    ``merge_multiplier`` when that reproduces the clip lengths on disk, and is
    read from the first clip otherwise; every clip length is checked either way.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf

    src_dir = Path(str(spec["source_dir"]))
    audio_root = str(spec["audio_root"]).rstrip("/") + "/"
    gs_prefix = gigaspeech_root.rstrip("/") + "/"
    rows: List[Tuple[str, str, int, List[str]]] = []
    for split in ("train", "dev"):
        with (src_dir / str(spec["files"][split])).open("r", encoding="utf-8") as f:
            for count, line in enumerate(f):
                if max_rows and count >= max_rows:
                    break
                record = json.loads(line)
                audios = record["audios"]
                rel = [a[len(audio_root):].split("/") for a in audios]
                if any(len(r) != 3 or r[:2] != rel[0][:2] for r in rel):
                    raise ReleaseSlmError(f"Unexpected clip layout in {split}: {audios[:2]}")
                rows.append((split, f"{rel[0][0]}_{rel[0][1]}", int(record.get("merge_multiplier", 1)), audios))

    meta = load_alignment(Path(str(spec["alignment_tsv"])), {utt for _, utt, _, _ in rows})
    for utt, m in meta.items():
        if not m["opus"].startswith(gs_prefix):
            raise ReleaseSlmError(f"{utt}: opus path outside gigaspeech_root: {m['opus']}")
    opus_files = sorted({m["opus"] for m in meta.values()})
    clip_paths = sorted({a for _, _, _, audios in rows for a in audios})
    with cf.ThreadPoolExecutor(64) as pool:
        opus_frames = dict(zip(opus_files, pool.map(lambda p: sf.info(p).frames, opus_files)))
        disk = dict(zip(clip_paths, pool.map(lambda p: sf.info(p), clip_paths)))

    clips: Dict[str, Dict[str, Any]] = {}
    utterances: Dict[str, Dict[str, Any]] = {}
    field_mismatch = 0
    field_values_on_mismatch: Dict[int, int] = {}
    for split, utt, field_mult, audios in rows:
        m = meta[utt]
        tlen = len(m["trajectory"])
        total = min(m["frames"], opus_frames[m["opus"]] - m["start"])
        base = total / tlen
        def clip_lengths(m: int) -> List[int]:
            return [
                max(1, int(round(min((i + 1) * m, tlen) * base)) - int(round(i * m * base)))
                for i in range(len(audios))
            ]

        if clip_lengths(field_mult) == [disk[a].frames for a in audios]:
            mult = field_mult
        else:
            mult = max(1, int(round(disk[audios[0]].frames / base)))
            field_mismatch += 1
            field_values_on_mismatch[field_mult] = field_values_on_mismatch.get(field_mult, 0) + 1
        gs_path = m["opus"][len(gs_prefix):]
        for idx, path in enumerate(audios):
            seg_start = int(round(idx * mult * base))
            seg_end = int(round(min((idx + 1) * mult, tlen) * base))
            frames = max(1, seg_end - seg_start)
            info = disk[path]
            if info.frames != frames or info.samplerate != SAMPLE_RATE:
                raise ReleaseSlmError(
                    f"{path}: disk has {info.frames} frames @ {info.samplerate} Hz, boundaries give {frames} (m={mult})"
                )
            start = m["start"] + seg_start
            record = {
                "key": f"audio/{lang}/{path[len(audio_root):]}",
                "utter_id": utt,
                "gigaspeech_audio_id": Path(gs_path).stem,
                "gigaspeech_path": gs_path,
                "clip_index": idx,
                "chunk_multiplier": mult,
                "start_sample": start,
                "num_samples": frames,
                "start_sec": start / SAMPLE_RATE,
                "duration_sec": frames / SAMPLE_RATE,
                "pad_samples": max(0, start + frames - opus_frames[m["opus"]]),
            }
            previous = clips.setdefault(record["key"], record)
            if previous != record:
                raise ReleaseSlmError(f"Conflicting boundaries for {record['key']}")
        entry = utterances.setdefault(utt, {
            "id": utt,
            "audio": f"{gs_path}:{m['start']}:{m['frames']}",
            "n_frames": m["frames"],
            "trajectory_len": tlen,
            "chunk_multiplier": mult,
            "num_clips": len(audios),
            "splits": [],
            "tgt_text": m["tgt_text"],
            "trajectory": json.dumps(m["trajectory"], ensure_ascii=False),
        })
        if entry["chunk_multiplier"] != mult or entry["num_clips"] != len(audios):
            raise ReleaseSlmError(f"{utt}: rows disagree on clip layout")
        if split not in entry["splits"]:
            entry["splits"].append(split)

    clip_rows = [clips[k] for k in sorted(clips)]
    pq.write_table(pa.Table.from_pylist(clip_rows), lang_dir / "audio_clip_map.parquet", compression="zstd")
    fields = ["id", "audio", "n_frames", "trajectory_len", "chunk_multiplier", "num_clips", "splits", *ALIGNMENT_TEXT_COLUMNS]
    lines = ["\t".join(fields)]
    for utt in sorted(utterances):
        entry = {**utterances[utt], "splits": ",".join(utterances[utt]["splits"])}
        values = [str(entry[name]) for name in fields]
        if any("\t" in v or "\n" in v for v in values):
            raise ReleaseSlmError(f"{utt}: tab or newline inside a TSV field")
        lines.append("\t".join(values))
    (lang_dir / "aligned_utterances.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert_no_leak((lang_dir / "aligned_utterances.tsv").read_text(encoding="utf-8"), f"{lang}/aligned_utterances.tsv")
    return {
        "clips": len(clip_rows),
        "utterances": len(utterances),
        "gigaspeech_opus_files": len(opus_files),
        "clips_with_padding": sum(1 for r in clip_rows if r["pad_samples"]),
        "rows_whose_merge_multiplier_differs_from_clips": field_mismatch,
        "merge_multiplier_values_on_those_rows": {str(k): v for k, v in sorted(field_values_on_mismatch.items())},
        "rows": len(rows),
    }


def write_dataset_card(stage_root: Path, manifest: Mapping[str, Any], per_lang: Mapping[str, Any]) -> None:
    rid = repo_id(manifest)
    rec = recipe(manifest)
    audio_source = release_data_meta(manifest).get("audio_source", {})
    src_dataset = str(audio_source.get("dataset") or "GigaSpeech")
    src_url = str(audio_source.get("url") or "https://github.com/SpeechColab/GigaSpeech")
    download_root = str(release_data_meta(manifest).get("local_path") or "data/slm_training")

    rows = [
        "| Language | Train rows | Dev rows | Utterances | Clips | Unique GigaSpeech audio IDs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    stale_notes = []
    for lang in sorted(per_lang):
        info = per_lang[lang]
        clip_map = info["audio_clip_map"]
        rows.append(
            f"| {lang} | {info['train_rows']} | {info['dev_rows']} | {clip_map['utterances']} | "
            f"{clip_map['clips']} | {info['unique_gigaspeech_audio_ids']} |"
        )
        stale = clip_map["rows_whose_merge_multiplier_differs_from_clips"]
        if stale:
            stale_notes.append(f"`{lang}` ({stale} of {clip_map['rows']} rows)")
    counts_table = "\n".join(rows)
    stale_text = ""
    if stale_notes:
        stale_text = (
            "\nThe JSONL field `merge_multiplier` does not reproduce the clips in "
            + ", ".join(stale_notes)
            + ". The clips are what the models were trained on, so use `chunk_multiplier` from "
            "`audio_clip_map.parquet` when you need the chunk size.\n"
        )

    text = f"""---
license: other
pretty_name: RASST Speech-LLM SFT Data (cap16 denoise-budget term tagging)
tags:
- rasst
- speech-translation
- streaming-translation
- speech-llm
- research-artifact
---

# RASST Speech-LLM SFT Data (cap16 denoise-budget term tagging)

This dataset contains the supervised fine-tuning (SFT) data used to train the
RASST Speech-LLMs for streaming speech translation with domain terminology, for
`de`, `ja`, and `zh`: chunk-by-chunk translation trajectories built from
{src_dataset} ({src_url}), with retrieved term maps and `<t>` term tags.

**Audio is not included.** Each JSONL row has an `audios` field whose entries
are relative keys of the form:

```text
audio/<lang>/<gigaspeech_audio_id>/<utterance_index>/<clip_index>.wav
```

`audio_clip_map.parquet` gives the exact boundaries of every clip in the
original GigaSpeech recording, so the audio can be rebuilt from a GigaSpeech
copy (see below).

## Contents

```text
<lang>/<train_jsonl>             # SFT training rows (system/user/assistant messages + term metadata)
<lang>/<dev_jsonl>               # 355 dev rows (their utterances also appear in the train file)
<lang>/audio_clip_map.parquet    # clip key -> GigaSpeech opus file, start sample, length (16 kHz)
<lang>/aligned_utterances.tsv    # utterance spans and target-side trajectories the clips were cut from
<lang>/*_wrap_stats.json         # term-tagging wrap stats
<lang>/validation_summary.json
audio_sources.json               # GigaSpeech provenance + referenced audio IDs
dataset_manifest.json            # recipe, repo id, and per-language counts
```

## Row counts

{counts_table}

## Audio clip boundaries

Each utterance is a span `<gigaspeech_opus>:<start_sample>:<num_samples>` of a
GigaSpeech recording at 16 kHz (the `audio` column of `aligned_utterances.tsv`,
same format as the original aligned TSV). The span is split into
`trajectory_len` equal base chunks, one per entry of the target
`trajectory`, and clip `i` covers base chunks `[i*m, min((i+1)*m, trajectory_len))`
with `m = chunk_multiplier`. In this release every base chunk is 0.96 s
(15,360 samples).

`audio_clip_map.parquet` has one row per clip key with the result:
`gigaspeech_path` (path inside the GigaSpeech release, e.g.
`audio/audiobook/P0001/AUD0000000003.opus`), `start_sample` and `num_samples` in the
recording, `start_sec` and `duration_sec`, `utter_id`, `clip_index`,
`chunk_multiplier`, and `pad_samples` (zeros appended past the end of the recording).
{stale_text}
To rebuild the clips you need the full-length GigaSpeech opus files from the
official GigaSpeech download. Then:

```bash
git clone https://github.com/LeiLiLab/RASST.git
cd RASST
python code/rasst/tools/reconstruct_slm_sft_audio.py \\
  --dataset-dir <local copy of this dataset> \\
  --gigaspeech-root <dir that contains GigaSpeech audio/>
```

Clips are written to `<dataset-dir>/<key>`, where the JSONL `audios` keys point.
Start positions and lengths match the training clips exactly; sample values can
differ slightly with the Opus decoder build.

English transcripts are part of the GigaSpeech database and are not
redistributed here; `aligned_utterances.tsv` carries only the span boundaries
and our target-language translations.

## Recipe

- Term-map variant: `{rec.get('term_map_variant', 'cap16_denoise_budget_ttag')}`
- Retriever: `{rec.get('retriever', 'hn1024_tau078')}`
- Max terms per chunk: `{rec.get('max_terms_per_chunk', 16)}`
- Assistant term tagging template: `{rec.get('tag_template', '<t>{translation}</t>')}`
- Denoise-budget policy: `{rec.get('denoise_budget_version', 'cap16_denoise_budget_v1')}`
- Base model: `{rec.get('base_model', 'Qwen3-Omni-30B-A3B-Instruct')}` (LoRA r{rec.get('lora_rank', 32)} a{rec.get('lora_alpha', 32)}, {rec.get('max_epochs', 1)} epoch, max_length {rec.get('max_length', 3072)})

The full data-prep, training, and HF export recipe is described in the RASST
repository at `code/rasst/manifests/slm_training.cap16_denoise_budget_ttag.json`.

## Download into the RASST repo

```bash
git clone https://github.com/LeiLiLab/RASST.git
cd RASST
RASST_ALLOW_DOWNLOAD=1 bash code/rasst/scripts/upload_hf_slm_dataset.sh download --execute
```

This populates the ignored local path `{download_root}`.

Source repository: https://github.com/LeiLiLab/RASST

HF dataset repo: `{rid}`

Released as a research artifact. GigaSpeech audio is licensed for non-commercial
research and education; follow the GigaSpeech terms for any audio you rebuild.
"""
    # The card uses a public clone path in its commands and carries no internal
    # data/audio paths, so it is exempt from the strict leak guard applied to the
    # JSONL/stats artifacts.
    (stage_root / "README.md").write_text(text, encoding="utf-8")


def prepare_package(manifest: Mapping[str, Any], stage_root: Path, *, force: bool, max_rows: int) -> None:
    if stage_root.exists():
        if not force:
            raise ReleaseSlmError(f"Stage root already exists. Use --force to rebuild: {stage_root}")
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True)

    langs = languages(manifest)
    audio_source = dict(release_data_meta(manifest).get("audio_source", {}))
    per_lang_report: Dict[str, Any] = {}
    audio_sources: Dict[str, Any] = {
        "source_dataset": audio_source.get("dataset", "GigaSpeech"),
        "source_url": audio_source.get("url", "https://github.com/SpeechColab/GigaSpeech"),
        "redistributed": bool(audio_source.get("redistributed", False)),
        "relative_prefix": audio_source.get("relative_prefix", "audio"),
        "note": audio_source.get(
            "note", "Audio not included; 'audios' fields hold relative GigaSpeech-style keys."
        ),
        "languages": {},
    }

    for lang in sorted(langs):
        spec = langs[lang]
        if not isinstance(spec, dict):
            raise ReleaseSlmError(f"metadata.languages.{lang} must be an object.")
        src_dir = Path(str(spec["source_dir"]))
        audio_root = str(spec["audio_root"])
        files = spec.get("files")
        if not isinstance(files, dict) or "train" not in files or "dev" not in files:
            raise ReleaseSlmError(f"metadata.languages.{lang}.files must define train and dev.")
        lang_dir = stage_root / lang
        stats = AudioStats()
        n_train = stage_jsonl(
            src_dir / str(files["train"]), lang_dir / str(files["train"]),
            lang=lang, audio_root=audio_root, max_rows=max_rows, stats=stats,
        )
        n_dev = stage_jsonl(
            src_dir / str(files["dev"]), lang_dir / str(files["dev"]),
            lang=lang, audio_root=audio_root, max_rows=max_rows, stats=stats,
        )
        if not max_rows:
            exp_train = int(spec.get("expected_train_rows", 0) or 0)
            exp_dev = int(spec.get("expected_dev_rows", 0) or 0)
            if exp_train and n_train != exp_train:
                raise ReleaseSlmError(f"{lang} train row mismatch: got {n_train}, expected {exp_train}")
            if exp_dev and n_dev != exp_dev:
                raise ReleaseSlmError(f"{lang} dev row mismatch: got {n_dev}, expected {exp_dev}")
        for stat_name in files.get("stats", []) or []:
            stage_stats(src_dir / str(stat_name), lang_dir / str(stat_name))
        clip_map = build_clip_map(
            lang, spec, str(audio_source["gigaspeech_root"]), lang_dir, max_rows=max_rows
        )
        print(f"[CLIP_MAP] {lang} {clip_map}", flush=True)
        per_lang_report[lang] = {
            "train_jsonl": str(files["train"]),
            "dev_jsonl": str(files["dev"]),
            "train_rows": n_train,
            "dev_rows": n_dev,
            "audio_refs": stats.audio_refs,
            "unique_clips": len(stats.clip_keys),
            "unique_gigaspeech_audio_ids": len(stats.uttids),
            "stats_files": list(files.get("stats", []) or []),
            "audio_clip_map": clip_map,
        }
        audio_sources["languages"][lang] = {
            "relative_prefix": f"audio/{lang}",
            "internal_clip_dir_name": Path(audio_root).name,
            "num_audio_refs": stats.audio_refs,
            "num_unique_clips": len(stats.clip_keys),
            "num_unique_gigaspeech_audio_ids": len(stats.uttids),
            "gigaspeech_audio_ids": sorted(stats.uttids),
        }

    audio_text = json.dumps(audio_sources, indent=2, ensure_ascii=False) + "\n"
    assert_no_leak(audio_text, "audio_sources.json")
    (stage_root / "audio_sources.json").write_text(audio_text, encoding="utf-8")

    dataset_manifest = {
        "hf_repo_id": repo_id(manifest),
        "source_manifest_event_id": manifest.get("event_id"),
        "local_download_root": str(release_data_meta(manifest).get("local_path") or "data/slm_training"),
        "recipe": recipe(manifest),
        "languages": per_lang_report,
        "audio": {
            "redistributed": False,
            "source": audio_source.get("dataset", "GigaSpeech"),
            "relative_prefix": audio_source.get("relative_prefix", "audio"),
        },
        "truncated_max_rows": max_rows or None,
    }
    dm_text = json.dumps(dataset_manifest, indent=2, ensure_ascii=False) + "\n"
    assert_no_leak(dm_text, "dataset_manifest.json")
    (stage_root / "dataset_manifest.json").write_text(dm_text, encoding="utf-8")

    write_dataset_card(stage_root, manifest, per_lang_report)

    total_train = sum(info["train_rows"] for info in per_lang_report.values())
    total_dev = sum(info["dev_rows"] for info in per_lang_report.values())
    print(f"status=prepared stage_root={stage_root}")
    print(
        f"languages={sorted(per_lang_report)} total_train_rows={total_train} "
        f"total_dev_rows={total_dev} max_rows={max_rows or 'all'}"
    )


def upload_package(manifest: Mapping[str, Any], stage_root: Path, *, dry_run: bool) -> None:
    rid = repo_id(manifest)
    if not stage_root.exists():
        raise ReleaseSlmError(f"Stage root missing (run prepare first): {stage_root}")
    print(f"[UPLOAD_SLM_DATA] {stage_root} -> {rid}")
    if dry_run:
        print("status=dry_run")
        return
    if os.environ.get("RASST_ALLOW_HF_UPLOAD") != "1":
        raise ReleaseSlmError("Set RASST_ALLOW_HF_UPLOAD=1 to upload public HF data.")
    api = HfApi()
    api.create_repo(repo_id=rid, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=rid,
        repo_type="dataset",
        folder_path=str(stage_root),
        path_in_repo=".",
        commit_message="Upload RASST Speech-LLM SFT dataset (JSONL + stats, audio held out)",
    )


def download_package(manifest: Mapping[str, Any], root: Path, *, dry_run: bool, force: bool) -> None:
    rid = repo_id(manifest)
    target = local_root(manifest, root)
    revision = str(release_data_meta(manifest).get("hf_revision") or "main")
    print(f"[DOWNLOAD_SLM_DATA] {rid}@{revision} -> {target}")
    non_placeholder = []
    if target.exists():
        non_placeholder = [item for item in target.iterdir() if item.name != ".gitkeep"]
    if non_placeholder and not force:
        print(f"[SKIP] target exists: {target}")
        return
    if dry_run:
        print("status=dry_run")
        return
    if os.environ.get("RASST_ALLOW_DOWNLOAD") != "1":
        raise ReleaseSlmError("Set RASST_ALLOW_DOWNLOAD=1 to download HF release data.")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=rid,
        repo_type="dataset",
        revision=revision,
        local_dir=str(target),
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
    p.add_argument("--max-rows", type=int, default=0, help="Stage only the first N rows per JSONL (0 = all). Use for fast validation.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.max_rows < 0:
        raise ReleaseSlmError("--max-rows must be >= 0.")
    root = repo_root()
    manifest_path = Path(args.manifest) if args.manifest else default_manifest(root)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = load_manifest(manifest_path)
    stage_root = Path(args.stage_root).expanduser()
    if not stage_root.is_absolute():
        stage_root = root / stage_root
    if args.action == "prepare":
        prepare_package(manifest, stage_root, force=args.force, max_rows=args.max_rows)
    elif args.action == "upload":
        upload_package(manifest, stage_root, dry_run=not args.execute)
    elif args.action == "download":
        download_package(manifest, root, dry_run=not args.execute, force=args.force)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseSlmError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
