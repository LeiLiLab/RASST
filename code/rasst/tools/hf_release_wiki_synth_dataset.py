#!/usr/bin/env python3
"""Prepare, upload, and download the public RASST wiki-synthetic retriever data.

The release has three tables and one audio set:

- ``terms/wikidata_terms_top1m.parquet``: the ranked Wikidata term list.
- ``utterances/wiki_synth_utterances.parquet``: every generated sentence, with
  ``utter_id`` matching the retriever training rows.
- ``retriever_train/index.parquet``: the wiki-synthetic retriever training rows
  (term, MFA span inside the chunk, context duration, shard name).
- ``retriever_train/audio/train-NNNNN.tar``: WebDataset shards holding
  ``<key>.flac`` (lossless re-encode of the 16 kHz training WAV) and
  ``<key>.json`` (the same row as the index).

``prepare`` writes the tables, card, and per-shard build plans. ``upload``
builds tar shards batch by batch, commits each batch, checks the remote sizes,
and deletes the local tars. Re-running ``upload`` skips shards already on the
Hub, so an interrupted upload resumes where it stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

WIKI_PREFIX = "wiki_synth_"
WIKI_LINE_MARKER = '"utter_id": "wiki_synth_'
AUDIO_PREFIX = "retriever_train/audio"
TEXT_FILES = (
    "README.md",
    "dataset_manifest.json",
    "terms/wikidata_terms_top1m.parquet",
    "utterances/wiki_synth_utterances.parquet",
    "retriever_train/index.parquet",
)
ROW_FIELDS = (
    "term",
    "term_key",
    "p31_rank",
    "chunk_idx",
    "context_duration_sec",
    "mfa_term_start_in_chunk",
    "mfa_term_end_in_chunk",
    "mfa_term_duration",
    "mfa_term_full_duration",
)


class ReleaseWikiSynthError(RuntimeError):
    pass


def repo_root() -> Path:
    root_text = os.environ.get("RASST_ROOT")
    if root_text:
        return Path(root_text).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def load_manifest(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("metadata"), dict):
        raise ReleaseWikiSynthError(f"Manifest must be an object with metadata: {path}")
    return data


def section(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = manifest["metadata"].get(name)
    if not isinstance(value, dict):
        raise ReleaseWikiSynthError(f"metadata.{name} must be an object.")
    return value


def repo_id(manifest: Mapping[str, Any]) -> str:
    return str(section(manifest, "release_data")["hf_repo_id"])


def write_parquet(records: List[Dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), path, compression="zstd")


def shard_name(index: int) -> str:
    return f"train-{index:05d}"


# --------------------------------------------------------------------------- prepare


def load_terms(path: Path, expected: int) -> List[Dict[str, Any]]:
    terms = json.loads(path.read_text(encoding="utf-8"))
    if len(terms) != expected:
        raise ReleaseWikiSynthError(f"Term count {len(terms)} != expected {expected}: {path}")
    return terms


def load_p31_ranking(path: Path) -> Dict[str, Dict[str, Any]]:
    # Same rule as the training-data builder: one entry per term_key, the last one wins.
    ranking: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            ranking[item["term_key"]] = item
    return ranking


def load_utterances(path: Path, expected: int) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        utts = [json.loads(line) for line in f if line.strip()]
    if len(utts) != expected:
        raise ReleaseWikiSynthError(f"Utterance count {len(utts)} != expected {expected}: {path}")
    return utts


def prepare(manifest: Mapping[str, Any], stage: Path, *, force: bool, source_commit: str) -> None:
    sources = section(manifest, "sources")
    expected = section(manifest, "expected")
    packaging = section(manifest, "packaging")
    samples_per_shard = int(packaging["samples_per_shard"])
    sample_rate = int(packaging["sample_rate"])
    rank_cutoff = int(section(manifest, "hn1024_training")["wiki_rank_cutoff"])

    if stage.exists():
        if not force:
            raise ReleaseWikiSynthError(f"Stage root exists, pass --force to rebuild: {stage}")
        shutil.rmtree(stage)
    plan_dir = stage / "_work" / "plan"
    plan_dir.mkdir(parents=True)

    terms = load_terms(Path(sources["terms_json"]), int(expected["terms"]))
    term_info: Dict[str, Dict[str, Any]] = {}
    for item in terms:
        if item["term"] in term_info:
            raise ReleaseWikiSynthError(f"Duplicate term in term list: {item['term']!r}")
        term_info[item["term"]] = item
    ranking = load_p31_ranking(Path(sources["p31_ranking_jsonl"]))
    print(f"[prepare] terms={len(terms)} p31_ranking_keys={len(ranking)}", flush=True)

    def p31(term_key: str) -> Dict[str, Any]:
        item = ranking.get(term_key)
        if item is None:
            return {"p31_rank": -1, "primary_p31_qid": None, "primary_p31_label": None}
        return {"p31_rank": item["rank"], "primary_p31_qid": item["primary_p31_qid"],
                "primary_p31_label": item["primary_p31_label"] or None}

    utts = load_utterances(Path(sources["utterances_jsonl"]), int(expected["utterances"]))
    # The trainer's utter_id numbering is offset for the second TTS batch, so training rows join
    # to sentences through the TTS wav path, and release ids are sentence row numbers.
    utt_index = {utt["clean_audio_path"]: i for i, utt in enumerate(utts)}
    if len(utt_index) != len(utts):
        raise ReleaseWikiSynthError("TTS wav paths are not unique across utterances")
    print(f"[prepare] utterances={len(utts)}", flush=True)

    audio_root = str(sources["wiki_audio_root"]).rstrip("/") + "/"
    rows: List[Dict[str, Any]] = []
    srcs: List[str] = []
    keys = set()
    used_utts = set()
    hn1024_utts = set()
    hn1024_rows = 0
    total_lines = 0
    with Path(sources["retriever_train_jsonl"]).open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            if WIKI_LINE_MARKER not in line:
                continue
            row = json.loads(line)
            idx = utt_index[row["source_tts_wav_path"]]
            utt = utts[idx]
            if utt["term"] != row["term"]:
                raise ReleaseWikiSynthError(f"{row['utter_id']} term differs from its sentence row: {utt}")
            utter_id = f"{WIKI_PREFIX}{idx:07d}"
            src = row["chunk_audio_path"]
            if not src.startswith(audio_root):
                raise ReleaseWikiSynthError(f"Chunk audio outside wiki_audio_root: {src}")
            key = f"{utter_id}_ctx{row['context_duration_tag']}_chunk{row['chunk_idx']}"
            if "." in key or key in keys:
                raise ReleaseWikiSynthError(f"Bad or duplicate sample key: {key}")
            keys.add(key)
            if p31(row["term_key"])["p31_rank"] != row["p31_rank"]:
                raise ReleaseWikiSynthError(f"{row['utter_id']} p31_rank differs from the P31 ranking file")
            in_hn1024 = not (0 <= row["p31_rank"] and row["p31_rank"] >= rank_cutoff)
            used_utts.add(utter_id)
            if in_hn1024:
                hn1024_utts.add(utter_id)
                hn1024_rows += 1
            record = {"key": key, "utter_id": utter_id}
            record.update({name: row[name] for name in ROW_FIELDS})
            record["utterance"] = utt["utterance"]
            record["variant_idx"] = utt["variant_idx"]
            record["chunk_text"] = row["chunk_src_text"]
            record["in_hn1024_train"] = in_hn1024
            rows.append(record)
            srcs.append(src)
    if total_lines != int(expected["retriever_train_total_rows"]):
        raise ReleaseWikiSynthError(f"Training JSONL rows {total_lines} != expected")
    if len(rows) != int(expected["retriever_train_wiki_rows"]):
        raise ReleaseWikiSynthError(f"Wiki training rows {len(rows)} != expected")
    if hn1024_rows != int(expected["hn1024_wiki_rows"]):
        raise ReleaseWikiSynthError(f"Rows under the hn1024 rank cutoff {hn1024_rows} != expected")
    print(f"[prepare] training_lines={total_lines} wiki_rows={len(rows)} hn1024_rows={hn1024_rows}", flush=True)

    duration_counts: Dict[str, int] = {}
    num_shards = (len(rows) + samples_per_shard - 1) // samples_per_shard
    for shard in range(num_shards):
        lo, hi = shard * samples_per_shard, min(len(rows), (shard + 1) * samples_per_shard)
        name = shard_name(shard)
        with (plan_dir / f"{name}.jsonl").open("w", encoding="utf-8") as out:
            for i in range(lo, hi):
                rows[i]["shard"] = name
                frames = round(rows[i]["context_duration_sec"] * sample_rate)
                meta = {k: v for k, v in rows[i].items() if k != "shard"}
                out.write(json.dumps({"src": srcs[i], "frames": frames, "meta": meta}, ensure_ascii=False) + "\n")
                tag = f"{rows[i]['context_duration_sec']:.2f}"
                duration_counts[tag] = duration_counts.get(tag, 0) + 1
    write_parquet(rows, stage / "retriever_train/index.parquet")

    utterance_records = []
    for i, utt in enumerate(utts):
        info = term_info[utt["term"]]
        utter_id = f"{WIKI_PREFIX}{i:07d}"
        utterance_records.append({
            "utter_id": utter_id,
            "term": utt["term"],
            "term_key": info["term_key"],
            "short_description": info["short_description"],
            "p31_rank": p31(info["term_key"])["p31_rank"],
            "variant_idx": utt["variant_idx"],
            "utterance": utt["utterance"],
            "has_train_chunk": utter_id in used_utts,
            "in_hn1024_train": utter_id in hn1024_utts,
        })
    write_parquet(utterance_records, stage / "utterances/wiki_synth_utterances.parquet")
    term_records = [
        {"term": t["term"], "term_key": t["term_key"], "short_description": t["short_description"], **p31(t["term_key"])}
        for t in terms
    ]
    write_parquet(term_records, stage / "terms/wikidata_terms_top1m.parquet")

    stats = {
        "hf_repo_id": repo_id(manifest),
        "license": section(manifest, "release_data")["license"],
        "source_manifest_event_id": manifest.get("event_id"),
        "source_commit": source_commit,
        "counts": {
            "terms": len(terms),
            "utterances": len(utts),
            "unique_terms_with_utterances": len({u["term"] for u in utts}),
            "utterances_with_train_chunk": len(used_utts),
            "retriever_train_rows": len(rows),
            "hn1024_train_rows": hn1024_rows,
            "terms_without_p31_rank": sum(1 for t in term_records if t["p31_rank"] < 0),
            "retriever_train_rows_by_context_sec": dict(sorted(duration_counts.items())),
            "retriever_train_total_rows_with_gigaspeech": total_lines,
        },
        "packaging": {
            "audio_format": packaging["audio_format"],
            "sample_rate": sample_rate,
            "channels": 1,
            "samples_per_shard": samples_per_shard,
            "num_shards": num_shards,
            "shard_pattern": f"{AUDIO_PREFIX}/train-{{00000..{num_shards - 1:05d}}}.tar",
        },
        "provenance": dict(section(manifest, "provenance")),
        "hn1024_training": dict(section(manifest, "hn1024_training")),
    }
    (stage / "dataset_manifest.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_card(stage, stats)
    print(f"[prepare] status=prepared stage={stage} shards={num_shards}", flush=True)


def write_card(stage: Path, stats: Mapping[str, Any]) -> None:
    counts = stats["counts"]
    pack = stats["packaging"]
    prov = stats["provenance"]
    hn = stats["hn1024_training"]
    by_ctx = "\n".join(
        f"| {sec} s | {n:,} |" for sec, n in counts["retriever_train_rows_by_context_sec"].items()
    )
    text = f"""---
license: {stats['license']}
language:
- en
pretty_name: RASST Wiki-Synthetic Terminology Speech (retriever training data)
size_categories:
- 1M<n<10M
task_categories:
- automatic-speech-recognition
- text-retrieval
tags:
- rasst
- terminology
- speech-retrieval
- streaming-speech-translation
- synthetic-speech
- wikidata
configs:
- config_name: retriever_train
  default: true
  data_files:
  - split: train
    path: {AUDIO_PREFIX}/*.tar
- config_name: retriever_train_index
  data_files:
  - split: train
    path: retriever_train/index.parquet
- config_name: utterances
  data_files:
  - split: train
    path: utterances/wiki_synth_utterances.parquet
- config_name: terms
  data_files:
  - split: train
    path: terms/wikidata_terms_top1m.parquet
---

# RASST Wiki-Synthetic Terminology Speech

This is the wiki-synthetic part of the training data for the RASST speech-to-term
retriever ([`{prov['trained_checkpoint']}`](https://huggingface.co/{prov['trained_checkpoint']})).
It pairs {counts['terms']:,} Wikidata terms with {counts['utterances']:,} English
sentences that contain them, synthesized speech for every sentence, and
{counts['retriever_train_rows']:,} speech chunks with word-aligned term spans built for
retriever training, {counts['hn1024_train_rows']:,} of which the released checkpoint used.

Paper: [RASST](https://arxiv.org/abs/2601.22777). Code: https://github.com/LeiLiLab/RASST

## How it was built

1. **Terms.** {prov['term_source']}. Terms in the held-out evaluation glossaries
   were removed before training.
2. **Sentences.** {prov['utterance_generator']}.
3. **Speech.** {prov['tts']}.
4. **Chunks.** {prov['alignment']}. Each chunk keeps the term span inside it.

## Configs

| Config | Rows | Content |
| --- | ---: | --- |
| `terms` | {counts['terms']:,} | `term`, `term_key`, `short_description`, `p31_rank`, `primary_p31_qid`, `primary_p31_label` |
| `utterances` | {counts['utterances']:,} | `utter_id`, `term`, `short_description`, `variant_idx`, `utterance`, `has_train_chunk`, `in_hn1024_train` |
| `retriever_train_index` | {counts['retriever_train_rows']:,} | one row per training chunk, without audio |
| `retriever_train` | {counts['retriever_train_rows']:,} | WebDataset shards: `<key>.flac` + `<key>.json` |

Training chunks by context length:

| Context | Rows |
| --- | ---: |
{by_ctx}

Each training row has `key`, `utter_id`, `term`, `term_key`, `p31_rank`, `utterance`,
`variant_idx`, `chunk_text` (words inside the chunk), `chunk_idx`,
`context_duration_sec`, the MFA term span `mfa_term_start_in_chunk`,
`mfa_term_end_in_chunk`, `mfa_term_duration`, `mfa_term_full_duration` in seconds,
and `in_hn1024_train`. The released retriever kept rows whose `p31_rank` is below
{hn['wiki_rank_cutoff']:,} (rank -1 means unranked and is kept), which is
{counts['hn1024_train_rows']:,} of the {counts['retriever_train_rows']:,} rows.
Filter on `in_hn1024_train` to reproduce its wiki-synthetic training set.
Audio is {pack['sample_rate']} Hz mono, stored as FLAC (a lossless re-encode of the
PCM16 WAV the trainer read). There are {pack['num_shards']} shards of up to
{pack['samples_per_shard']:,} samples: `{pack['shard_pattern']}`.

## Usage

```python
from datasets import load_dataset

sentences = load_dataset("{stats['hf_repo_id']}", "utterances", split="train")
chunks = load_dataset("{stats['hf_repo_id']}", "retriever_train", split="train", streaming=True)
sample = next(iter(chunks))  # sample["flac"] holds the audio, sample["json"] the row
```

Text only:

```bash
hf download --repo-type dataset {stats['hf_repo_id']} --include "*.parquet" "*.json" "README.md"
```

## Not included

The retriever was also trained on {counts['retriever_train_total_rows_with_gigaspeech'] - counts['retriever_train_rows']:,}
GigaSpeech-derived chunks. GigaSpeech audio cannot be redistributed, so those rows
are not part of this release.

## License and sources

Released under CC BY-NC 4.0 for non-commercial research.

- Terms and descriptions come from [Wikidata](https://www.wikidata.org/) (CC0).
- Sentences were generated with Gemini 2.5 Flash. They are synthetic and may
  contain factual errors about the entities they mention.
- Speech was synthesized with [Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
  using speaker prompts cut from [GigaSpeech](https://github.com/SpeechColab/GigaSpeech),
  whose audio is licensed for non-commercial research and education.
"""
    (stage / "README.md").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- shards


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def add_member(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(payload))


def build_shard(args: Sequence[Any]) -> Dict[str, Any]:
    import soundfile as sf

    stage, name, sample_rate = Path(args[0]), str(args[1]), int(args[2])
    out = stage / AUDIO_PREFIX / f"{name}.tar"
    done = stage / "_work" / "done" / f"{name}.json"
    if out.exists() and done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    done.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{name}.tar.tmp")
    count = 0
    with tarfile.open(tmp, "w", format=tarfile.USTAR_FORMAT) as tar:
        with (stage / "_work" / "plan" / f"{name}.jsonl").open("r", encoding="utf-8") as plan:
            for line in plan:
                item = json.loads(line)
                data, sr = sf.read(item["src"], dtype="int16")
                if sr != sample_rate or data.ndim != 1 or len(data) != item["frames"]:
                    raise ReleaseWikiSynthError(
                        f"Unexpected audio {item['src']}: sr={sr} shape={data.shape} expected_frames={item['frames']}"
                    )
                buf = io.BytesIO()
                sf.write(buf, data, sr, format="FLAC", subtype="PCM_16")
                key = item["meta"]["key"]
                add_member(tar, f"{key}.flac", buf.getvalue())
                add_member(tar, f"{key}.json", json.dumps(item["meta"], ensure_ascii=False).encode("utf-8"))
                count += 1
    os.replace(tmp, out)
    record = {"name": name, "samples": count, "bytes": out.stat().st_size, "sha256": sha256_file(out)}
    done.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return record


# --------------------------------------------------------------------------- upload


def heartbeat(stage: Path, **fields: Any) -> None:
    fields["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = stage / "_work" / "heartbeat.json"
    path.write_text(json.dumps(fields) + "\n", encoding="utf-8")


def commit_with_retry(api: Any, rid: str, operations: List[Any], message: str, attempts: int = 6) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return api.create_commit(repo_id=rid, repo_type="dataset", operations=operations, commit_message=message)
        except Exception as exc:  # network boundary: Hub errors are retried, the last one is raised
            if attempt == attempts:
                raise
            wait = min(600, 30 * 2 ** (attempt - 1))
            print(f"[upload] commit failed ({exc!r}), retry {attempt}/{attempts - 1} in {wait}s", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def verify_remote(api: Any, rid: str, expected_sizes: Mapping[str, int]) -> None:
    infos = api.get_paths_info(rid, list(expected_sizes), repo_type="dataset")
    remote = {info.path: info.size for info in infos}
    for path, size in expected_sizes.items():
        if remote.get(path) != size:
            raise ReleaseWikiSynthError(f"Remote size mismatch for {path}: remote={remote.get(path)} local={size}")


def upload(manifest: Mapping[str, Any], stage: Path, *, execute: bool, workers: int, batch_shards: int, refresh_text: bool) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    rid = repo_id(manifest)
    sample_rate = int(section(manifest, "packaging")["sample_rate"])
    plans = sorted(p.stem for p in (stage / "_work" / "plan").glob("train-*.jsonl"))
    if not plans:
        raise ReleaseWikiSynthError(f"No shard plans under {stage} (run prepare first)")
    for rel in TEXT_FILES:
        if not (stage / rel).is_file():
            raise ReleaseWikiSynthError(f"Missing staged file {stage / rel}")
    print(f"[upload] {stage} -> {rid} shards={len(plans)}", flush=True)
    if not execute:
        print("[upload] status=dry_run (pass --execute with RASST_ALLOW_HF_UPLOAD=1)", flush=True)
        return
    if os.environ.get("RASST_ALLOW_HF_UPLOAD") != "1":
        raise ReleaseWikiSynthError("Set RASST_ALLOW_HF_UPLOAD=1 to upload.")

    api = HfApi()
    owner = rid.split("/", 1)[0]
    if api.whoami()["name"] != owner:
        raise ReleaseWikiSynthError(f"HF token does not belong to {owner}")
    api.create_repo(rid, repo_type="dataset", private=False, exist_ok=True)
    remote = set(api.list_repo_files(rid, repo_type="dataset"))

    if refresh_text or not all(rel in remote for rel in TEXT_FILES):
        ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(stage / rel)) for rel in TEXT_FILES]
        commit_with_retry(api, rid, ops, "Add terms, utterances, retriever index, and dataset card")
        verify_remote(api, rid, {rel: (stage / rel).stat().st_size for rel in TEXT_FILES})
        print("[upload] text files uploaded", flush=True)

    pending = [name for name in plans if f"{AUDIO_PREFIX}/{name}.tar" not in remote]
    print(f"[upload] audio shards pending={len(pending)} done={len(plans) - len(pending)}", flush=True)
    uploaded_log = stage / "_work" / "uploaded.jsonl"
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(workers) as pool:
        for start in range(0, len(pending), batch_shards):
            batch = pending[start:start + batch_shards]
            t0 = time.time()
            records = pool.map(build_shard, [(str(stage), name, sample_rate) for name in batch])
            built = time.time() - t0
            heartbeat(stage, phase="uploading", batch_first=batch[0], batch_last=batch[-1],
                      shards_done=len(plans) - len(pending) + start, shards_total=len(plans))
            ops = [
                CommitOperationAdd(path_in_repo=f"{AUDIO_PREFIX}/{r['name']}.tar",
                                   path_or_fileobj=str(stage / AUDIO_PREFIX / f"{r['name']}.tar"))
                for r in records
            ]
            commit = commit_with_retry(api, rid, ops, f"Add retriever_train audio shards {batch[0]}..{batch[-1]}")
            verify_remote(api, rid, {f"{AUDIO_PREFIX}/{r['name']}.tar": r["bytes"] for r in records})
            with uploaded_log.open("a", encoding="utf-8") as log:
                for r in records:
                    log.write(json.dumps({**r, "commit": commit.oid}) + "\n")
            for r in records:
                (stage / AUDIO_PREFIX / f"{r['name']}.tar").unlink()
            done_count = len(plans) - len(pending) + start + len(batch)
            heartbeat(stage, phase="batch_committed", batch_last=batch[-1], shards_done=done_count, shards_total=len(plans))
            gb = sum(r["bytes"] for r in records) / 1e9
            print(f"[upload] {batch[0]}..{batch[-1]} {gb:.1f}GB build={built:.0f}s total={time.time() - t0:.0f}s "
                  f"done={done_count}/{len(plans)} commit={commit.oid}", flush=True)

    final = set(api.list_repo_files(rid, repo_type="dataset"))
    missing = [name for name in plans if f"{AUDIO_PREFIX}/{name}.tar" not in final]
    if missing:
        raise ReleaseWikiSynthError(f"{len(missing)} shards missing on the Hub after upload: {missing[:5]}")
    revision = api.dataset_info(rid).sha
    heartbeat(stage, phase="complete", shards_done=len(plans), shards_total=len(plans), revision=revision)
    print(f"[upload] status=complete repo={rid} revision={revision}", flush=True)


def download(manifest: Mapping[str, Any], root: Path, *, execute: bool, text_only: bool) -> None:
    from huggingface_hub import snapshot_download

    release = section(manifest, "release_data")
    target = root / str(release["local_path"])
    patterns = ["*.parquet", "*.json", "README.md"] if text_only else None
    print(f"[download] {release['hf_repo_id']} -> {target} text_only={text_only}", flush=True)
    if not execute:
        print("[download] status=dry_run (pass --execute with RASST_ALLOW_DOWNLOAD=1)", flush=True)
        return
    if os.environ.get("RASST_ALLOW_DOWNLOAD") != "1":
        raise ReleaseWikiSynthError("Set RASST_ALLOW_DOWNLOAD=1 to download.")
    snapshot_download(repo_id=str(release["hf_repo_id"]), repo_type="dataset",
                      revision=str(release.get("hf_revision", "main")), local_dir=str(target),
                      allow_patterns=patterns)
    print(f"[download] status=complete local_dir={target}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare", "upload", "download"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, default=root / "code/rasst/manifests/retriever_wiki_synth_dataset.json")
        if name != "download":
            p.add_argument("--stage-root", type=Path, required=True)
    sub.choices["prepare"].add_argument("--force", action="store_true")
    sub.choices["prepare"].add_argument("--source-commit", default=os.environ.get("RASST_SOURCE_COMMIT", "unknown"))
    up = sub.choices["upload"]
    up.add_argument("--execute", action="store_true")
    up.add_argument("--workers", type=int, default=16)
    up.add_argument("--batch-shards", type=int, default=16)
    up.add_argument("--refresh-text", action="store_true")
    down = sub.choices["download"]
    down.add_argument("--execute", action="store_true")
    down.add_argument("--text-only", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    try:
        if args.action == "prepare":
            prepare(manifest, args.stage_root, force=args.force, source_commit=args.source_commit)
        elif args.action == "upload":
            upload(manifest, args.stage_root, execute=args.execute, workers=args.workers,
                   batch_shards=args.batch_shards, refresh_text=args.refresh_text)
        else:
            download(manifest, root, execute=args.execute, text_only=args.text_only)
    except ReleaseWikiSynthError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
