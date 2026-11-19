#!/usr/bin/env python3
"""Rebuild the RASST Speech-LLM SFT audio clips from a local GigaSpeech copy.

The SFT dataset (gavinlaw/rasst-speech-llm-sft-cap16-denoise-ttag) ships
``<lang>/audio_clip_map.parquet`` with one row per clip key: the GigaSpeech
opus file, the start sample, and the number of samples at 16 kHz. This script
cuts every clip and writes it as 16-bit PCM WAV to ``<out-dir>/<key>``, so the
``audios`` entries in the JSONL resolve relative to ``<out-dir>``.

It needs the full-length GigaSpeech opus files (``audio/{audiobook,podcast,
youtube}/P*/*.opus`` from the official download). Samples past the end of an
opus file are written as zeros, as in the original clips.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

SAMPLE_RATE = 16000


def cut_opus(job: Tuple[str, str, List[Tuple[str, int, int]]]) -> int:
    import numpy as np
    import soundfile as sf

    opus_path, out_dir, clips = job
    with sf.SoundFile(opus_path) as src:
        if src.samplerate != SAMPLE_RATE:
            raise RuntimeError(f"{opus_path}: sample rate {src.samplerate}, expected {SAMPLE_RATE}")
        for key, start, frames in clips:
            data = np.zeros(frames, dtype="float32")
            if start < len(src):
                src.seek(start)
                read = src.read(min(frames, len(src) - start), dtype="float32", always_2d=False)
                if read.ndim > 1:
                    read = read.mean(axis=1)
                data[: read.size] = read
            out = Path(out_dir) / key
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out), data, SAMPLE_RATE, subtype="PCM_16")
    return len(clips)


def main(argv: Sequence[str] | None = None) -> int:
    import pyarrow.parquet as pq

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset-dir", type=Path, required=True, help="Local copy of the HF SFT dataset.")
    p.add_argument("--gigaspeech-root", type=Path, required=True, help="Directory that contains GigaSpeech's audio/ tree.")
    p.add_argument("--out-dir", type=Path, default=None, help="Where clips go (default: --dataset-dir).")
    p.add_argument("--langs", nargs="+", default=["de", "ja", "zh"])
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args(argv)
    out_dir = args.out_dir or args.dataset_dir

    jobs: Dict[str, List[Tuple[str, int, int]]] = {}
    for lang in args.langs:
        table = pq.read_table(args.dataset_dir / lang / "audio_clip_map.parquet",
                              columns=["key", "gigaspeech_path", "start_sample", "num_samples"]).to_pylist()
        for row in table:
            jobs.setdefault(row["gigaspeech_path"], []).append((row["key"], row["start_sample"], row["num_samples"]))
    missing = [rel for rel in jobs if not (args.gigaspeech_root / rel).is_file()]
    if missing:
        print(f"ERROR: {len(missing)} GigaSpeech opus files not found under {args.gigaspeech_root}, e.g. {missing[:3]}",
              file=sys.stderr)
        return 1

    total = sum(len(v) for v in jobs.values())
    print(f"cutting {total} clips from {len(jobs)} opus files into {out_dir}", flush=True)
    done = 0
    with cf.ProcessPoolExecutor(args.workers) as pool:
        work = [(str(args.gigaspeech_root / rel), str(out_dir), clips) for rel, clips in jobs.items()]
        for n in pool.map(cut_opus, work, chunksize=8):
            done += n
    print(f"wrote {done} clips", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
