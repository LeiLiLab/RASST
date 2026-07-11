#!/usr/bin/env python3
"""Extract a pre-event ACL glossary from paper PDFs with a pinned Gemini model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROMPT_TEMPLATE = """You are building a pre-event glossary for a technical ACL conference talk.

Extract technical terms from the paper excerpt below. Select domain-specific NLP,
speech, and machine-learning concepts that are likely to be spoken in a talk. Include
acronyms, named methods, datasets, metrics, and substantive multiword concepts. Do not
include authors, affiliations, citations, section headings, or generic words.

For every English term, provide its conventional technical translation in Simplified
Chinese (zh), German (de), and Japanese (ja). Keep an acronym unchanged when that is
the conventional target-language form.

Return only a JSON array with this exact shape:
[
  {
    "term": "attention mechanism",
    "target_translations": {
      "zh": "注意力机制",
      "de": "Aufmerksamkeitsmechanismus",
      "ja": "注意機構"
    }
  }
]

Constraints:
- Prefer precise terms that would help an interpreter prepare for this specific talk.
- Do not consult or infer any evaluation annotation; use only the supplied paper text.
- Deduplicate terms within this excerpt.
- Every item must contain non-empty zh, de, and ja translations.

PAPER EXCERPT ({chunk_label}):
{paper_text}
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_text(text: Any) -> str:
    return " ".join(str(text or "").replace("\x00", " ").split())


def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required for paper-text extraction") from exc
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(part for part in pages if part.strip()).replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 1_000:
        raise ValueError(f"Extracted paper text is unexpectedly short ({len(text)} chars): {path}")
    return text


def split_text(text: str, chunk_chars: int, max_chunks: int) -> List[str]:
    if chunk_chars < 1_000 or max_chunks < 1:
        raise ValueError("chunk_chars must be >=1000 and max_chunks must be >=1")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_chars = 0
    for paragraph in paragraphs:
        addition = len(paragraph) + (2 if current else 0)
        if current and current_chars + addition > chunk_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_chars = 0
        if len(paragraph) > chunk_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_chars = 0
            chunks.extend(
                paragraph[start : start + chunk_chars]
                for start in range(0, len(paragraph), chunk_chars)
            )
            continue
        current.append(paragraph)
        current_chars += addition
    if current:
        chunks.append("\n\n".join(current))
    if len(chunks) <= max_chunks:
        return chunks
    if max_chunks == 1:
        return [chunks[0]]
    indices = [round(index * (len(chunks) - 1) / (max_chunks - 1)) for index in range(max_chunks)]
    return [chunks[index] for index in indices]


def parse_json_array(text: str) -> List[Dict[str, Any]]:
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start = candidate.find("[")
    end = candidate.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("Gemini response does not contain a JSON array")
    value = json.loads(candidate[start : end + 1])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("Gemini response JSON root must be an array of objects")
    return value


def validate_and_merge_terms(raw_terms: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in raw_terms:
        term = normalise_text(item.get("term"))
        translations = item.get("target_translations")
        if not term or not isinstance(translations, dict):
            raise ValueError(f"Malformed glossary item: {item!r}")
        cleaned = {lang: normalise_text(translations.get(lang)) for lang in ("zh", "de", "ja")}
        if any(not cleaned[lang] for lang in cleaned):
            raise ValueError(f"Missing target translation for term {term!r}: {cleaned}")
        key = term.casefold()
        if key in merged:
            continue
        merged[key] = {
            "term": term,
            "target_translations": cleaned,
            "source": "gemini_paper_extracted",
        }
    if not merged:
        raise ValueError("Gemini returned no valid glossary terms")
    return merged


def model_metadata(client: Any, model: str) -> Dict[str, Any]:
    try:
        info = client.models.get(model=model)
    except Exception as exc:
        return {"lookup_status": f"unavailable:{type(exc).__name__}", "requested_model": model}
    fields = ("name", "display_name", "description", "version", "input_token_limit", "output_token_limit")
    return {
        "lookup_status": "ok",
        "requested_model": model,
        **{field: getattr(info, field, None) for field in fields},
    }


def generate_terms(
    *,
    client: Any,
    types_module: Any,
    model: str,
    paper_text: str,
    chunk_label: str,
    temperature: float,
    max_output_tokens: int,
) -> tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    prompt = PROMPT_TEMPLATE.format(chunk_label=chunk_label, paper_text=paper_text)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types_module.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        ),
    )
    response_text = str(response.text or "")
    usage = getattr(response, "usage_metadata", None)
    usage_dict = {
        key: getattr(usage, key, None)
        for key in (
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
            "thoughts_token_count",
        )
    }
    return parse_json_array(response_text), response_text, usage_dict


def process_paper(
    *,
    client: Any,
    types_module: Any,
    paper: Path,
    output_dir: Path,
    model: str,
    chunk_chars: int,
    max_chunks: int,
    temperature: float,
    max_output_tokens: int,
    sleep_seconds: float,
) -> Dict[str, Any]:
    text = extract_pdf_text(paper)
    chunks = split_text(text, chunk_chars, max_chunks)
    raw_terms: List[Dict[str, Any]] = []
    response_rows: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        terms, response_text, usage = generate_terms(
            client=client,
            types_module=types_module,
            model=model,
            paper_text=chunk,
            chunk_label=f"{index + 1}/{len(chunks)}",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        raw_terms.extend(terms)
        response_rows.append(
            {
                "paper_id": paper.stem,
                "chunk_index": index,
                "chunk_sha256": sha256_bytes(chunk.encode("utf-8")),
                "prompt_sha256": sha256_bytes(
                    PROMPT_TEMPLATE.format(
                        chunk_label=f"{index + 1}/{len(chunks)}",
                        paper_text=chunk,
                    ).encode("utf-8")
                ),
                "usage": usage,
                "response_text": response_text,
            }
        )
        if sleep_seconds > 0 and index + 1 < len(chunks):
            time.sleep(sleep_seconds)

    glossary = validate_and_merge_terms(raw_terms)
    glossary_dir = output_dir / "glossaries"
    response_dir = output_dir / "raw_responses"
    glossary_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)
    glossary_path = glossary_dir / f"{paper.stem}.json"
    responses_path = response_dir / f"{paper.stem}.jsonl"
    glossary_path.write_text(
        json.dumps(glossary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    responses_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in response_rows),
        encoding="utf-8",
    )
    return {
        "paper_id": paper.stem,
        "pdf_path": str(paper.resolve()),
        "pdf_sha256": sha256_file(paper),
        "extracted_text_sha256": sha256_bytes(text.encode("utf-8")),
        "extracted_text_chars": len(text),
        "chunks": len(chunks),
        "raw_term_rows": len(raw_terms),
        "unique_terms": len(glossary),
        "glossary_path": str(glossary_path.resolve()),
        "glossary_sha256": sha256_file(glossary_path),
        "raw_responses_path": str(responses_path.resolve()),
        "raw_responses_sha256": sha256_file(responses_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--paper-ids", nargs="*", default=[])
    parser.add_argument("--chunk-chars", type=int, default=25_000)
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=8_192)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    args = parser.parse_args()

    try:
        from google import genai
        from google.genai import types
        import google.genai as google_genai
    except ImportError as exc:
        raise RuntimeError("google-genai is required for Gemini extraction") from exc

    api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"Gemini API key file is empty: {args.api_key_file}")
    papers = sorted(args.papers_dir.glob("*.pdf"))
    if args.paper_ids:
        requested = set(args.paper_ids)
        papers = [paper for paper in papers if paper.stem in requested]
        missing = requested - {paper.stem for paper in papers}
        if missing:
            raise FileNotFoundError(f"Requested paper IDs are missing: {sorted(missing)}")
    if not papers:
        raise FileNotFoundError(f"No PDF papers found in {args.papers_dir}")

    client = genai.Client(api_key=api_key)
    manifest: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_metadata": model_metadata(client, args.model),
        "sdk": "google-genai",
        "sdk_version": getattr(google_genai, "__version__", "unknown"),
        "python": platform.python_version(),
        "prompt_sha256": sha256_bytes(PROMPT_TEMPLATE.encode("utf-8")),
        "parameters": {
            "chunk_chars": args.chunk_chars,
            "max_chunks": args.max_chunks,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
        },
        "data_access_policy": {
            "inputs": "associated ACL paper PDFs only",
            "excluded": [
                "speech transcript",
                "reference translation",
                "gold term tags",
                "gold evaluation glossary",
            ],
            "manual_filtering": False,
            "normalization": "whitespace normalization and case-insensitive exact deduplication only",
        },
        "papers": [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, paper in enumerate(papers):
        manifest["papers"].append(
            process_paper(
                client=client,
                types_module=types,
                paper=paper,
                output_dir=args.output_dir,
                model=args.model,
                chunk_chars=args.chunk_chars,
                max_chunks=args.max_chunks,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                sleep_seconds=args.sleep_seconds,
            )
        )
        if args.sleep_seconds > 0 and index + 1 < len(papers):
            time.sleep(args.sleep_seconds)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "papers": len(papers)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
