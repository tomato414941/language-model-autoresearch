"""Prepare a tokenizer from fixed prefixes of three allowed training sources."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prepare
from tools.prepare_tokenizer import build_encoding, token_byte_lengths
from tools.prepare_training_cache import digest, text_rows, training_sources

SOURCES = ("fineweb", "finemath-4plus", "common-pile")


def mixed_prefix(max_chars, doc_cap, statistics):
    """Use equal character budgets, original file/document order, and the native document cap."""
    if max_chars < len(SOURCES) or doc_cap <= 0:
        raise ValueError("Provide a positive character budget for each source and a positive document cap.")
    base, remainder = divmod(max_chars, len(SOURCES))
    for index, name in enumerate(SOURCES):
        budget = base + (index < remainder)
        paths, provenance = training_sources(name, None)
        case = dict(source=provenance, requested_characters=budget, documents=0, characters=0,
                    utf8_bytes=0, files=[], complete=False)
        statistics.append(case)
        ordered = hashlib.sha256()
        for path in paths:
            case["files"].append(dict(filename=path.name, sha256=digest(path)))
            for text in text_rows(path):
                document = text[:doc_cap]
                raw = document.encode("utf-8")
                ordered.update(len(raw).to_bytes(8, "little"))
                ordered.update(raw)
                case["documents"] += 1
                case["characters"] += len(document)
                case["utf8_bytes"] += len(raw)
                yield document
                if case["characters"] >= budget:
                    break
            if case["characters"] >= budget:
                break
        if case["characters"] < budget:
            raise ValueError("Download enough original training shards to fill every predeclared source prefix.")
        case.update(ordered_document_sha256=ordered.hexdigest(), complete=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--max-chars", type=int, default=1_000_000_000)
    parser.add_argument("--doc-cap", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_chars < len(SOURCES) or args.doc_cap <= 0:
        parser.error("Provide a positive budget for every source and a positive document cap.")
    args.output.mkdir(parents=True, exist_ok=False)
    reference = Path(prepare.TOKENIZER_DIR) / "tokenizer.pkl"
    reference_sha256 = digest(reference)
    statistics = []
    report = dict(status="preparing", started_at=datetime.now(timezone.utc).isoformat(),
                  vocabulary_requested=args.vocab_size, max_chars=args.max_chars, doc_cap=args.doc_cap,
                  source_order=SOURCES, source_prefixes=statistics, split_pattern=prepare.SPLIT_PATTERN,
                  special_tokens=prepare.SPECIAL_TOKENS, reference_tokenizer_sha256=reference_sha256,
                  validation_used=False, quality_evaluations=0, model_weights_loaded=False,
                  hypothesis="Equal character prefixes of FineWeb-Edu, FineMath-4plus and Common Pile may improve math/code representation without increasing the 8192-way output head. Quality and GPU speed remain unmeasured.",
                  protocol="Predeclared equal character budgets with native per-document truncation; consume original training prefixes in order, including empty and repeated documents. No content filtering or evaluation text.")
    manifest = args.output / "manifest.json"

    def save():
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(manifest)

    save()
    try:
        encoding = build_encoding(mixed_prefix(args.max_chars, args.doc_cap, statistics), args.vocab_size)
        if len(statistics) != len(SOURCES) or not all(case["complete"] for case in statistics):
            raise ValueError("Every predeclared training prefix must be consumed.")
        for text in ("Hello world!", "x = 1 + 2\n", "你好 🌍", "\\frac{1}{2}", ""):
            if encoding.decode(encoding.encode_ordinary(text)) != text:
                raise ValueError("The experimental tokenizer must preserve every input byte.")
        tokenizer_file = args.output / "tokenizer.pkl"
        with tokenizer_file.open("xb") as output:
            pickle.dump(encoding, output)
        byte_file = args.output / "token_bytes.pt"
        torch.save(token_byte_lengths(encoding), byte_file)
        report.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(),
                      vocabulary=encoding.n_vocab,
                      files={path.name: dict(sha256=digest(path), bytes=path.stat().st_size)
                             for path in (tokenizer_file, byte_file)},
                      reference_tokenizer_preserved=digest(reference) == reference_sha256)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    save()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
