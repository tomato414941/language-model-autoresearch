"""Prepare an experimental tokenizer from the fixed native training prefix on CPU."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
from datetime import datetime, timezone

import rustbpe
import tiktoken
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gpu_config import active_gpu
import prepare
from tools.token_statistics import digest, training_snapshot


def training_texts(max_chars, doc_cap, statistics):
    if max_chars <= 0 or doc_cap <= 0:
        raise ValueError("The character budget and document cap must be positive.")
    ordered = hashlib.sha256()
    statistics.update(documents=0, characters=0, utf8_bytes=0)
    for text in prepare.text_iterator(max_chars=max_chars, doc_cap=doc_cap):
        raw = text.encode("utf-8")
        ordered.update(len(raw).to_bytes(8, "little"))
        ordered.update(raw)
        statistics["documents"] += 1
        statistics["characters"] += len(text)
        statistics["utf8_bytes"] += len(raw)
        yield text
    statistics["ordered_document_sha256"] = ordered.hexdigest()


def build_encoding(texts, vocabulary):
    if vocabulary < 256 + len(prepare.SPECIAL_TOKENS):
        raise ValueError("The vocabulary must contain all byte values and special tokens.")
    trainer = rustbpe.Tokenizer()
    trainer.train_from_iterator(texts, vocabulary - len(prepare.SPECIAL_TOKENS),
                                pattern=prepare.SPLIT_PATTERN)
    ranks = {bytes(key): value for key, value in trainer.get_mergeable_ranks()}
    specials = {name: len(ranks) + index
                for index, name in enumerate(prepare.SPECIAL_TOKENS)}
    encoding = tiktoken.Encoding(name="rustbpe", pat_str=trainer.get_pattern(),
                                 mergeable_ranks=ranks, special_tokens=specials)
    if encoding.n_vocab != vocabulary:
        raise RuntimeError("The training prefix could not fill the requested vocabulary.")
    return encoding


def token_byte_lengths(encoding):
    # Preserve prepare.py's byte-table construction for the reference BPB metric.
    special_set = set(prepare.SPECIAL_TOKENS)
    sizes = []
    for token_id in range(encoding.n_vocab):
        text = encoding.decode([token_id])
        sizes.append(0 if text in special_set else len(text.encode("utf-8")))
    return torch.tensor(sizes, dtype=torch.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--max-chars", type=int, default=1_000_000_000)
    parser.add_argument("--doc-cap", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_chars <= 0 or args.doc_cap <= 0:
        parser.error("--max-chars and --doc-cap must be positive")
    if args.vocab_size < 256 + len(prepare.SPECIAL_TOKENS):
        parser.error("--vocab-size must include all byte values and special tokens")
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = {name: digest(ROOT / name) for name in (
        "prepare.py", "program.md", f"gpus/{active_gpu().identifier}/train.py", "tools/token_statistics.py",
        "tools/prepare_tokenizer.py",
    )}
    corpus = training_snapshot()
    reference_file = Path(prepare.TOKENIZER_DIR) / "tokenizer.pkl"
    reference_sha256 = digest(reference_file)
    statistics = {}
    report = dict(
        started_at=datetime.now(timezone.utc).isoformat(), status="running",
        requested_vocabulary=args.vocab_size, max_chars=args.max_chars,
        doc_cap=args.doc_cap, split="train", corpus_snapshot=corpus,
        source_hashes=source_hashes, reference_tokenizer_sha256=reference_sha256,
        versions={name: importlib.metadata.version(name)
                  for name in ("rustbpe", "tiktoken", "torch")},
        pattern=prepare.SPLIT_PATTERN, special_tokens=prepare.SPECIAL_TOKENS,
        training_prefix=statistics,
        protocol="One native prepare.text_iterator prefix, in source order, excluding the pinned validation shard. The native character budget, per-document cap, split pattern and special tokens are retained. No content-based selection or deduplication.",
        token_bytes_protocol="Unchanged prepare.py construction: decode each token to text, encode it as UTF-8, and count those bytes; special tokens count as zero.",
        purpose="CPU preparation only; no model weights, other model outputs, held-out evaluation or adoption decision.",
    )
    manifest = args.output / "manifest.json"

    def save():
        manifest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        encoding = build_encoding(training_texts(args.max_chars, args.doc_cap, statistics),
                                  args.vocab_size)
        with reference_file.open("rb") as stream:
            reference = pickle.load(stream)
        reference_ranks = reference._mergeable_ranks
        new_ranks = encoding._mergeable_ranks
        matching = sum(new_ranks.get(token) == rank
                       for token, rank in reference_ranks.items())
        roundtrip = "Hello world! Numbers: 123. Unicode: 你好 🌍\nI'm testing."
        assert encoding.decode(encoding.encode_ordinary(roundtrip)) == roundtrip
        tokenizer_file = args.output / "tokenizer.pkl"
        with tokenizer_file.open("wb") as stream:
            pickle.dump(encoding, stream)
        byte_file = args.output / "token_bytes.pt"
        torch.save(token_byte_lengths(encoding), byte_file)
        if training_snapshot() != corpus:
            raise RuntimeError("The training corpus changed during preparation.")
        if digest(reference_file) != reference_sha256:
            raise RuntimeError("The reference tokenizer changed during preparation.")
        if any(digest(ROOT / name) != value for name, value in source_hashes.items()):
            raise RuntimeError("A source changed during preparation.")
        report.update(
            status="complete", finished_at=datetime.now(timezone.utc).isoformat(),
            vocabulary=encoding.n_vocab,
            files={path.name: dict(bytes=path.stat().st_size, sha256=digest(path))
                   for path in (tokenizer_file, byte_file)},
            reference_merge_comparison=dict(
                reference_vocabulary=reference.n_vocab,
                reference_mergeable_tokens=len(reference_ranks),
                tokens_with_identical_rank=matching,
                all_reference_ranks_retained=matching == len(reference_ranks),
                same_pattern=reference._pat_str == encoding._pat_str,
            ),
        )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    save()
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("corpus_snapshot", "source_hashes", "pattern")}), flush=True)


if __name__ == "__main__":
    main()
