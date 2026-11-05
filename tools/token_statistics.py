"""Build reproducible token-frequency tables from the allowed training stream."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import prepare


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def count_training_tokens(tokenizer, max_documents, checkpoints=(), on_checkpoint=None):
    if max_documents <= 0:
        raise ValueError("The document prefix must be positive.")
    checkpoints = set(checkpoints) | {max_documents}
    if any(point <= 0 or point > max_documents for point in checkpoints):
        raise ValueError("Checkpoints must lie within the document prefix.")
    vocabulary = tokenizer.get_vocab_size()
    bos = tokenizer.get_bos_token_id()
    counts = np.zeros(vocabulary, dtype="<u8")
    documents = tokens = characters = utf8_bytes = 0
    first_epoch = last_epoch = None
    document_hash = hashlib.sha256()
    token_hash = hashlib.sha256()
    stream = prepare._document_batches("train")

    def metadata():
        return dict(documents=documents, tokens=tokens, characters=characters,
                    utf8_bytes=utf8_bytes, first_epoch=first_epoch, last_epoch=last_epoch,
                    vocabulary=vocabulary, bos_token_id=bos,
                    ordered_document_sha256=document_hash.hexdigest(),
                    ordered_token_sequence_sha256=token_hash.hexdigest())

    while documents < max_documents:
        texts, epoch = next(stream)
        texts = texts[:max_documents - documents]
        encoded = tokenizer.encode(texts, prepend=bos)
        if len(encoded) != len(texts):
            raise ValueError("The tokenizer changed the number of documents.")
        for text, token_ids in zip(texts, encoded, strict=True):
            ids = np.asarray(token_ids, dtype=np.int64)
            if ids.ndim != 1 or len(ids) == 0 or ids[0] != bos:
                raise ValueError("Every counted document must start with its BOS token.")
            if ids.min() < 0 or ids.max() >= vocabulary:
                raise ValueError("The tokenizer emitted a token outside its vocabulary.")
            histogram = np.bincount(ids, minlength=vocabulary).astype(counts.dtype)
            counts += histogram
            raw = text.encode("utf-8")
            document_hash.update(len(raw).to_bytes(8, "little"))
            document_hash.update(raw)
            token_hash.update(len(ids).to_bytes(8, "little"))
            token_hash.update(ids.astype("<u4").tobytes())
            documents += 1
            tokens += len(ids)
            characters += len(text)
            utf8_bytes += len(raw)
            if first_epoch is None:
                first_epoch = epoch
            last_epoch = epoch
            if documents in checkpoints and on_checkpoint is not None:
                on_checkpoint(counts, metadata())
    if int(counts.sum()) != tokens:
        raise RuntimeError("The histogram does not account for every encoded token.")
    return counts, metadata()


def training_snapshot():
    return [
        dict(name=Path(path).name, bytes=Path(path).stat().st_size, sha256=digest(path))
        for path in prepare.list_parquet_files()
        if Path(path).name != prepare.VAL_FILENAME
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=65536)
    parser.add_argument("--checkpoint", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-directory", type=Path)
    args = parser.parse_args()
    if args.documents <= 0:
        parser.error("--documents must be positive")
    tokenizer_directory = args.tokenizer_directory or Path(prepare.TOKENIZER_DIR)
    tokenizer = prepare.Tokenizer.from_directory(tokenizer_directory)
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = {name: digest(ROOT / name) for name in
                     ("prepare.py", "program.md", "train.py", "tools/token_statistics.py")}
    snapshot = training_snapshot()
    report = dict(
        started_at=datetime.now(timezone.utc).isoformat(), status="running",
        documents_requested=args.documents,
        checkpoints=sorted(set(args.checkpoint) | {args.documents}),
        split="train", corpus_snapshot=snapshot,
        source_hashes=source_hashes,
        tokenizer_sha256=digest(tokenizer_directory / "tokenizer.pkl"),
        protocol="One fixed document prefix in native training-stream order; encode complete documents with one prepended BOS. Count every occurrence, including repeated and empty text.",
        scope="Raw training-document statistics, before best-fit packing and cropping; not packed input/target frequencies or a model-quality evaluation.",
        snapshots=[],
    )
    manifest = args.output / "manifest.json"

    def save():
        manifest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def checkpoint(counts, metadata):
        path = args.output / f'counts-{metadata["documents"]}.npy'
        np.save(path, counts, allow_pickle=False)
        entry = dict(**metadata, file=path.name, sha256=digest(path), bytes=path.stat().st_size)
        report["snapshots"].append(entry)
        save()
        print(json.dumps(entry), flush=True)

    save()
    try:
        counts, metadata = count_training_tokens(tokenizer, args.documents,
                                                args.checkpoint, checkpoint)
        if training_snapshot() != snapshot:
            raise RuntimeError("The training corpus changed during counting.")
        if any(digest(ROOT / name) != value for name, value in source_hashes.items()):
            raise RuntimeError("A source changed during counting.")
        path = args.output / "counts.npy"
        np.save(path, counts, allow_pickle=False)
        report.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(),
                      totals=metadata, counts_file=dict(name=path.name, sha256=digest(path),
                                                       bytes=path.stat().st_size),
                      observed_token_ids=int(np.count_nonzero(counts)))
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    save()
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("snapshots", "corpus_snapshot", "source_hashes")}), flush=True)


if __name__ == "__main__":
    main()

