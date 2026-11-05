"""Prepare immutable token lookups from the allowed training shards on the CPU."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import islice
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prepare


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def training_sources(name, count):
    if count <= 0:
        raise ValueError("Prepare a positive prefix of allowed training shards.")
    if name == "fineweb":
        paths = [Path(path) for path in prepare.list_parquet_files()
                 if Path(path).name != prepare.VAL_FILENAME]
        repository, revision = prepare.BASE_URL.rsplit("/resolve/", 1)
    elif name in prepare.FINEMATH_SUBSETS:
        paths = sorted((Path(prepare.FINEMATH_DIR) / name).glob("*.parquet"))
        repository, revision = prepare.FINEMATH_REPOSITORY, prepare.FINEMATH_REVISION
    elif name == "common-pile":
        paths = sorted(Path(prepare.COMMON_PILE_DIR).glob("*.json.gz"))
        repository, revision = prepare.COMMON_PILE_REPOSITORY, prepare.COMMON_PILE_REVISION
    else:
        raise ValueError("Choose a source allowed by prepare.py.")
    if len(paths) < count:
        raise ValueError("Download the requested training-shard prefix before preparing its cache.")
    return paths[:count], dict(name=name, repository=repository, revision=revision)


def text_rows(path):
    if path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=["text"]):
            yield from batch.column("text").to_pylist()
    else:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                yield json.loads(line)["text"]


def build_cache(texts, encoding, destination, tokenizer_sha256, source, threads=2):
    """Cache tokenization only; callers still consume every original training document."""
    if threads <= 0:
        raise ValueError("Use a positive tokenizer thread count.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    bos = encoding.encode_single_token(prepare.BOS_TOKEN)
    dtype = "<u2" if encoding.n_vocab <= 65536 else "<u4"
    report = dict(format_version=1, complete=False, source=source, tokenizer_sha256=tokenizer_sha256,
                  vocabulary=encoding.n_vocab, bos_id=bos, token_dtype=dtype,
                  started_at=datetime.now(timezone.utc).isoformat(), documents=0, tokens=0,
                  text_bytes=0, training_only=True, validation_used=False,
                  quality_evaluations=0, model_weights_loaded=False)
    manifest_path = destination / "manifest.json"

    def save():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(manifest_path)

    save()
    locations = {}
    ordered_text = hashlib.sha256()
    ordered_tokens = hashlib.sha256()
    unique_tokens = 0
    iterator = iter(texts)
    with (destination / "tokens.bin").open("xb", buffering=16 * 1024 * 1024) as output:
        while batch := list(islice(iterator, 128)):
            if not all(isinstance(text, str) for text in batch):
                raise ValueError("Every allowed document must contain text.")
            encoded = encoding.encode_ordinary_batch(batch, num_threads=threads)
            for text, tokens in zip(batch, encoded, strict=True):
                raw = text.encode("utf-8")
                key = hashlib.sha256(raw).digest()
                ids = np.asarray([bos, *tokens], dtype=dtype)
                if ids.min() < 0 or ids.max() >= encoding.n_vocab:
                    raise ValueError("A document contains a token outside the vocabulary.")
                payload = ids.tobytes()
                ordered_text.update(len(raw).to_bytes(8, "little"))
                ordered_text.update(raw)
                ordered_tokens.update(len(ids).to_bytes(8, "little"))
                ordered_tokens.update(payload)
                if key not in locations:
                    locations[key] = (unique_tokens, len(ids))
                    output.write(payload)
                    unique_tokens += len(ids)
                elif locations[key][1] != len(ids):
                    raise ValueError("The same document fingerprint has inconsistent tokenization.")
                report["documents"] += 1
                report["tokens"] += len(ids)
                report["text_bytes"] += len(raw)
            if report["documents"] % 12800 == 0:
                save()
                print(json.dumps({key: report[key] for key in ("documents", "tokens", "text_bytes")}), flush=True)
    if not locations:
        raise ValueError("A token cache requires at least one training document.")
    keys = sorted(locations)
    np.save(destination / "fingerprints.npy", np.asarray(keys, dtype="S32"), allow_pickle=False)
    np.save(destination / "locations.npy", np.asarray([locations[key] for key in keys], dtype="<u8"),
            allow_pickle=False)
    report.update(unique_documents=len(keys), unique_tokens=unique_tokens,
                  ordered_text_sha256=ordered_text.hexdigest(),
                  ordered_tokens_sha256=ordered_tokens.hexdigest(),
                  files={name: dict(sha256=digest(destination / name), bytes=(destination / name).stat().st_size)
                         for name in ("tokens.bin", "fingerprints.npy", "locations.npy")},
                  complete=True, finished_at=datetime.now(timezone.utc).isoformat())
    save()
    return report


class TokenLookup:
    """Return read-only BOS-prefixed arrays, with explicit misses for online encoding."""

    def __init__(self, directory, tokenizer_sha256, verify_files=False):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        report = self.manifest
        if report["format_version"] != 1 or not report["complete"]:
            raise ValueError("The training token cache is incomplete.")
        if report["tokenizer_sha256"] != tokenizer_sha256:
            raise ValueError("Use the tokenizer that prepared these exact token IDs.")
        if verify_files:
            for name, metadata in report["files"].items():
                if digest(directory / name) != metadata["sha256"]:
                    raise ValueError("A training token cache file is incomplete.")
        self.fingerprints = np.load(directory / "fingerprints.npy", mmap_mode="r", allow_pickle=False)
        self.locations = np.load(directory / "locations.npy", mmap_mode="r", allow_pickle=False)
        self.tokens = np.memmap(directory / "tokens.bin", mode="r", dtype=report["token_dtype"])
        if (len(self.tokens) != report["unique_tokens"]
                or self.fingerprints.shape != (report["unique_documents"],)
                or self.locations.shape != (report["unique_documents"], 2)):
            raise ValueError("The token lookup does not match its completed manifest.")

    def lookup_many(self, texts):
        keys = np.asarray([hashlib.sha256(text.encode("utf-8")).digest() for text in texts], dtype="S32")
        positions = np.searchsorted(self.fingerprints, keys)
        result = []
        for key, position in zip(keys, positions, strict=True):
            if position == len(self.fingerprints) or self.fingerprints[position] != key:
                result.append(None)
                continue
            start, length = map(int, self.locations[position])
            tokens = self.tokens[start:start + length]
            if len(tokens) != length or tokens[0] != self.manifest["bos_id"]:
                raise ValueError("A cached document is incomplete.")
            result.append(tokens)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("fineweb", *prepare.FINEMATH_SUBSETS, "common-pile"), required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--tokenizer-directory", type=Path, default=Path(prepare.TOKENIZER_DIR))
    parser.add_argument("--tokenizer-threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths, source = training_sources(args.source, args.shards)
    with (args.tokenizer_directory / "tokenizer.pkl").open("rb") as stream:
        encoding = pickle.load(stream)
    tokenizer_sha256 = digest(args.tokenizer_directory / "tokenizer.pkl")
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = []
    for path in paths:
        source_file = dict(source, filename=path.name, sha256=digest(path))
        destination = args.output / path.name.removesuffix(".parquet").removesuffix(".json.gz")
        report = build_cache(text_rows(path), encoding, destination, tokenizer_sha256, source_file,
                             args.tokenizer_threads)
        if path.suffix == ".parquet" and report["documents"] != pq.ParquetFile(path).metadata.num_rows:
            raise RuntimeError("Every original training document must be represented.")
        inventory.append(dict(directory=destination.name, manifest_sha256=digest(destination / "manifest.json"),
                              source=source_file, documents=report["documents"], tokens=report["tokens"]))
        print(json.dumps(inventory[-1]), flush=True)
    manifest = dict(format_version=1, complete=True, source=source, tokenizer_sha256=tokenizer_sha256,
                    shards=inventory, training_only=True, validation_used=False, quality_evaluations=0,
                    data_selection="Complete lexicographic training-shard prefix, without content filtering.")
    with (args.output / "manifest.json").open("x") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
