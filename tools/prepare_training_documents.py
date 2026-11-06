"""Prepare ordered training-token caches with every original document and metadata field."""

import argparse
from array import array
from bisect import bisect_right
import base64
import gzip
import hashlib
from itertools import islice
import json
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prepare


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def metadata_bytes(record):
    metadata = {key: value for key, value in record.items() if key != "text"}
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def framed_update(value, payload):
    value.update(len(payload).to_bytes(8, "little"))
    value.update(payload)


def build(records, tokenizer, destination, tokenizer_sha256, source, batch_size=128,
          threads=2, progress=None):
    if batch_size <= 0 or threads <= 0:
        raise ValueError("Use positive document batches and tokenizer thread counts.")
    vocabulary, bos = tokenizer.get_vocab_size(), tokenizer.get_bos_token_id()
    if not 0 < vocabulary <= 2**32 or not 0 <= bos < vocabulary:
        raise ValueError("Use a vocabulary and BOS representable by unsigned 32-bit IDs.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    dtype = "<u2" if vocabulary <= 65536 else "<u4"
    offsets = array("Q", [0])
    text_hash, token_hash, meta_hash = (hashlib.sha256() for _ in range(3))
    report = dict(
        format_version=1, complete=False, started_at=time.time(), source=source,
        tokenizer_sha256=tokenizer_sha256, vocabulary=vocabulary, bos=bos,
        token_dtype=dtype, offset_dtype="<u8", documents=0, tokens=0, text_bytes=0,
        empty_documents=0, metadata_records=0, training_only=True,
        validation_used=False, quality_evaluations=0,
        document_order="Complete original file order, including empty/repeated documents.",
    )
    manifest = destination / "manifest.json"

    def save():
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(manifest)

    save()
    rows = iter(records)
    with (destination / "tokens.bin").open("xb", buffering=16*1024**2) as output, \
            gzip.open(destination / "metadata.jsonl.gz", "wb", compresslevel=6) as metadata:
        while batch := list(islice(rows, batch_size)):
            texts = [record["text"] for record in batch]
            if not all(isinstance(text, str) for text in texts):
                raise ValueError("Every original document must contain text.")
            encoded = tokenizer.encode(texts, prepend=bos, num_threads=threads)
            for record, ids in zip(batch, encoded, strict=True):
                if not ids or ids[0] != bos or min(ids) < 0 or max(ids) >= vocabulary:
                    raise ValueError("Every document must retain BOS and in-vocabulary token IDs.")
                raw = record["text"].encode("utf-8")
                payload = np.asarray(ids, dtype=dtype).tobytes()
                meta = metadata_bytes(record)
                output.write(payload)
                metadata.write(meta + b"\n")
                offsets.append(offsets[-1] + len(ids))
                framed_update(text_hash, raw)
                token_hash.update(len(ids).to_bytes(8, "little"))
                token_hash.update(payload)
                framed_update(meta_hash, meta)
                report["documents"] += 1
                report["tokens"] += len(ids)
                report["text_bytes"] += len(raw)
                report["empty_documents"] += not record["text"]
                report["metadata_records"] += 1
            if report["documents"] % 12800 == 0:
                save()
                if progress:
                    progress(report)
    if not report["documents"]:
        raise ValueError("Prepare at least one original training document.")
    np.asarray(offsets, dtype="<u8").tofile(destination / "offsets.bin")
    report.update(
        complete=True, finished_at=time.time(), ordered_text_sha256=text_hash.hexdigest(),
        ordered_tokens_sha256=token_hash.hexdigest(), ordered_metadata_sha256=meta_hash.hexdigest(),
        files={name: dict(bytes=(destination / name).stat().st_size,
                          sha256=digest(destination / name))
               for name in ("tokens.bin", "offsets.bin", "metadata.jsonl.gz")},
    )
    save()
    return report


class Documents:
    """Read complete documents as readonly views into indexed token storage."""

    # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/indexed_dataset.py
    # https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
    def __init__(self, directory, tokenizer_sha256, verify_files=False):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        report = self.manifest
        if (report["format_version"] != 1 or not report["complete"]
                or not report["training_only"] or report["validation_used"]
                or report["tokenizer_sha256"] != tokenizer_sha256):
            raise ValueError("Use a complete training cache bound to this tokenizer.")
        dtype = "<u2" if report["vocabulary"] <= 65536 else "<u4"
        if report["token_dtype"] != dtype or report["offset_dtype"] != "<u8":
            raise ValueError("Use the declared unsigned token and offset representation.")
        for name in ("tokens.bin", "offsets.bin", "metadata.jsonl.gz"):
            path = self.directory / name
            expected = report["files"][name]
            if path.stat().st_size != expected["bytes"]:
                raise ValueError("A training cache file has the wrong size.")
            if verify_files and digest(path) != expected["sha256"]:
                raise ValueError("A training cache file differs from its manifest.")
        self.offsets = np.memmap(self.directory / "offsets.bin", dtype="<u8", mode="r")
        self.tokens = np.memmap(self.directory / "tokens.bin", dtype=dtype, mode="r")
        if (len(self.offsets) != report["documents"] + 1 or not report["documents"]
                or self.offsets[0] != 0 or self.offsets[-1] != len(self.tokens)
                or len(self.tokens) != report["tokens"]
                or not bool((self.offsets[1:] > self.offsets[:-1]).all())):
            raise ValueError("Every original document must retain its full ordered offset range.")

    def __len__(self):
        return self.manifest["documents"]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        begin, end = map(int, self.offsets[index:index + 2])
        row = self.tokens[begin:end]
        if row[0] != self.manifest["bos"]:
            raise ValueError("Every original document must retain its leading BOS token.")
        return row


class ChainedDocuments:
    """Read existing cache segments in the declared complete source-file order."""

    def __init__(self, directories, tokenizer_sha256, source, verify_files=False):
        self.caches = [Documents(path, tokenizer_sha256, verify_files=verify_files) for path in directories]
        if not self.caches:
            raise ValueError("Use at least one complete training cache segment.")
        fields = ("filename", "bytes", "sha256")
        expected = [{key: entry[key] for key in fields} for entry in source["files"]]
        names = [entry["filename"] for entry in expected]
        if not names or names != sorted(set(names)):
            raise ValueError("Declare the original unique source files in their native order.")
        actual, self.ends = [], []
        shape = {key: self.caches[0].manifest[key] for key in ("vocabulary", "bos", "token_dtype")}
        for cache in self.caches:
            origin = cache.manifest["source"]
            if any(origin.get(key) != source.get(key) for key in ("name", "repository", "revision", "license")):
                raise ValueError("Every cache segment must belong to the same pinned training source.")
            if any(cache.manifest[key] != value for key, value in shape.items()):
                raise ValueError("Every cache segment must use the same token representation.")
            actual.extend({key: entry[key] for key in fields} for entry in origin["files"])
            self.ends.append((self.ends[-1] if self.ends else 0) + len(cache))
        if actual != expected:
            raise ValueError("Cache segments must cover the declared complete source prefix in order.")

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        segment = bisect_right(self.ends, index)
        start = self.ends[segment - 1] if segment else 0
        return self.caches[segment][index - start]

    def batches(self, batch_size=128):
        if batch_size <= 0:
            raise ValueError("Use positive document batches.")
        epoch = 1
        while True:
            for start in range(0, len(self), batch_size):
                yield [self[index] for index in range(start, min(start + batch_size, len(self)))], epoch
            epoch += 1


def verify(records, tokenizer, directory, tokenizer_sha256, batch_size=113, progress=None):
    if batch_size <= 0:
        raise ValueError("Use a positive verification batch size.")
    directory = Path(directory)
    cache = Documents(directory, tokenizer_sha256, verify_files=True)
    text_hash, token_hash, meta_hash = (hashlib.sha256() for _ in range(3))
    rows = iter(records)
    documents = tokens = text_bytes = empty = 0
    with gzip.open(directory / "metadata.jsonl.gz", "rb") as metadata:
        while batch := list(islice(rows, batch_size)):
            expected = tokenizer.encode([record["text"] for record in batch],
                                        prepend=tokenizer.get_bos_token_id(), num_threads=1)
            for record, native in zip(batch, expected, strict=True):
                cached = cache[documents]
                np.testing.assert_array_equal(cached, np.asarray(native, dtype=cache.manifest["token_dtype"]))
                if tokenizer.decode(cached[1:].tolist()) != record["text"]:
                    raise ValueError("The cached document must decode to the complete original text.")
                raw, meta = record["text"].encode("utf-8"), metadata_bytes(record)
                loaded = json.loads(next(metadata))
                if metadata_bytes(dict(loaded, text="")) != meta:
                    raise ValueError("Every original metadata field must be retained in order.")
                framed_update(text_hash, raw)
                framed_update(meta_hash, meta)
                token_hash.update(len(cached).to_bytes(8, "little"))
                token_hash.update(cached.tobytes())
                documents += 1
                tokens += len(cached)
                text_bytes += len(raw)
                empty += not record["text"]
            if documents % 14464 == 0 and progress:
                progress(dict(documents=documents, tokens=tokens))
        if metadata.readline() != b"":
            raise ValueError("The cache must have exactly one metadata record per original document.")
    for field, value in (("documents", documents), ("tokens", tokens),
                         ("text_bytes", text_bytes), ("empty_documents", empty)):
        if cache.manifest[field] != value:
            raise ValueError("The original documents and cache counts differ.")
    for field, value in (("ordered_text_sha256", text_hash), ("ordered_tokens_sha256", token_hash),
                         ("ordered_metadata_sha256", meta_hash)):
        if cache.manifest[field] != value.hexdigest():
            raise ValueError("The original documents and cache order differ.")
    return dict(documents=documents, tokens=tokens, text_bytes=text_bytes, empty_documents=empty,
                every_original_document_and_token_matches=True, metadata_records_match=True,
                every_text_roundtrips_through_native_decode=True)


def compress_payload(directory):
    directory = Path(directory)
    files = {}
    for name in ("tokens.bin", "offsets.bin"):
        path = directory / name
        target = path.with_suffix(path.suffix + ".gz")
        with path.open("rb") as source, target.open("xb") as output:
            with gzip.GzipFile(filename="", fileobj=output, mode="wb", compresslevel=6, mtime=0) as compressed:
                shutil.copyfileobj(source, compressed, 1024**2)
        restored = hashlib.sha256()
        with gzip.open(target, "rb") as source:
            for block in iter(lambda: source.read(1024**2), b""):
                restored.update(block)
        if restored.hexdigest() != digest(path):
            raise ValueError("The compressed cache must roundtrip to the original token and offset bytes.")
        files[target.name] = dict(bytes=target.stat().st_size, sha256=digest(target),
                                  roundtrip_verified=True)
    path = directory / "metadata.jsonl.gz"
    files[path.name] = dict(bytes=path.stat().st_size, sha256=digest(path))
    return dict(files=files, compressed_payload_bytes=sum(row["bytes"] for row in files.values()))


def source_files(name, count):
    if not isinstance(count, int) or count <= 0:
        raise ValueError("Prepare a positive prefix of original training shards.")
    if name in prepare.FINEMATH_SUBSETS:
        source = dict(
            repository=prepare.FINEMATH_REPOSITORY, revision=prepare.FINEMATH_REVISION,
            subset=name, directory=Path(prepare.FINEMATH_DIR) / name,
            license="ODC-By-1.0", pattern=re.escape(name) + r"/train-\d+-of-\d+\.parquet",
        )
    elif name == "common-pile":
        source = dict(
            repository=prepare.COMMON_PILE_REPOSITORY, revision=prepare.COMMON_PILE_REVISION,
            subset="", directory=Path(prepare.COMMON_PILE_DIR),
            license="Original per-file open licenses", pattern=r"stack-edu-\d+\.json\.gz",
        )
    else:
        raise ValueError("Choose a pinned allowed extra training source.")
    entries = prepare._additional_manifest(source)
    if count > len(entries):
        raise ValueError("The requested prefix exceeds this pinned source.")
    paths, files = [], []
    for entry in entries[:count]:
        path = source["directory"] / Path(entry["path"]).name
        if not prepare._additional_shard_matches(path, entry):
            raise ValueError("Download the complete pinned shard prefix before preparing its cache.")
        metadata = dict(filename=path.name, bytes=entry["bytes"], sha256=entry["sha256"])
        if path.suffix == ".parquet":
            schema = pq.ParquetFile(path).schema_arrow.serialize().to_pybytes()
            metadata["parquet_schema_base64"] = base64.b64encode(schema).decode("ascii")
        paths.append(path)
        files.append(metadata)
    origin = dict(name=name, repository=source["repository"], revision=source["revision"],
                  license=source["license"], files=files)
    return paths, origin


def source_records(paths, inventory=None):
    def records(path):
        if path.suffix == ".parquet":
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                yield from batch.to_pylist()
        else:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    yield json.loads(line)
    for index, path in enumerate(paths):
        count = 0
        for record in records(path):
            count += 1
            yield record
        if inventory is not None:
            inventory[index]["documents"] = count


def prepare_documents(name, count, tokenizer_directory, destination, threads=2, compress=False):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    paths, origin = source_files(name, count)
    tokenizer_directory = Path(tokenizer_directory)
    tokenizer_sha256 = digest(tokenizer_directory / "tokenizer.pkl")
    tokenizer = prepare.Tokenizer.from_directory(tokenizer_directory)

    def progress(state):
        print(json.dumps({key: state[key] for key in ("documents", "tokens")}), flush=True)

    result = build(source_records(paths, origin["files"]), tokenizer, destination, tokenizer_sha256, origin,
                   threads=threads, progress=progress)
    manifest = destination / "manifest.json"

    def save():
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(manifest)

    try:
        if all(path.suffix == ".parquet" for path in paths):
            if result["documents"] != sum(pq.ParquetFile(path).metadata.num_rows for path in paths):
                raise ValueError("Every row from the complete original shard prefix must be retained.")
        validation = verify(source_records(paths), tokenizer, destination, tokenizer_sha256, progress=progress)
        for path, expected in zip(paths, origin["files"], strict=True):
            if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
                raise ValueError("An original source file changed during preparation.")
        if digest(tokenizer_directory / "tokenizer.pkl") != tokenizer_sha256:
            raise ValueError("The tokenizer changed during preparation.")
        result.update(verified_against_original=True, validation=validation)
        if compress:
            result["compression"] = compress_payload(destination)
        save()
    except BaseException as error:
        result.update(complete=False, preparation_error_type=type(error).__name__)
        save()
        raise
    summary = dict(source=origin, output=destination.name, tokenizer_sha256=tokenizer_sha256,
                   manifest_sha256=digest(manifest), validation=validation,
                   model_training=False, quality_evaluations=0)
    if compress:
        summary["compression"] = result["compression"]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=(*prepare.FINEMATH_SUBSETS, "common-pile"), required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--tokenizer-directory", type=Path, default=Path(prepare.TOKENIZER_DIR))
    parser.add_argument("--tokenizer-threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compress", action="store_true", help="Also create roundtrip-verified gzip copies.")
    args = parser.parse_args()
    result = prepare_documents(args.source, args.shards, args.tokenizer_directory, args.output,
                               threads=args.tokenizer_threads, compress=args.compress)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
