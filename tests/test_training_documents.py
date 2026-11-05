from contextlib import redirect_stdout
import gc
import gzip
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import prepare
from tools import prepare_training_documents as documents


class ByteTokenizer:
    def __init__(self, bos=256):
        self.bos = bos

    def get_vocab_size(self):
        return self.bos + 1

    def get_bos_token_id(self):
        return self.bos

    def encode(self, texts, prepend=None, num_threads=1):
        return [([] if prepend is None else [prepend]) + list(text.encode("utf-8")) for text in texts]

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8")


def records():
    texts = ("", "duplicate", "duplicate", "数学 🌍\nx=1\n", "<|reserved_0|>", "x"*257)
    return [dict(text=texts[index % len(texts)], ordinal=index,
                 metadata=dict(detected_licenses=["MIT", "BSD-3-Clause"], nullable=None, score=float("nan")))
            for index in range(259)]


class TrainingDocumentsTests(unittest.TestCase):
    def test_keeps_every_original_document_token_and_metadata_field_in_order(self):
        for bos in (256, 70000):
            for batch_size in (1, 3, 128):
                with self.subTest(bos=bos, batch_size=batch_size), tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "cache"
                    rows = records()
                    tokenizer = ByteTokenizer(bos)
                    report = documents.build(iter(rows), tokenizer, path, "tokenizer", dict(name="fixture"),
                                             batch_size=batch_size)
                    validation = documents.verify(iter(rows), tokenizer, path, "tokenizer", batch_size=17)
                    cache = documents.Documents(path, "tokenizer", verify_files=True)
                    self.assertEqual(report["documents"], len(rows))
                    self.assertEqual(validation["documents"], len(rows))
                    self.assertEqual(len(cache), len(rows))
                    self.assertEqual(report["empty_documents"], 44)
                    self.assertEqual(cache[0].tolist(), [bos])
                    for index, row in enumerate(rows):
                        self.assertEqual(cache[index].tolist(), [bos, *row["text"].encode("utf-8")])
                    with gzip.open(path / "metadata.jsonl.gz", "rt") as stream:
                        metadata = [json.loads(line) for line in stream]
                    self.assertEqual([row["ordinal"] for row in metadata], list(range(len(rows))))
                    self.assertEqual(metadata[1]["metadata"]["detected_licenses"], ["MIT", "BSD-3-Clause"])
                    self.assertIsNone(metadata[1]["metadata"]["nullable"])
                    self.assertTrue(math.isnan(metadata[1]["metadata"]["score"]))
                    self.assertEqual(cache[1].tolist(), cache[2].tolist())

    def test_returns_readonly_document_views_that_remain_valid_after_reader_is_released(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache"
            documents.build(iter([dict(text="hello")]), ByteTokenizer(), path, "tokenizer", {})
            reader = documents.Documents(path, "tokenizer")
            row = reader[0]
            del reader
            gc.collect()
            self.assertEqual(row.tolist(), [256, *b"hello"])
            with self.assertRaises(ValueError):
                row[0] = 0

    def test_preserves_unsigned_token_ids_at_storage_boundaries(self):
        for bos in (65535, 70000, 2**32 - 1):
            with self.subTest(bos=bos), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "cache"
                documents.build(iter([dict(text="a")]), ByteTokenizer(bos), path, "tokenizer", {})
                self.assertEqual(documents.Documents(path, "tokenizer")[0].tolist(), [bos, 97])
                self.assertTrue(documents.verify(iter([dict(text="a")]), ByteTokenizer(bos), path,
                                                 "tokenizer")["every_original_document_and_token_matches"])

    def test_preserves_an_existing_cache_when_preparation_is_requested_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache"
            documents.build(iter([dict(text="first")]), ByteTokenizer(), path, "tokenizer", {})
            original = (path / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                documents.build(iter([dict(text="second")]), ByteTokenizer(), path, "tokenizer", {})
            self.assertEqual((path / "manifest.json").read_bytes(), original)
            self.assertEqual(documents.Documents(path, "tokenizer")[0].tolist(), [256, *b"first"])

    def test_rejects_a_cache_bound_to_another_tokenizer_or_changed_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache"
            documents.build(iter([dict(text="a")]), ByteTokenizer(), path, "tokenizer", {})
            with self.assertRaises(ValueError):
                documents.Documents(path, "different-tokenizer", verify_files=True)
            raw = bytearray((path / "tokens.bin").read_bytes())
            raw[-1] ^= 1
            (path / "tokens.bin").write_bytes(raw)
            with self.assertRaises(ValueError):
                documents.Documents(path, "tokenizer", verify_files=True)

    def test_rejects_nonincreasing_document_offsets(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache"
            documents.build(iter([dict(text="a"), dict(text="b")]), ByteTokenizer(), path, "tokenizer", {})
            offsets = np.fromfile(path / "offsets.bin", dtype="<u8")
            offsets[1] = 0
            offsets.tofile(path / "offsets.bin")
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["files"]["offsets.bin"]["sha256"] = documents.digest(path / "offsets.bin")
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                documents.Documents(path, "tokenizer", verify_files=True)

    def test_verifies_license_metadata_and_order_for_repeated_texts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache"
            rows = [dict(text="same", license="MIT", ordinal=1),
                    dict(text="same", license="BSD-3-Clause", ordinal=2)]
            documents.build(iter(rows), ByteTokenizer(), path, "tokenizer", {})
            with self.assertRaises(ValueError):
                documents.verify(iter(reversed(rows)), ByteTokenizer(), path, "tokenizer")

    def test_compresses_token_and_offset_files_with_an_exact_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache"
            documents.build(iter(records()), ByteTokenizer(), path, "tokenizer", {})
            compressed = documents.compress_payload(path)
            for name in ("tokens.bin", "offsets.bin"):
                with gzip.open(path / (name + ".gz"), "rb") as stream:
                    self.assertEqual(stream.read(), (path / name).read_bytes())
                self.assertTrue(compressed["files"][name + ".gz"]["roundtrip_verified"])
            self.assertEqual(len(documents.Documents(path, "tokenizer")), 259)

    def make_source(self, root, name, rows_by_file):
        if name.startswith("finemath"):
            directory = root / "math" / name
            repository, revision = prepare.FINEMATH_REPOSITORY, prepare.FINEMATH_REVISION
            subset = name
        else:
            directory = root / "code"
            repository, revision = prepare.COMMON_PILE_REPOSITORY, prepare.COMMON_PILE_REVISION
            subset = ""
        directory.mkdir(parents=True)
        paths = []
        for index, rows in enumerate(rows_by_file):
            if name.startswith("finemath"):
                path = directory / f"train-{index:05d}-of-{len(rows_by_file):05d}.parquet"
                pq.write_table(pa.Table.from_pylist(rows), path)
            else:
                path = directory / f"stack-edu-{index:04d}.json.gz"
                with gzip.open(path, "wt") as stream:
                    for row in rows:
                        stream.write(json.dumps(row) + "\n")
            paths.append(path)
        manifest = dict(repository=repository, revision=revision, subset=subset, license="fixture",
                        files=[dict(path=f"{subset}/{path.name}" if subset else path.name,
                                    bytes=path.stat().st_size, sha256=documents.digest(path))
                               for path in paths])
        (directory / "manifest.json").write_text(json.dumps(manifest))
        return paths

    def test_prepares_complete_pinned_parquet_and_gzip_prefixes_with_source_boundaries(self):
        rows = [[dict(text="", license="MIT"), dict(text="same", license="BSD-3-Clause")],
                [dict(text="same", license="MIT"), dict(text="数学 🌍", license="Apache-2.0")]]
        for name in ("finemath-4plus", "common-pile"):
            with self.subTest(source=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.make_source(root, name, rows)
                tokenizer = root / "tokenizer"
                tokenizer.mkdir()
                (tokenizer / "tokenizer.pkl").write_bytes(b"fixture")
                destination = root / "prepared"
                with patch.object(prepare, "FINEMATH_DIR", str(root / "math")), \
                        patch.object(prepare, "COMMON_PILE_DIR", str(root / "code")), \
                        patch.object(prepare.Tokenizer, "from_directory", return_value=ByteTokenizer()), \
                        redirect_stdout(io.StringIO()):
                    result = documents.prepare_documents(name, 2, tokenizer, destination, compress=True)
                self.assertEqual(result["validation"]["documents"], 4)
                self.assertEqual([entry["documents"] for entry in result["source"]["files"]], [2, 2])
                manifest = json.loads((destination / "manifest.json").read_text())
                self.assertTrue(manifest["verified_against_original"])
                cache = documents.Documents(destination, documents.digest(tokenizer / "tokenizer.pkl"), True)
                self.assertEqual([ByteTokenizer().decode(cache[index][1:].tolist()) for index in range(4)],
                                 [row["text"] for batch in rows for row in batch])
                with gzip.open(destination / "metadata.jsonl.gz", "rt") as stream:
                    self.assertEqual([json.loads(line)["license"] for line in stream],
                                     [row["license"] for batch in rows for row in batch])

    def test_requires_the_published_first_shard_even_when_a_later_one_is_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_source(root, "finemath-4plus", [[dict(text="first")], [dict(text="second")]])
            paths[0].unlink()
            with patch.object(prepare, "FINEMATH_DIR", str(root / "math")):
                with self.assertRaises(ValueError):
                    documents.source_files("finemath-4plus", 1)

    def test_requires_source_bytes_to_match_the_pinned_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_source(root, "common-pile", [[dict(text="first")]])
            with paths[0].open("ab") as stream:
                stream.write(b"changed")
            with patch.object(prepare, "COMMON_PILE_DIR", str(root / "code")):
                with self.assertRaises(ValueError):
                    documents.source_files("common-pile", 1)

    def test_marks_preparation_incomplete_when_a_source_changes_during_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_source(root, "finemath-4plus", [[dict(text="first")]])
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            (tokenizer / "tokenizer.pkl").write_bytes(b"fixture")
            original_verify = documents.verify

            def change_source(*args, **kwargs):
                result = original_verify(*args, **kwargs)
                with paths[0].open("ab") as stream:
                    stream.write(b"changed")
                return result

            destination = root / "prepared"
            with patch.object(prepare, "FINEMATH_DIR", str(root / "math")), \
                    patch.object(prepare.Tokenizer, "from_directory", return_value=ByteTokenizer()), \
                    patch.object(documents, "verify", side_effect=change_source), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    documents.prepare_documents("finemath-4plus", 1, tokenizer, destination)
            self.assertFalse(json.loads((destination / "manifest.json").read_text())["complete"])
            with self.assertRaises(ValueError):
                documents.Documents(destination, documents.digest(tokenizer / "tokenizer.pkl"))


if __name__ == "__main__":
    unittest.main()
