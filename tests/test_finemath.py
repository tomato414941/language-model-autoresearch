import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

import prepare


class Response:
    def __init__(self, *, entries=None, data=b"", next_url=None):
        self.entries = entries
        self.data = data
        self.links = {"next": {"url": next_url}} if next_url else {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def json(self):
        return self.entries

    def iter_content(self, chunk_size):
        yield self.data[:2]
        yield b""
        yield self.data[2:]


class FineMathDownloadTests(unittest.TestCase):
    subset = "finemath-4plus"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.extra = self.root / "finemath"
        self.patch = patch.object(prepare, "FINEMATH_DIR", str(self.extra))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def entry(self, index, data):
        return {
            "type": "file",
            "path": f"{self.subset}/train-{index:05d}-of-00064.parquet",
            "size": len(data),
            "lfs": {"oid": hashlib.sha256(data).hexdigest()},
        }

    def test_downloads_the_verified_raw_shard_prefix_in_sorted_source_order(self):
        first, second = b"complete first shard bytes", b"complete second shard bytes"
        calls = []
        entries = [self.entry(1, second), self.entry(0, first)]
        def get(url, **kwargs):
            calls.append(url)
            if "/api/datasets/" in url:
                return Response(entries=entries)
            self.assertIn("/resolve/" + prepare.FINEMATH_REVISION + "/", url)
            return Response(data=first)
        with patch.object(prepare.requests, "get", side_effect=get):
            paths = prepare.download_finemath(self.subset, 1, 1)
        self.assertEqual([path.name for path in paths], ["train-00000-of-00064.parquet"])
        self.assertEqual(paths[0].read_bytes(), first)
        metadata = json.loads((self.extra / self.subset / "manifest.json").read_text())
        self.assertEqual(metadata["revision"], prepare.FINEMATH_REVISION)
        self.assertEqual(metadata["license"], "ODC-By-1.0")
        self.assertEqual(len(metadata["files"]), 2)
        self.assertEqual(len(calls), 2)

    def test_reuses_a_verified_cached_shard_and_its_pinned_manifest(self):
        data = b"all original bytes"
        get = Mock(side_effect=[Response(entries=[self.entry(0, data)]), Response(data=data)])
        with patch.object(prepare.requests, "get", get):
            first = prepare.download_finemath(self.subset, 1, 1)
            repeated = prepare.download_finemath(self.subset, 1, 1)
        self.assertEqual(first, repeated)
        self.assertEqual(repeated[0].read_bytes(), data)
        self.assertEqual(get.call_count, 2)

    def test_retries_incomplete_transfer_and_publishes_the_verified_payload(self):
        data = b"complete original payload"
        get = Mock(side_effect=[Response(entries=[self.entry(0, data)]),
                                Response(data=b"short"), Response(data=data)])
        with patch.object(prepare.requests, "get", get):
            with patch.object(prepare.time, "sleep"):
                paths = prepare.download_finemath(self.subset, 1, 1)
        self.assertEqual(paths[0].read_bytes(), data)
        self.assertEqual(get.call_count, 3)

    def test_replaces_a_corrupt_cache_file_with_the_published_bytes(self):
        data = b"correct original payload"
        directory = self.extra / self.subset
        directory.mkdir(parents=True)
        target = directory / "train-00000-of-00064.parquet"
        target.write_bytes(b"x" * len(data))
        get = Mock(side_effect=[Response(entries=[self.entry(0, data)]), Response(data=data)])
        with patch.object(prepare.requests, "get", get):
            paths = prepare.download_finemath(self.subset, 1, 1)
        self.assertEqual(paths[0], target)
        self.assertEqual(target.read_bytes(), data)

    def test_checks_available_space_for_the_selected_shards_before_transfer(self):
        data = b"larger than the available disk"
        get = Mock(return_value=Response(entries=[self.entry(0, data)]))
        with patch.object(prepare.requests, "get", get):
            with patch.object(prepare.shutil, "disk_usage", return_value=types.SimpleNamespace(free=1)):
                with self.assertRaises(OSError):
                    prepare.download_finemath(self.subset, 1, 1)
        self.assertEqual(get.call_count, 1)

    def test_follows_manifest_pages_at_the_approved_revision(self):
        first, second = b"one", b"two"
        base = (f"https://huggingface.co/api/datasets/{prepare.FINEMATH_REPOSITORY}/tree/"
                f"{prepare.FINEMATH_REVISION}/{self.subset}")
        get = Mock(side_effect=[
            Response(entries=[self.entry(1, second)], next_url=base + "?cursor=next"),
            Response(entries=[self.entry(0, first)]),
            Response(data=first), Response(data=second),
        ])
        with patch.object(prepare.requests, "get", get):
            paths = prepare.download_finemath(self.subset, -1, 1)
        self.assertEqual([path.read_bytes() for path in paths], [first, second])
        self.assertEqual(get.call_count, 4)

    def test_requires_an_approved_subset_and_valid_download_parameters(self):
        for subset, count, workers in (
            ("infiwebmath-4plus", 1, 1), (self.subset, 0, 1),
            (self.subset, -2, 1), (self.subset, 1, 0),
        ):
            with self.subTest(subset=subset, count=count, workers=workers):
                with self.assertRaises(ValueError):
                    prepare.download_finemath(subset, count, workers)

    def test_checks_the_cached_manifest_source_before_reusing_it(self):
        directory = self.extra / self.subset
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({
            "repository": prepare.FINEMATH_REPOSITORY,
            "revision": "a different revision",
            "subset": self.subset,
            "files": [],
        }))
        with self.assertRaises(ValueError):
            prepare.download_finemath(self.subset, 1, 1)

    def test_keeps_default_fineweb_training_documents_after_extra_data_is_present(self):
        native = self.root / "data"
        native.mkdir()
        pq.write_table(pa.table({"text": ["default FineWeb document"]}), native / "shard_00000.parquet")
        pq.write_table(pa.table({"text": ["held-out document"]}), native / prepare.VAL_FILENAME)
        math = self.extra / self.subset
        math.mkdir(parents=True)
        pq.write_table(pa.table({"text": ["additional math document"]}), math / "train-00000-of-00064.parquet")
        with patch.object(prepare, "DATA_DIR", str(native)):
            rows, epoch = next(prepare._document_batches("train"))
            self.assertEqual(rows, ["default FineWeb document"])
            self.assertEqual(epoch, 1)
            self.assertEqual(list(prepare.text_iterator()), ["default FineWeb document"])

    def test_exposes_a_separate_finemath_preparation_command(self):
        observed = []
        with patch.object(prepare, "download_finemath", side_effect=lambda *args: observed.append(args)):
            prepare.main(["--finemath", self.subset, "--finemath-shards", "2", "--download-workers", "3"])
        self.assertEqual(observed, [(self.subset, 2, 3)])

    def test_prepares_default_fineweb_data_and_tokenizer_with_the_original_arguments(self):
        observed = []
        def download(num_shards, download_workers):
            observed.append(("fineweb", num_shards, download_workers))
        with patch.object(prepare, "download_data", side_effect=download):
            with patch.object(prepare, "train_tokenizer", side_effect=lambda: observed.append(("tokenizer",))):
                prepare.main(["--num-shards", "2", "--download-workers", "3"])
        self.assertEqual(observed, [("fineweb", 2, 3), ("tokenizer",)])


if __name__ == "__main__":
    unittest.main()
