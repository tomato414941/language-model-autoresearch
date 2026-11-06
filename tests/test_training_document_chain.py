from copy import deepcopy
import gc
import tempfile
from pathlib import Path
import unittest

from tools.prepare_training_documents import build, ChainedDocuments
from tests.test_training_documents import ByteTokenizer, records


class TrainingDocumentChainTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rows = records()
        self.tokenizer = ByteTokenizer()
        self.source = dict(name="fixture", repository="fixture", revision="fixed", license="fixture", files=[
            dict(filename="train-00000.parquet", bytes=1, sha256="a" * 64),
            dict(filename="train-00001.parquet", bytes=2, sha256="b" * 64)])
        self.directories = [self.root / "first", self.root / "second"]
        for index, rows in enumerate((self.rows[:131], self.rows[131:])):
            origin = dict(self.source, files=[self.source["files"][index]])
            build(iter(rows), self.tokenizer, self.directories[index], "tokenizer", origin)

    def chain(self, directories=None, source=None):
        return ChainedDocuments(self.directories if directories is None else directories,
                                "tokenizer", self.source if source is None else source, verify_files=True)

    def test_preserves_every_document_across_file_boundaries_without_combining_duplicates(self):
        chain = self.chain()
        self.assertEqual(len(chain), len(self.rows))
        for index, row in enumerate(self.rows):
            self.assertEqual(chain[index].tolist(), [256, *row["text"].encode("utf-8")])
        self.assertEqual(chain[1].tolist(), chain[2].tolist())
        self.assertEqual(chain[0].tolist(), [256])

    def test_batches_across_files_and_restarts_only_after_the_complete_source_epoch(self):
        batches = self.chain().batches(128)
        actual = [next(batches) for _ in range(4)]
        self.assertEqual([(len(rows), epoch) for rows, epoch in actual], [(128, 1), (128, 1), (3, 1), (128, 2)])
        for index, row in enumerate(actual[1][0]):
            expected = self.rows[128 + index]["text"].encode("utf-8")
            self.assertEqual(row.tolist(), [256, *expected])
        self.assertEqual(actual[3][0][0].tolist(), [256])

    def test_keeps_a_readonly_view_valid_after_the_chain_is_released(self):
        chain = self.chain()
        row = chain[131]
        del chain
        gc.collect()
        self.assertEqual(row.tolist(), [256, *self.rows[131]["text"].encode("utf-8")])
        with self.assertRaises(ValueError):
            row[0] = 0

    def test_checks_complete_declared_file_order_and_source_identity(self):
        for directories in (list(reversed(self.directories)), [self.directories[0], self.directories[0]]):
            with self.subTest(directories=directories), self.assertRaisesRegex(ValueError, "complete source prefix"):
                self.chain(directories)
        source = deepcopy(self.source)
        source["revision"] = "another revision"
        with self.assertRaisesRegex(ValueError, "same pinned training source"):
            self.chain(source=source)

    def test_rejects_indices_outside_the_complete_source(self):
        chain = self.chain()
        for index in (-1, len(chain)):
            with self.subTest(index=index), self.assertRaises(IndexError):
                chain[index]


if __name__ == "__main__":
    unittest.main()
