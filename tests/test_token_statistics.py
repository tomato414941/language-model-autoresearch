import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import prepare
from tools.token_statistics import count_training_tokens


class ToyTokenizer:
    def __init__(self):
        self.documents = []

    def get_vocab_size(self):
        return 5

    def get_bos_token_id(self):
        return 0

    def encode(self, texts, prepend):
        self.documents.extend(texts)
        tokens = {"a": 1, "b": 2, "c": 3, "v": 4}
        return [[prepend] + [tokens[letter] for letter in text] for text in texts]


class TokenStatisticsTests(unittest.TestCase):
    def test_counts_training_documents_in_order_including_empty_and_repeated_text(self):
        texts = ["a", "a", ""]
        with tempfile.TemporaryDirectory() as directory:
            pq.write_table(pa.table({"text": texts}), Path(directory) / "shard_00000.parquet")
            pq.write_table(pa.table({"text": ["vvv"]}), Path(directory) / prepare.VAL_FILENAME)
            tokenizer = ToyTokenizer()
            with patch.object(prepare, "DATA_DIR", directory):
                counts, metadata = count_training_tokens(tokenizer, 3)
        np.testing.assert_array_equal(counts, [3, 2, 0, 0, 0])
        self.assertEqual(tokenizer.documents, texts)
        self.assertEqual((metadata["documents"], metadata["tokens"]), (3, 5))
        expected_hash = hashlib.sha256()
        for text in texts:
            raw = text.encode("utf-8")
            expected_hash.update(len(raw).to_bytes(8, "little"))
            expected_hash.update(raw)
        self.assertEqual(metadata["ordered_document_sha256"], expected_hash.hexdigest())

    def test_respects_the_document_prefix_and_emits_cumulative_checkpoints(self):
        tokenizer = ToyTokenizer()
        snapshots = []

        def observe(counts, metadata):
            snapshots.append((counts.copy(), metadata))

        with patch.object(prepare, "_document_batches", return_value=iter([(["a", "b", "c"], 7)])):
            counts, metadata = count_training_tokens(tokenizer, 2, (1, 2), observe)
        self.assertEqual(tokenizer.documents, ["a", "b"])
        np.testing.assert_array_equal(counts, [2, 1, 1, 0, 0])
        self.assertEqual([item[1]["documents"] for item in snapshots], [1, 2])
        np.testing.assert_array_equal(snapshots[0][0], [1, 1, 0, 0, 0])
        np.testing.assert_array_equal(snapshots[1][0], counts)
        self.assertEqual((metadata["first_epoch"], metadata["last_epoch"]), (7, 7))
        self.assertEqual((metadata["characters"], metadata["utf8_bytes"]), (2, 2))

    def test_requires_token_ids_within_the_tokenizer_vocabulary(self):
        for invalid_id in (-1, 5):
            with self.subTest(invalid_id=invalid_id):
                tokenizer = ToyTokenizer()
                tokenizer.encode = lambda texts, prepend: [[prepend, invalid_id] for _ in texts]
                with patch.object(prepare, "_document_batches", return_value=iter([(["a"], 1)])):
                    with self.assertRaises(ValueError):
                        count_training_tokens(tokenizer, 1)


if __name__ == "__main__":
    unittest.main()

