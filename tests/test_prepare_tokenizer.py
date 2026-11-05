from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import prepare
from tools.prepare_tokenizer import build_encoding, token_byte_lengths, training_texts


class PrepareTokenizerTests(unittest.TestCase):
    def test_reads_the_native_training_prefix_in_order_with_its_document_cap(self):
        texts = ["aaaaaa", "", "bb", "ccc", "ignored"]
        with tempfile.TemporaryDirectory() as directory:
            pq.write_table(pa.table({"text": texts}), Path(directory) / "shard_00000.parquet")
            pq.write_table(pa.table({"text": ["validation"]}),
                           Path(directory) / prepare.VAL_FILENAME)
            statistics = {}
            with patch.object(prepare, "DATA_DIR", directory):
                result = list(training_texts(7, 4, statistics))
        self.assertEqual(result, ["aaaa", "", "bb", "ccc"])
        self.assertEqual(statistics["documents"], 4)
        self.assertEqual(statistics["characters"], 9)
        self.assertEqual(statistics["utf8_bytes"], 9)
        self.assertEqual(len(statistics["ordered_document_sha256"]), 64)

    def test_trains_the_requested_vocabulary_and_roundtrips_unicode_and_special_tokens(self):
        text = "Alpha beta gamma delta. Numbers: 12345. 你好世界 🌍! I'm testing.\n"
        encoding = build_encoding(iter([text] * 8), 288)
        self.assertEqual(encoding.n_vocab, 288)
        self.assertEqual(encoding.decode(encoding.encode_ordinary(text)), text)
        special_ids = [encoding.encode_single_token(name) for name in prepare.SPECIAL_TOKENS]
        self.assertEqual(special_ids, list(range(288 - len(special_ids), 288)))
        sizes = token_byte_lengths(encoding)
        for index in special_ids:
            self.assertEqual(sizes[index].item(), 0)
        ordinary_sizes = sizes[:288 - len(special_ids)]
        self.assertTrue((ordinary_sizes > 0).all().item())
        expected = [len(encoding.decode([index]).encode("utf-8"))
                    for index in range(288 - len(special_ids))]
        self.assertEqual(ordinary_sizes.tolist(), expected)

    def test_requires_positive_prefix_limits_and_a_complete_byte_vocabulary(self):
        for budget, cap in ((0, 4), (7, 0), (-1, 4)):
            with self.subTest(budget=budget, cap=cap):
                with self.assertRaises(ValueError):
                    list(training_texts(budget, cap, {}))
        with self.assertRaises(ValueError):
            build_encoding(iter(["hello"]), 255 + len(prepare.SPECIAL_TOKENS))


if __name__ == "__main__":
    unittest.main()
