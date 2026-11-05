from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tiktoken

import prepare
from tools.prepare_training_cache import TokenLookup, build_cache, training_sources


class TrainingCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoding = tiktoken.Encoding(
            name="training-cache-test", pat_str=prepare.SPLIT_PATTERN,
            mergeable_ranks={bytes([value]): value for value in range(256)},
            special_tokens={name: 256 + index for index, name in enumerate(prepare.SPECIAL_TOKENS)})

    def prepare_cache(self, directory, texts):
        return build_cache(texts, self.encoding, Path(directory) / "cache", "fixture tokenizer",
                           {"name": "synthetic training fixture"})

    def test_recovers_complete_native_tokens_for_repeated_empty_and_unicode_documents(self):
        texts = ["hello", "", "你好 🌍", "hello", "<|reserved_0|>"]
        with tempfile.TemporaryDirectory() as directory:
            report = self.prepare_cache(directory, texts)
            cache = TokenLookup(Path(directory) / "cache", "fixture tokenizer", verify_files=True)
            actual = cache.lookup_many(texts)
            bos = self.encoding.encode_single_token(prepare.BOS_TOKEN)
            for text, tokens in zip(texts, actual, strict=True):
                np.testing.assert_array_equal(tokens, [bos, *self.encoding.encode_ordinary(text)])
                self.assertFalse(tokens.flags.writeable)
            self.assertEqual(report["documents"], len(texts))
            self.assertEqual(report["unique_documents"], len(set(texts)))
            self.assertEqual(report["tokens"], sum(len(tokens) for tokens in actual))

    def test_returns_explicit_misses_while_preserving_query_order(self):
        with tempfile.TemporaryDirectory() as directory:
            self.prepare_cache(directory, ["a", "b"])
            cache = TokenLookup(Path(directory) / "cache", "fixture tokenizer")
            results = cache.lookup_many(["b", "missing", "a", "b"])
            self.assertIsNone(results[1])
            self.assertEqual(results[0].tolist(), results[3].tolist())
            self.assertNotEqual(results[0].tolist(), results[2].tolist())
            self.assertEqual(cache.lookup_many([]), [])

    def test_requires_the_matching_tokenizer_and_complete_immutable_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            self.prepare_cache(directory, ["a"])
            path = Path(directory) / "cache"
            with self.assertRaises(ValueError):
                TokenLookup(path, "different tokenizer")
            tokens = path / "tokens.bin"
            tokens.write_bytes(tokens.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                TokenLookup(path, "fixture tokenizer", verify_files=True)

    def test_preserves_every_document_and_token_across_encoding_batches(self):
        texts = [f"training document {index}" for index in range(259)]
        with tempfile.TemporaryDirectory() as directory:
            report = self.prepare_cache(directory, texts)
            cache = TokenLookup(Path(directory) / "cache", "fixture tokenizer")
            actual = cache.lookup_many(texts)
            self.assertEqual(report["documents"], 259)
            self.assertEqual(report["unique_documents"], 259)
            bos = self.encoding.encode_single_token(prepare.BOS_TOKEN)
            for text, tokens in zip(texts, actual, strict=True):
                self.assertEqual(tokens.tolist(), [bos, *self.encoding.encode_ordinary(text)])

    def test_selects_a_training_prefix_and_excludes_the_pinned_validation_shard(self):
        names = ["shard_00000.parquet", "shard_00001.parquet", prepare.VAL_FILENAME]
        with patch.object(prepare, "list_parquet_files", return_value=names):
            paths, source = training_sources("fineweb", 2)
            self.assertEqual([path.name for path in paths], names[:2])
            self.assertEqual(source["revision"], prepare.BASE_URL.rsplit("/resolve/", 1)[1])
            with self.assertRaises(ValueError):
                training_sources("fineweb", 3)


if __name__ == "__main__":
    unittest.main()
