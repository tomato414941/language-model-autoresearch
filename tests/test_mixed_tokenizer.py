import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import prepare_mixed_tokenizer as mixed


class MixedTokenizerPrefixTests(unittest.TestCase):
    def prefixes(self, rows, max_chars, doc_cap):
        statistics = []
        def sources(name, count):
            self.assertIsNone(count)
            return [Path(name)], dict(name=name, revision="fixture")
        with patch.object(mixed, "training_sources", side_effect=sources), \
                patch.object(mixed, "digest", side_effect=lambda path: str(path)), \
                patch.object(mixed, "text_rows", side_effect=lambda path: iter(rows[path.name])):
            output = list(mixed.mixed_prefix(max_chars, doc_cap, statistics))
        return output, statistics

    def test_uses_equal_original_prefixes_and_preserves_empty_repeated_and_unicode_documents(self):
        rows = {"fineweb": ["", "aa", "aa", "later"],
                "finemath-4plus": ["你好", "12", "later"],
                "common-pile": ["x=1", "\n", "later"]}
        actual, statistics = self.prefixes(rows, 12, 10)
        self.assertEqual(actual, ["", "aa", "aa", "你好", "12", "x=1", "\n"])
        self.assertEqual([case["requested_characters"] for case in statistics], [4, 4, 4])
        self.assertEqual([case["characters"] for case in statistics], [4, 4, 4])
        self.assertTrue(all(case["complete"] for case in statistics))

    def test_caps_each_document_and_keeps_the_complete_last_document_at_the_budget_boundary(self):
        rows = {name: ["abcdef", "again", "later"] for name in mixed.SOURCES}
        actual, statistics = self.prefixes(rows, 16, 4)
        self.assertEqual(actual, ["abcd", "agai"] * 3)
        self.assertEqual([case["requested_characters"] for case in statistics], [6, 5, 5])
        self.assertEqual([case["characters"] for case in statistics], [8, 8, 8])
        expected = hashlib.sha256()
        for text in ("abcd", "agai"):
            raw = text.encode()
            expected.update(len(raw).to_bytes(8, "little"))
            expected.update(raw)
        self.assertEqual(statistics[0]["ordered_document_sha256"], expected.hexdigest())

    def test_requires_the_full_predeclared_budget_of_every_source(self):
        rows = {name: ["x"] for name in mixed.SOURCES}
        with self.assertRaises(ValueError):
            self.prefixes(rows, 12, 10)


if __name__ == "__main__":
    unittest.main()
