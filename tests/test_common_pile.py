import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import prepare
from tests.test_finemath import Response


class CommonPileDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name) / "common_pile" / "stackv2_edu_filtered"
        self.directory_patch = patch.object(prepare, "COMMON_PILE_DIR", str(self.directory))
        self.directory_patch.start()

    def tearDown(self):
        self.directory_patch.stop()
        self.temporary.cleanup()

    def entry(self, index, data):
        return {"type": "file", "path": f"stack-edu-{index:04d}.json.gz", "size": len(data),
                "lfs": {"oid": hashlib.sha256(data).hexdigest()}}

    def test_downloads_original_code_bytes_and_file_licenses_in_source_order(self):
        rows = [
            {"text": "print('first')", "metadata": {"license": "MIT"}},
            {"text": "print('second')", "metadata": {"license": "Apache-2.0"}},
        ]
        first = gzip.compress("\n".join(json.dumps(row) for row in rows).encode())
        second = gzip.compress(b'{"text":"other code"}\n')
        entries = [self.entry(1, second), {"type": "file", "path": "README.md", "size": 20},
                   self.entry(0, first)]
        calls = []
        def get(url, **kwargs):
            calls.append(url)
            self.assertIn(prepare.COMMON_PILE_REVISION, url)
            return Response(entries=entries) if "/api/datasets/" in url else Response(data=first)
        with patch.object(prepare.requests, "get", side_effect=get):
            paths = prepare.download_common_pile(1, 1)
        self.assertEqual([path.name for path in paths], ["stack-edu-0000.json.gz"])
        self.assertEqual(paths[0].read_bytes(), first)
        decoded = [json.loads(line) for line in gzip.decompress(paths[0].read_bytes()).splitlines()]
        self.assertEqual(decoded, rows)
        metadata = json.loads((self.directory / "manifest.json").read_text())
        self.assertEqual(metadata["repository"], prepare.COMMON_PILE_REPOSITORY)
        self.assertEqual(metadata["revision"], prepare.COMMON_PILE_REVISION)
        self.assertEqual(len(metadata["files"]), 2)
        self.assertEqual(len(calls), 2)

    def test_reuses_the_verified_code_shards_and_pinned_inventory(self):
        data = gzip.compress(b'{"text":"original code","metadata":{"license":"BSD-3-Clause"}}\n')
        get = Mock(side_effect=[Response(entries=[self.entry(0, data)]), Response(data=data)])
        with patch.object(prepare.requests, "get", get):
            paths = prepare.download_common_pile(1, 1)
            repeated = prepare.download_common_pile(1, 1)
        self.assertEqual(repeated, paths)
        self.assertEqual(repeated[0].read_bytes(), data)
        self.assertEqual(get.call_count, 2)

    def test_prepares_both_allowed_extra_sources_through_the_cli(self):
        observed = []
        with patch.object(prepare, "download_finemath", side_effect=lambda *args: observed.append(("math", *args))):
            with patch.object(prepare, "download_common_pile", side_effect=lambda *args: observed.append(("code", *args))):
                prepare.main(["--finemath", "finemath-4plus", "--common-pile",
                              "--common-pile-shards", "2", "--download-workers", "3"])
        self.assertEqual(observed, [("math", "finemath-4plus", 1, 3), ("code", 2, 3)])


if __name__ == "__main__":
    unittest.main()
