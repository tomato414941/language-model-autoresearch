from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from evaluation.count_tokens import OLMES_REVISION, SUITES, TokenObserver
from evaluation.encoding_cache import EncodingReader, EncodingWriter, input_identity
from evaluation.tokenizer_adapter import TokenizerAdapter
from gpu_config import active_gpu
from tools.package_encoding_cache import digest, package


class EncodingAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer_path = active_gpu().cache_directory("tokenizer") / "tokenizer.pkl"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "runtime.json"
        self.assets = self.root / "gpu-assets.tar.gz"
        self.cache = self.root / "cache.sqlite"
        self.output = self.root / "future variant"
        self.prefix = "autoresearch/gpus/rtx5090"
        self.manifest = dict(complete=True, olmes_revision=OLMES_REVISION, suites=list(SUITES),
                             max_length=64, batch_size=2, tasks=[{"alias": "synthetic"}],
                             total_examples=1, total_requests=3)
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.payloads = {
            f"{self.prefix}/olmes/manifest.json": self.manifest_path.read_bytes(),
            f"{self.prefix}/olmes/requests.jsonl.gz": gzip.compress(b'{"question":"unchanged choices"}\n'),
            f"{self.prefix}/tokenizer/tokenizer.pkl": self.tokenizer_path.read_bytes(),
            f"{self.prefix}/tokenizer/token_bytes.pt": b"unchanged reference byte table",
            f"{self.prefix}/olmes/" + "long-" * 30 + "requests.jsonl.gz": gzip.compress(b"long path payload"),
        }
        with tarfile.open(self.assets, "w:gz", format=tarfile.GNU_FORMAT) as archive:
            for index, (name, payload) in enumerate(self.payloads.items()):
                entry = tarfile.TarInfo(name)
                entry.size, entry.mode, entry.mtime = len(payload), 0o640, 1000 + index
                archive.addfile(entry, io.BytesIO(payload))
            symlink = tarfile.TarInfo(f"{self.prefix}/olmes/request-link")
            symlink.type, symlink.linkname = tarfile.SYMTYPE, "requests.jsonl.gz"
            archive.addfile(symlink)
            hardlink = tarfile.TarInfo(f"{self.prefix}/tokenizer/table-link.pt")
            hardlink.type, hardlink.linkname = tarfile.LNKTYPE, f"{self.prefix}/tokenizer/token_bytes.pt"
            archive.addfile(hardlink)
        (self.root / "gpu-env.tar.gz").write_bytes(b"fixed environment")
        self.base = dict(
            gpu_id="rtx5090", evaluation_metric="olmes_bpb", gpu_env_sha256=digest(self.root / "gpu-env.tar.gz"),
            gpu_assets_sha256=digest(self.assets), tokenizer={
                "tokenizer.pkl": digest(self.tokenizer_path),
                "token_bytes.pt": hashlib.sha256(b"unchanged reference byte table").hexdigest()},
            olmes=dict(revision=OLMES_REVISION, leaf_tasks=1, examples=1, requests=3,
                       request_manifest_sha256=digest(self.manifest_path)))
        self.runtime.write_text(json.dumps(self.base))
        observer = TokenObserver(TokenizerAdapter(self.tokenizer_path), max_length=64, batch_size=2)
        self.identity = input_identity(observer, self.tokenizer_path, self.manifest_path)
        self.vectors = observer._encode_pair("question:", "answer")
        writer = EncodingWriter(self.cache, self.identity)
        writer.put("pair", ["question:", "answer"], self.vectors)
        writer.finish()
        writer.close()

    def test_packages_replayable_encodings_and_preserves_all_original_payloads_modes_and_links(self):
        original = {path.name: path.read_bytes() for path in (self.runtime, self.assets, self.cache)}
        report = package(self.runtime, self.cache, self.output, gpu="rtx5090")
        self.assertTrue(report["original_members_and_payloads_unchanged"])
        self.assertEqual(report["quality_evaluations"], 0)
        with tarfile.open(self.assets) as source, tarfile.open(self.output / "gpu-assets.tar.gz") as target:
            for entry in source.getmembers():
                actual = target.getmember(entry.name)
                self.assertEqual((actual.mode, actual.mtime, actual.type, actual.linkname),
                                 (entry.mode, entry.mtime, entry.type, entry.linkname))
                if entry.isfile():
                    self.assertEqual(target.extractfile(actual).read(), source.extractfile(entry).read())
            restored = self.root / "restored.sqlite"
            restored.write_bytes(target.extractfile(report["encoding_cache"]["asset_path"]).read())
        reader = EncodingReader(restored, self.identity)
        self.addCleanup(reader.close)
        self.assertEqual(tuple(reader.get("pair", ["question:", "answer"])), self.vectors)
        for path in (self.runtime, self.assets, self.cache):
            self.assertEqual(path.read_bytes(), original[path.name])
        self.assertEqual((self.output / "gpu-env.tar.gz").stat().st_ino,
                         (self.root / "gpu-env.tar.gz").stat().st_ino)
        packaged = json.loads((self.output / "runtime.json").read_text())
        self.assertEqual(packaged["olmes"], self.base["olmes"])
        self.assertEqual(packaged["tokenizer"], self.base["tokenizer"])

    def test_preserves_completed_variants_when_the_same_destination_is_reused(self):
        package(self.runtime, self.cache, self.output, gpu="rtx5090")
        before = (self.output / "runtime.json").read_bytes()
        with self.assertRaises(FileExistsError):
            package(self.runtime, self.cache, self.output, gpu="rtx5090")
        self.assertEqual((self.output / "runtime.json").read_bytes(), before)

    def test_keeps_the_original_assets_when_encodings_belong_to_other_requests(self):
        other = self.root / "other.sqlite"
        identity = deepcopy(self.identity)
        identity["request_manifest_sha256"] = "other requests"
        writer = EncodingWriter(other, identity)
        writer.put("pair", ["question:", "answer"], self.vectors)
        writer.finish()
        original = self.assets.read_bytes()
        with self.assertRaisesRegex(ValueError, "matching these inputs"):
            package(self.runtime, other, self.output, gpu="rtx5090")
        self.assertEqual(self.assets.read_bytes(), original)

    def test_keeps_the_original_assets_when_the_requested_gpu_differs(self):
        original = self.assets.read_bytes()
        with self.assertRaisesRegex(ValueError, "selected GPU"):
            package(self.runtime, self.cache, self.output, gpu="h100")
        self.assertEqual(self.assets.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
