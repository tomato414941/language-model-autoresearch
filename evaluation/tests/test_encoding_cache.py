from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from oe_eval import run_eval
from oe_eval.components.requests import LoglikelihoodRequest

from evaluation.count_tokens import OLMES_REVISION, SUITES, TokenObserver, TokensObserved
from evaluation.encoding_cache import EncodingReader, EncodingWriter, input_identity, use_encodings
from evaluation.gold_bpb import gold_request_indices
from evaluation.prepare_encodings import prepare_cache
from evaluation.protocol import read_requests, score_task, sha256_file
from evaluation.tests.test_count_tokens import ForwardProbe
from evaluation.tests.test_protocol import synthetic_task
from evaluation.tokenizer_adapter import TokenizerAdapter
from gpu_config import active_gpu


class NonuniformProbe(ForwardProbe):
    def _model_call(self, inputs, **kwargs):
        self.shapes.append(tuple(inputs.shape))
        positions = torch.arange(self.vocab_size)
        return -((positions - (inputs[..., None] * 37 + 11)) % 97).float() * .01


class EncodingCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer_path = active_gpu().cache_directory("tokenizer") / "tokenizer.pkl"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "requests.json"
        self.manifest.write_text('{"input": "synthetic"}\n')
        self.tokenizer = TokenizerAdapter(self.tokenizer_path)
        self.observer = TokenObserver(self.tokenizer, max_length=64, batch_size=2)
        self.identity = input_identity(self.observer, self.tokenizer_path, self.manifest)
        self.path = self.root / "native.sqlite"

    def record(self, requests):
        writer = EncodingWriter(self.path, self.identity)
        try:
            with use_encodings(self.observer, writer):
                with self.assertRaises(TokensObserved):
                    self.observer.loglikelihood_verbose(requests, disable_tqdm=True)
            writer.finish()
        finally:
            writer.close()

    def test_replays_native_tokens_likelihoods_and_batches_without_tokenizing_again(self):
        requests = [LoglikelihoodRequest("", "An empty context."),
                    LoglikelihoodRequest("Trailing whitespace \n\t", "answer!"),
                    LoglikelihoodRequest("日本語🙂\n", "数学と Python: x = 2 ** 3\n"),
                    LoglikelihoodRequest("prefix " * 100, "final answer"),
                    LoglikelihoodRequest("repeat", "same"), LoglikelihoodRequest("repeat", "same")]
        self.record(requests)
        reference = NonuniformProbe(self.tokenizer, max_length=64, batch_size=2)
        expected = reference.loglikelihood_verbose(requests, disable_tqdm=True)
        probe = NonuniformProbe(self.tokenizer, max_length=64, batch_size=2)
        reader = EncodingReader(self.path, self.identity)
        self.addCleanup(reader.close)
        with use_encodings(probe, reader), \
                patch.object(self.tokenizer, "encode", side_effect=AssertionError("Use the prepared encodings.")):
            actual = probe.loglikelihood_verbose(requests, disable_tqdm=True)
        self.assertEqual(actual, expected)
        self.assertEqual(probe.shapes, reference.shapes)
        self.assertEqual(reader.hits, len(requests))

    def test_preserves_masked_unicode_tokens_special_tokens_and_left_truncation(self):
        cases = [("a日本語🙂é\n", {"masked_spans": [(0, 4)]}),
                 ("a日本語🙂é\n", {"left_truncate_len": 4}),
                 ("plain text", {"add_special_tokens": True}),
                 ("unpaired surrogate: \ud800", {})]
        writer = EncodingWriter(self.path, self.identity)
        with use_encodings(self.observer, writer):
            expected = [self.observer.tok_encode(text, **kwargs) for text, kwargs in cases]
        writer.finish()
        self.addCleanup(writer.close)
        reader = EncodingReader(self.path, self.identity)
        self.addCleanup(reader.close)
        with use_encodings(self.observer, reader):
            actual = [self.observer.tok_encode(text, **kwargs) for text, kwargs in cases]
        self.assertEqual(actual, expected)
        self.assertTrue(any(token < 0 for token in expected[0]))
        self.assertEqual(len(expected[1]), 4)
        self.assertEqual(expected[2][0], self.tokenizer.bos_token_id)

    def test_checks_input_identity_and_actual_model_encoding_settings(self):
        self.record([LoglikelihoodRequest("question", "answer")])
        with self.assertRaisesRegex(ValueError, "matching these inputs"):
            EncodingReader(self.path, dict(self.identity, request_manifest_sha256="another input"))
        reader = EncodingReader(self.path, self.identity)
        self.addCleanup(reader.close)
        self.observer.add_bos_token = True
        with self.assertRaisesRegex(ValueError, "same native encoder settings"):
            with use_encodings(self.observer, reader):
                self.observer.tok_encode("question")

    def test_restores_native_encoding_after_a_cache_miss(self):
        self.record([LoglikelihoodRequest("question", "answer")])
        expected = self.observer.tok_encode("a new request")
        reader = EncodingReader(self.path, self.identity)
        self.addCleanup(reader.close)
        with self.assertRaisesRegex(ValueError, "does not contain this request"):
            with use_encodings(self.observer, reader):
                self.observer.tok_encode("a new request")
        self.assertEqual(self.observer.tok_encode("a new request"), expected)

    def test_retains_completed_artifacts_and_reports_incomplete_preparation(self):
        writer = EncodingWriter(self.path, self.identity)
        writer.put("single", ["example"], [[1, -100, 2]])
        writer.close()
        with self.assertRaisesRegex(ValueError, "complete native encoding cache"):
            EncodingReader(self.path, self.identity)
        original = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            EncodingWriter(self.path, self.identity)
        self.assertEqual(self.path.read_bytes(), original)

    def test_replays_signed_and_wide_token_ids_as_independent_vectors(self):
        identity = deepcopy(self.identity)
        identity["model"]["vocab_size"] = 70001
        writer = EncodingWriter(self.path, identity)
        vectors = [[65536, 65535, 0], [-100, 70000, 2]]
        writer.put("pair", ["context", "continuation"], vectors)
        writer.finish()
        self.addCleanup(writer.close)
        reader = EncodingReader(self.path, identity)
        self.addCleanup(reader.close)
        actual = reader.get("pair", ["context", "continuation"])
        self.assertEqual(actual, vectors)
        actual[0][0] = 9
        self.assertEqual(reader.get("pair", ["context", "continuation"]), vectors)

    def test_prepares_and_replays_every_gold_request_from_original_fields(self):
        task_config = {"task_name": "synthetic_cache", "metadata": {"alias": "synthetic_cache"}}
        task = synthetic_task(task_config)
        task.download()
        task.build_all_requests()
        requests = self.root / "requests"
        requests.mkdir()
        payload = b"".join((json.dumps(instance.to_dict()) + "\n").encode() for instance in task._instances)
        (requests / "requests.jsonl.gz").write_bytes(gzip.compress(payload))
        entry = dict(alias="synthetic_cache", index=0, native_config=task_config, task_hash=task._task_hash,
                     requests=len(task._instances), examples=2,
                     resolved_task_config=task.task_config, requests_file="requests.jsonl.gz",
                     requests_sha256=sha256_file(requests / "requests.jsonl.gz"))
        entry["uncompressed_requests_sha256"] = hashlib.sha256(payload).hexdigest()
        manifest = dict(complete=True, olmes_revision=OLMES_REVISION, suites=list(SUITES),
                        max_length=64, batch_size=2, tasks=[entry], total_examples=2)
        (requests / "manifest.json").write_text(json.dumps(manifest))
        report = prepare_cache(requests, self.tokenizer_path, self.path)
        self.assertTrue(report["complete"])
        self.assertEqual(report["examples"], 2)
        self.assertEqual(report["recorded_calls"], 2)
        self.assertEqual(report["quality_evaluations"], 0)
        _, instances = read_requests(requests, entry)
        selected = gold_request_indices(instances, entry["resolved_task_config"])
        observer = TokenObserver(self.tokenizer, max_length=64, batch_size=2)
        reader = EncodingReader(self.path, input_identity(observer, self.tokenizer_path, requests / "manifest.json"))
        self.addCleanup(reader.close)
        with use_encodings(observer, reader):
            with self.assertRaises(TokensObserved):
                observer.loglikelihood_verbose([instances[index].request for index in selected], disable_tqdm=True)
        self.assertEqual(reader.hits, 2)
        config = run_eval.process_eval_args(vars(run_eval._parser.parse_args([
            "--model", "autoresearch", "--task", json.dumps(task_config), "--batch-size", "2"])))
        metrics = []
        for name, cached in (("native", False), ("cached", True)):
            output = self.root / name
            output.mkdir()
            compute = dict(config["compute_config"], output_dir=str(output))
            probe = NonuniformProbe(self.tokenizer, max_length=64, batch_size=2)
            with patch.object(run_eval, "load_task", side_effect=synthetic_task):
                if cached:
                    with use_encodings(probe, reader):
                        result = score_task(probe, entry, *read_requests(requests, entry),
                                            config["model_config"], compute)
                else:
                    result = score_task(probe, entry, *read_requests(requests, entry),
                                        config["model_config"], compute)
            metrics.append(result)
        self.assertEqual(metrics[0]["metrics"], metrics[1]["metrics"])
        self.assertEqual(metrics[0]["task_config"], metrics[1]["task_config"])
        self.assertEqual(metrics[0]["num_instances"], metrics[1]["num_instances"])


if __name__ == "__main__":
    unittest.main()
