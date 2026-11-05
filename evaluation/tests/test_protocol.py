from gpu_config import active_gpu
from copy import deepcopy
import json
import gzip
import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from oe_eval import launch, run_eval
from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
from oe_eval.tasks.base_task import MultipleChoiceTask
from oe_eval.components.instances import RequestInstance
from oe_eval.components.requests import GenerateUntilAndLoglikelihoodRequest
from oe_eval.utils import hash_dict

from evaluation.count_tokens import OLMES_REVISION, SUITES
from evaluation.protocol import aggregate_suites, evaluate_prepared, read_requests, score_task
from evaluation.tests.test_count_tokens import ForwardProbe
from evaluation.tokenizer_adapter import TokenizerAdapter


class SyntheticTask(MultipleChoiceTask):
    TASK_CONFIG_DEFAULTS = {
        "dataset_path": "synthetic", "split": "test", "num_shots": 0,
        "primary_metric": "bits_per_byte_corr", "compute_gold_bpb": True,
    }

    def download(self):
        self.docs = [
            {"id": "one", "question": "A sample question:", "choices": ["one", "two"], "gold": 0},
            {"id": "two", "question": "A second question:", "choices": ["yes", "no"], "gold": 1},
        ]

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.docs

    def doc_to_text(self, doc):
        return doc["question"]


def synthetic_task(config, output_dir=None):
    values = deepcopy(config)
    name = values.pop("task_name")
    task = SyntheticTask(task_name=name, task_config=values)
    task._task_hash = hash_dict(task.task_config, run_eval.TASK_DEFAULTS)
    task._output_dir = output_dir
    return task


class NativeProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer = TokenizerAdapter(active_gpu().cache_directory("tokenizer") / "tokenizer.pkl")

    def test_scores_prebuilt_requests_like_the_complete_native_runner(self):
        task_config = {"task_name": "synthetic_protocol", "metadata": {"alias": "synthetic_protocol"}}
        with tempfile.TemporaryDirectory() as directory:
            native_output = Path(directory) / "native"
            args = vars(run_eval._parser.parse_args([
                "--model", "autoresearch", "--task", json.dumps(task_config),
                "--batch-size", "2", "--output-dir", str(native_output),
            ]))
            config = run_eval.process_eval_args(deepcopy(args))
            native_model = ForwardProbe(self.tokenizer, max_length=64, batch_size=2)
            with patch.object(run_eval, "load_task", side_effect=synthetic_task), \
                    patch.object(run_eval, "load_model", return_value=native_model):
                run_eval.run_eval(args)
            native_metrics = [json.loads(line) for line in (native_output / "metrics-all.jsonl").read_text().splitlines()][0]
            task = synthetic_task(task_config)
            task.download()
            task.build_all_requests()
            raw = [instance.to_dict() for instance in task._instances]
            entry = {"index": 0, "native_config": task_config, "task_hash": task._task_hash,
                     "resolved_task_config": deepcopy(task.task_config), "requests": len(raw), "examples": 2}
            compute = deepcopy(config["compute_config"])
            compute["output_dir"] = str(Path(directory) / "prepared")
            Path(compute["output_dir"]).mkdir()
            with patch.object(run_eval, "load_task", side_effect=synthetic_task):
                actual = score_task(ForwardProbe(self.tokenizer, max_length=64, batch_size=2),
                                    entry, raw, task._instances, config["model_config"], compute)
            for name in ("bits_per_byte_corr", "primary_score"):
                self.assertEqual(actual["metrics"][name], native_metrics["metrics"][name])
            self.assertEqual(actual["computation"]["computed_requests"], 2)
            self.assertEqual(actual["num_instances"], native_metrics["num_instances"])
            self.assertEqual(actual["task_config"], native_metrics["task_config"])

    def test_preserves_native_nested_macros_across_all_three_suites(self):
        aliases = [alias for suite in SUITES for alias in launch.resolve_task_suite(suite, {})]
        math_aliases = set(launch.resolve_task_suite(SUITES[1], {}))
        code_aliases = set(launch.resolve_task_suite(SUITES[2], {}))
        metrics = []
        for index, alias in enumerate(aliases):
            value = 2.0 if alias in math_aliases else 4.0 if alias in code_aliases else 3.0 if alias.startswith("mmlu_") else 1.0
            metrics.append({
                "task_name": alias, "task_config": {"metadata": {"alias": alias}},
                "metrics": {"primary_score": value}, "num_instances": index + 1,
                "model_hash": "synthetic", "model_config": {}, "compute_config": {},
                "processing_time": 0.0, "current_date": "synthetic", "beaker_info": {},
            })
        suites, _ = aggregate_suites(metrics, aliases)
        self.assertEqual(set(suites), {"qa_bpb", "math_bpb", "code_bpb"})
        self.assertAlmostEqual(suites["qa_bpb"], 1.0 + 2.0 / len(TASK_SUITE_CONFIGS[SUITES[0]]["tasks"]))
        self.assertEqual(suites["math_bpb"], 2.0)
        self.assertEqual(suites["code_bpb"], 4.0)

    def test_scores_prepared_inputs_with_a_new_batch_size_and_records_the_actual_computation(self):
        task_config = {"task_name": "synthetic_batch", "metadata": {"alias": "synthetic_batch"}}
        task = synthetic_task(task_config)
        task.download()
        task.build_all_requests()
        raw = [instance.to_dict() for instance in task._instances]
        payload = b"".join((json.dumps(item)+"\n").encode() for item in raw)
        compressed = gzip.compress(payload)
        args = vars(run_eval._parser.parse_args([
            "--model", "autoresearch", "--task", json.dumps(task_config), "--batch-size", "8",
        ]))
        config = run_eval.process_eval_args(args)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests = root / "requests"
            requests.mkdir()
            (requests / "requests.jsonl.gz").write_bytes(compressed)
            entry = dict(index=0, alias="synthetic_batch", native_config=task_config,
                         task_hash=task._task_hash, resolved_task_config=deepcopy(task.task_config),
                         requests=len(raw), examples=2, requests_file="requests.jsonl.gz",
                         requests_sha256=hashlib.sha256(compressed).hexdigest(),
                         uncompressed_requests_sha256=hashlib.sha256(payload).hexdigest())
            manifest = dict(complete=True, olmes_revision=OLMES_REVISION, suites=list(SUITES),
                            max_length=64, batch_size=8, native_config=config, tasks=[entry],
                            total_examples=2, total_requests=len(raw))
            (requests / "manifest.json").write_text(json.dumps(manifest))
            expected = dict(qa_bpb=1., math_bpb=1., code_bpb=1.)
            with patch.object(run_eval, "load_task", side_effect=synthetic_task), \
                    patch("evaluation.protocol.aggregate_suites", return_value=(expected, [])):
                suites, returned = evaluate_prepared(
                    ForwardProbe(self.tokenizer, max_length=64, batch_size=32), requests, root / "output")
            recorded = json.loads((root / "output/metrics-all.jsonl").read_text().splitlines()[0])
        self.assertEqual(suites, expected)
        self.assertEqual(returned["batch_size"], 8)
        self.assertEqual(returned["computed_requests"], 2)
        self.assertEqual(recorded["compute_config"]["batch_size"], "32")
        self.assertEqual(recorded["computation"]["source_requests"], len(raw))
        self.assertEqual(recorded["task_config"]["metadata"]["alias"], "synthetic_batch")

    def test_restores_every_native_gold_bpb_request_field_and_score(self):
        original = RequestInstance(
            request_type="loglikelihood", doc={"choices": ["answer"], "gold": 0},
            request=GenerateUntilAndLoglikelihoodRequest(
                context="A question:", continuation=" answer", perplexity_context="original context",
                stop_sequences=["\n"], generation_kwargs={"max_gen_toks": 12}),
            idx=0, task_name="synthetic_gold_bpb", doc_id=0, native_id="fixture", label=0)
        raw = original.to_dict()
        payload = (json.dumps(raw) + "\n").encode()
        compressed = gzip.compress(payload)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "requests.jsonl.gz").write_bytes(compressed)
            entry = dict(requests_file="requests.jsonl.gz", requests=1,
                         requests_sha256=hashlib.sha256(compressed).hexdigest(),
                         uncompressed_requests_sha256=hashlib.sha256(payload).hexdigest())
            restored_raw, restored = read_requests(directory, entry)
        self.assertEqual(restored_raw, [raw])
        self.assertEqual(restored[0].to_dict(), raw)
        task_config = {"metric_kwargs": {}}
        native = run_eval.evaluate(ForwardProbe(self.tokenizer, max_length=64, batch_size=2),
                                   [original], task_config, {})
        actual = run_eval.evaluate(ForwardProbe(self.tokenizer, max_length=64, batch_size=2),
                                   restored, task_config, {})
        self.assertEqual(actual, native)


if __name__ == "__main__":
    unittest.main()
