import contextlib
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

import prepare


class CoreEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = prepare.verify_core_assets()

    def test_accepts_only_a_higher_score(self):
        for candidate, incumbent, accepted in [
            (0.13, 0.12, True),
            (0.1201, 0.12, True),
            (0.14, 0.132, True),
            (0.12, 0.12, False),
            (0.1199, 0.12, False),
            (0.11, 0.12, False),
        ]:
            with self.subTest(candidate=candidate, incumbent=incumbent):
                self.assertEqual(prepare.core_improves(candidate, incumbent), accepted)

    def test_requires_finite_scores_for_an_adoption_decision(self):
        for score in (math.nan, math.inf, -math.inf):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    prepare.core_improves(score, 0.12)
                with self.assertRaises(ValueError):
                    prepare.core_improves(0.12, score)

    def test_passes_all_official_examples_to_all_22_upstream_tasks(self):
        evaluator = prepare._load_nanochat_core()
        calls = []

        def observe_task(model, tokenizer, data, device, task_meta):
            calls.append((task_meta, len(data)))
            return 0.5

        # Synthetic task accuracies exercise the complete runner, not model quality.
        evaluator.__globals__["evaluate_task"] = observe_task
        with patch.object(prepare, "_load_nanochat_core", return_value=evaluator):
            with contextlib.redirect_stdout(io.StringIO()):
                result = prepare.evaluate_core(object(), object(), "cpu")
        with zipfile.ZipFile(Path(prepare.CORE_DIR) / "eval_bundle.zip") as archive:
            expected_counts = [
                len(archive.read("eval_bundle/eval_data/" + task["dataset_uri"]).splitlines())
                for task in self.tasks
            ]
        self.assertEqual(len(calls), 22)
        self.assertEqual([count for _, count in calls], expected_counts)
        self.assertEqual(list(result["results"]), [task["label"] for task in self.tasks])
        self.assertEqual(list(result["centered_results"]), list(result["results"]))
        self.assertEqual(calls[0][0]["num_fewshot"], 0)
        self.assertEqual(calls[1][0]["num_fewshot"], 10)
        self.assertEqual(calls[1][0]["continuation_delimiter"], "\nAnswer: ")
        self.assertTrue(math.isfinite(result["core_metric"]))

    def test_adapts_the_existing_tokenizer_and_logits_to_each_upstream_task_type(self):
        evaluator = prepare._load_nanochat_core()
        evaluate_example = evaluator.__globals__["evaluate_task"].__globals__["evaluate_example"]
        tokenizer = prepare.Tokenizer.from_directory()
        vocab_size = tokenizer.get_vocab_size()
        observed = []

        class Model:
            def __call__(self, input_ids):
                observed.append(input_ids)
                return torch.zeros(*input_ids.shape, vocab_size)

        examples = [
            ("multiple_choice", {"query": "Which letter?", "choices": ["a", "b"], "gold": 0}),
            ("schema", {"context_options": ["A cat sat.", "A dog ran."], "continuation": " It moved.", "gold": 0}),
            ("language_modeling", {"context": "A letter is", "continuation": " a"}),
        ]
        for task_type, item in examples:
            with self.subTest(task_type=task_type):
                metadata = {"task_type": task_type, "num_fewshot": 0, "continuation_delimiter": " "}
                result = evaluate_example(0, Model(), tokenizer, [item], "cpu", metadata)
                self.assertIsInstance(result, bool)
                self.assertTrue(torch.all(observed[-1][:, 0] == tokenizer.get_bos_token_id()))

    def test_surfaces_incomplete_and_nonfinite_evaluations(self):
        labels = [task["label"] for task in self.tasks]
        complete = {"results": dict.fromkeys(labels, 0.5),
                    "centered_results": dict.fromkeys(labels, 0.25), "core_metric": 0.25}
        incomplete = {**complete, "results": dict.fromkeys(labels[:-1], 0.5)}
        invalid = {**complete, "core_metric": math.nan}
        for result in (incomplete, invalid):
            with self.subTest(result=result):
                with patch.object(prepare, "_load_nanochat_core", return_value=lambda *a, **kw: result):
                    with self.assertRaises(RuntimeError):
                        prepare.evaluate_core(object(), object(), "cpu")

    def test_preserves_full_precision_json_and_the_native_per_task_csv(self):
        labels = [task["label"] for task in self.tasks]
        result = {"results": dict.fromkeys(labels, 0.31234567890123456),
                  "centered_results": dict.fromkeys(labels, 0.12345678901234566),
                  "core_metric": 0.12345678901234566}
        with tempfile.TemporaryDirectory() as directory:
            prepare.save_core_results(result, directory)
            stored = json.loads((Path(directory) / "core.json").read_text())
            with (Path(directory) / "core.csv").open(newline="") as f:
                rows = [[field.strip() for field in row] for row in csv.reader(f)]
        self.assertEqual(stored, result)
        self.assertEqual(rows[0], ["Task", "Accuracy", "Centered"])
        self.assertEqual([row[0] for row in rows[1:-1]], labels)
        self.assertEqual(rows[-1], ["CORE", "", "0.123457"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
