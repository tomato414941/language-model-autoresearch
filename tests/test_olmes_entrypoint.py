from copy import deepcopy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import prepare
from evaluation.arc_macro import APPROVED_CONFIG_FIX


class OlmesEntrypointTests(unittest.TestCase):
    def evaluate_fixture(self, suites, repeats=1):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = dict(complete=True, olmes_revision=prepare.OLMES_REVISION,
                            approved_config_fix=deepcopy(APPROVED_CONFIG_FIX), model_source_sha256="fixture",
                            suites=suites, peak_vram_mb=2.0)

            def export(model, destination):
                Path(destination).write_bytes(b"fixed own checkpoint")
                return "fixture"

            def execute(command, **kwargs):
                self.assertEqual(command[1:3], ["-m", "evaluation.run_prepared"])
                self.assertEqual(kwargs["cwd"], root)
                output = root / command[command.index("--output") + 1]
                output.mkdir()
                (output / "summary.json").write_text(json.dumps(expected))
                (output / "metrics-all.jsonl").write_text('{"synthetic": true}\n')

            tokenizer = types.SimpleNamespace(get_vocab_size=lambda: 8192)
            with patch.object(prepare, "__file__", str(root / "prepare.py")), \
                    patch("evaluation.checkpoint.export_checkpoint", side_effect=export), \
                    patch.object(prepare.subprocess, "run", side_effect=execute) as run, \
                    patch.object(prepare.torch.cuda, "empty_cache"), \
                    patch.object(prepare.torch.cuda, "memory_allocated", return_value=1024 ** 2):
                result = prepare.evaluate_olmes(object(), tokenizer)
                if repeats > 1:
                    with self.assertRaises(FileExistsError):
                        prepare.evaluate_olmes(object(), tokenizer)
                    self.assertEqual(run.call_count, 1)
            return result

    def test_returns_the_complete_native_suite_values_and_combined_memory(self):
        values = dict(qa_bpb=1.1, math_bpb=1.2, code_bpb=1.3)
        result = self.evaluate_fixture(values)
        self.assertEqual(result["suites"], values)
        self.assertEqual(result["combined_peak_vram_mb"], 3.0)

    def test_requires_all_three_finite_positive_suite_values(self):
        for values in (dict(qa_bpb=1, math_bpb=1), dict(qa_bpb=1, math_bpb=float("nan"), code_bpb=1),
                       dict(qa_bpb=1, math_bpb=0, code_bpb=1)):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    self.evaluate_fixture(values)

    def test_preserves_the_fixed_checkpoint_after_one_evaluation_attempt(self):
        self.evaluate_fixture(dict(qa_bpb=1, math_bpb=1, code_bpb=1), repeats=2)


if __name__ == "__main__":
    unittest.main()
