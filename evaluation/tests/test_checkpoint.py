from gpu_config import active_gpu
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch
from oe_eval.components.requests import LoglikelihoodRequest

from evaluation.checkpoint import export_checkpoint, load_checkpoint, load_definition, sdpa_attention
from gpu_config import get_profile
from evaluation.model import CheckpointModel
from tools.cpu_smoke import attention


class CheckpointTests(unittest.TestCase):
    def test_restores_the_independent_rtx_model_with_its_native_sdpa_backend(self):
        torch.set_num_threads(2)
        source = get_profile('rtx5090').train_source().read_text()
        module = load_definition(source, sdpa_attention, compile_functions=False)
        model = module.GPT(module.GPTConfig(vocab_size=64, n_layer=4, n_embd=32, n_head=2, n_kv_head=2))
        model.init_weights()
        model.eval()
        inputs = torch.arange(24).reshape(2, 12) % 64
        with torch.inference_mode():
            expected = model(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rtx-model.pt'
            export_checkpoint(model, path, source)
            restored, _ = load_checkpoint(path, 'cpu', compile_functions=False)
            with torch.inference_mode():
                actual = restored(inputs)
        self.assertEqual(restored.config.attention_backend, 'sdpa')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_transfers_the_model_definition_parameters_buffers_and_inference(self):
        torch.set_num_threads(2)
        source = active_gpu().train_source().read_text()
        module = load_definition(source, attention, compile_functions=False)
        config = module.GPTConfig(vocab_size=64, n_layer=4, n_embd=32, n_head=2, n_kv_head=2)
        model = module.GPT(config)
        model.init_weights()
        model.eval()
        inputs = torch.arange(24).reshape(2, 12) % config.vocab_size
        with torch.inference_mode():
            expected = model(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            source_hash = export_checkpoint(model, path, source)
            restored, actual_hash = load_checkpoint(path, "cpu", attention=attention, compile_functions=False)
            with torch.inference_mode():
                actual = restored(inputs)
        self.assertEqual(actual_hash, source_hash)
        self.assertEqual(asdict(restored.config), asdict(model.config))
        for name, value in model.state_dict().items():
            self.assertEqual(restored.state_dict()[name].dtype, value.dtype)
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
        torch.testing.assert_close(restored.cos, model.cos, rtol=0, atol=0)
        torch.testing.assert_close(restored.sin, model.sin, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_preserves_native_likelihoods_token_counts_and_cache_results_with_larger_batches(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        source = active_gpu().train_source().read_text()
        module = load_definition(source, attention, compile_functions=False)
        model = module.GPT(module.GPTConfig(vocab_size=8192, n_layer=4, n_embd=32,
                                           n_head=2, n_kv_head=2))
        model.init_weights()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name == "lm_head.weight" or name.endswith("c_proj.weight"):
                    parameter.normal_(std=0.01)
        requests = [LoglikelihoodRequest(
            f"Synthetic question {index % 61}: " + "echo " * (index % 17),
            "日本語🙂" if index % 5 == 0 else "yes" if index % 2 else "no")
            for index in range(73)]
        requests += [LoglikelihoodRequest("", "An answer."),
                     LoglikelihoodRequest("long context " * 100, "final answer")]
        tokenizer = active_gpu().cache_directory("tokenizer") / "tokenizer.pkl"
        results, observed_shapes = [], []
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "random-model.pt"
            export_checkpoint(model, checkpoint, source)
            for batch_size in (8, 32):
                bridge = CheckpointModel(checkpoint, tokenizer, "cpu", attention=attention,
                                         compile_functions=False, batch_size=batch_size)
                bridge._max_length = 64
                bridge._shown_model_input = True
                original = bridge._model_call
                shapes = []

                def capture(inputs, **kwargs):
                    shapes.append(tuple(inputs.shape))
                    return original(inputs, **kwargs)

                bridge._model_call = capture
                results.append(bridge.loglikelihood_verbose(requests, disable_tqdm=True))
                observed_shapes.append(shapes)
        for field in ("num_tokens", "num_tokens_all", "is_greedy"):
            self.assertEqual([item[field] for item in results[0]], [item[field] for item in results[1]])
        torch.testing.assert_close(
            torch.tensor([item["sum_logits"] for item in results[0]]),
            torch.tensor([item["sum_logits"] for item in results[1]]), rtol=1e-5, atol=1e-4)
        self.assertEqual(max(shape[0] for shape in observed_shapes[0]), 8)
        self.assertEqual(max(shape[0] for shape in observed_shapes[1]), 32)
        self.assertLess(len(observed_shapes[1]), len(observed_shapes[0]))


if __name__ == "__main__":
    unittest.main()
