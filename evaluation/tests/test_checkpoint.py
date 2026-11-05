from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from evaluation.checkpoint import export_checkpoint, load_checkpoint, load_definition
from tools.cpu_smoke import attention


class CheckpointTests(unittest.TestCase):
    def test_transfers_the_model_definition_parameters_buffers_and_inference(self):
        torch.set_num_threads(2)
        source = Path("train.py").read_text()
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


if __name__ == "__main__":
    unittest.main()
