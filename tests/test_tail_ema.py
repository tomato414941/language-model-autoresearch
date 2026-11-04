import math
import unittest

import torch
import torch.nn as nn

from tools.cpu_smoke import load_model_namespace


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict({'wte': nn.Embedding(2, 2)})
        self.value_embeds = nn.ModuleDict({'0': nn.Embedding(2, 2)})
        self.projection = nn.Linear(2, 2, bias=False)
        self.gain = nn.Parameter(torch.ones(1))
        self.register_buffer('rotary', torch.ones(2))


class TailEMATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_model_namespace()

    def average(self, model):
        return self.namespace['TailEMA'](model, start=0.8, tau=0.075, blend=0.6)

    def fill(self, model, value):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(value)
            model.rotary.fill_(value)

    def test_starts_in_the_declared_tail_and_preserves_live_weights_until_final_blending(self):
        model = TinyModel()
        average = self.average(model)
        self.fill(model, 1)
        average.update(0.79)
        self.assertEqual(average.num_updates, 0)
        self.fill(model, 2)
        average.update(0.8)
        self.fill(model, 4)
        average.update(0.875)
        for parameter in model.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 4))
        expected_average = 2 * math.exp(-1) + 4 * (1 - math.exp(-1))
        elapsed = average.apply()
        self.assertGreaterEqual(elapsed, 0)
        for name, parameter in model.named_parameters():
            value = 4 if name == 'transformer.wte.weight' else 0.4 * 4 + 0.6 * expected_average
            torch.testing.assert_close(parameter, torch.full_like(parameter, value))
        torch.testing.assert_close(model.rotary, torch.full_like(model.rotary, 4))

    def test_uses_elapsed_training_time_for_a_constant_iterate_across_different_update_intervals(self):
        outputs = []
        for progress in ((0.95,), (0.85, 0.95)):
            model = TinyModel()
            average = self.average(model)
            self.fill(model, 1)
            average.update(0.8)
            self.fill(model, 3)
            for point in progress:
                average.update(point)
            average.apply()
            outputs.append([p.detach().clone() for p in model.parameters()])
        for coarse, fine in zip(*outputs, strict=True):
            torch.testing.assert_close(coarse, fine)

    def test_keeps_float32_averages_for_bfloat16_parameters_and_rounds_the_returned_blend(self):
        model = TinyModel().bfloat16()
        average = self.average(model)
        self.fill(model, 1)
        average.update(0.8)
        self.fill(model, 1.5)
        average.update(0.9)
        for tensor in average.average:
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertFalse(tensor.requires_grad)
        expected = 0.4 * 1.5 + 0.6 * (math.exp(-0.1 / 0.075) + 1.5 * (1 - math.exp(-0.1 / 0.075)))
        average.apply()
        torch.testing.assert_close(model.projection.weight,
                                   torch.full_like(model.projection.weight, expected), rtol=0, atol=0)

    def test_returns_shared_blended_parameters_to_compiled_and_eager_model_calls(self):
        torch.manual_seed(42)
        config = self.namespace['GPTConfig'](
            sequence_len=16, vocab_size=64, n_layer=2, n_head=5, n_kv_head=1, n_embd=80)
        with torch.device('meta'):
            model = self.namespace['GPT'](config)
        model.to_empty(device='cpu')
        model.init_weights()
        average = self.average(model)
        compiled = torch.compile(model, backend='aot_eager', fullgraph=True, dynamic=False)
        original_ids = [id(p) for p in model.parameters()]
        inputs = torch.randint(0, 64, (2, 17))
        with torch.no_grad():
            original = model(inputs)
            average.update(0.8)
            model.lm_head.weight.mul_(2)
            average.update(0.9)
            average.apply()
            compiled.eval()
            expected = model(inputs)
            actual = compiled(inputs)
        self.assertFalse(model.training)
        self.assertEqual(original_ids, [id(p) for p in compiled.parameters()])
        self.assertIs(compiled._orig_mod, model)
        self.assertGreater((expected - original).norm().item(), 0)
        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
