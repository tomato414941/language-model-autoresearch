import copy
import unittest

import torch
import torch.nn.functional as F

from tools.cpu_smoke import load_model_namespace


class ValueGatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_model_namespace()

    def check_uniform_attention(self, kv_heads, value_embedding):
        torch.manual_seed(42)
        config = self.namespace['GPTConfig'](n_embd=32, n_head=4, n_kv_head=kv_heads, n_layer=4)
        layer = 3 if value_embedding else 0
        model = self.namespace['CausalSelfAttention'](config, layer).double()
        with torch.no_grad():
            model.c_q.weight.zero_()
            model.c_k.weight.zero_()
            model.c_proj.weight.copy_(torch.eye(32))
        reference = copy.deepcopy(model)
        inputs = torch.randn(2, 7, 32, dtype=torch.float64, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_()
        values = torch.randn(2, 7, kv_heads * 8, dtype=torch.float64, requires_grad=True) if value_embedding else None
        reference_values = values.detach().clone().requires_grad_() if value_embedding else None
        angles = torch.randn(1, 7, 1, 4, dtype=torch.float64)
        actual = model(inputs, values, (angles.cos(), angles.sin()), (2, 0))

        projected = F.linear(reference_inputs, reference.c_v.weight)
        if value_embedding:
            gates = 2 / (1 + (-F.linear(reference_inputs[..., :reference.ve_gate_channels],
                                       reference.ve_gate.weight)).exp())
            projected = projected + (gates[..., None] * reference_values.view(2, 7, kv_heads, 8)).flatten(2)
        gated = projected * torch.sigmoid(projected)
        head_values = gated.view(2, 7, kv_heads, 8).repeat_interleave(4 // kv_heads, dim=2)
        expected = torch.stack([head_values[:, max(0, index - 2):index + 1].mean(dim=1)
                                for index in range(7)], dim=1).flatten(2)
        expected = F.linear(expected, reference.c_proj.weight)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)

        upstream = torch.randn_like(actual)
        (actual * upstream).sum().backward()
        (expected * upstream).sum().backward()
        torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=1e-10, atol=1e-12)
        for name in ('c_v.weight', 'c_proj.weight', 've_gate.weight'):
            parameters = dict(model.named_parameters())
            reference_parameters = dict(reference.named_parameters())
            if name in parameters:
                torch.testing.assert_close(parameters[name].grad, reference_parameters[name].grad,
                                           rtol=1e-10, atol=1e-12)
        if value_embedding:
            torch.testing.assert_close(values.grad, reference_values.grad, rtol=1e-10, atol=1e-12)

    def test_aggregates_self_gated_values_with_causal_sliding_windows(self):
        for kv_heads in (4, 2):
            with self.subTest(kv_heads=kv_heads):
                self.check_uniform_attention(kv_heads, False)

    def test_backpropagates_through_self_gated_value_embedding_mixtures(self):
        for kv_heads in (4, 2):
            with self.subTest(kv_heads=kv_heads):
                self.check_uniform_attention(kv_heads, True)


if __name__ == '__main__':
    unittest.main()
