import contextlib
import copy
import types
import unittest

import torch

from tools.cpu_smoke import load_model_namespace


def separate_projection(self, x):
    return self.c_q(x), self.c_k(x), self.c_v(x)


class AttentionProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_model_namespace()

    def check_projection(self, kv_heads, layer, dtype):
        torch.manual_seed(42)
        config = self.namespace['GPTConfig'](n_embd=64, n_head=4, n_kv_head=kv_heads, n_layer=4)
        packed = self.namespace['CausalSelfAttention'](config, layer)
        reference = copy.deepcopy(packed)
        reference.project_qkv = types.MethodType(separate_projection, reference)
        x = torch.randn(2, 17, 64, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        ve = torch.randn(2, 17, kv_heads * 16, requires_grad=True) if packed.ve_gate is not None else None
        reference_ve = ve.detach().clone().requires_grad_() if ve is not None else None
        angles = torch.randn(1, 17, 1, 8)
        rotary = (angles.cos().bfloat16(), angles.sin().bfloat16())
        autocast = lambda: torch.autocast('cpu', dtype=dtype) if dtype == torch.bfloat16 else contextlib.nullcontext()
        with autocast():
            actual = packed(x, ve, rotary, (5, 0))
            expected = reference(reference_x, reference_ve, rotary, (5, 0))
        tolerances = dict(rtol=0.02, atol=0.003) if dtype == torch.bfloat16 else dict(rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual, expected, **tolerances)
        upstream = torch.randn_like(actual)
        (actual * upstream).sum().backward()
        (expected * upstream).sum().backward()
        if dtype == torch.bfloat16:
            # The packed backward sums Q/K/V contributions before BF16 rounding.
            difference = x.grad - reference_x.grad
            epsilon = torch.finfo(torch.bfloat16).eps
            self.assertLessEqual(difference.norm().item(), epsilon * reference_x.grad.norm().item())
            self.assertLessEqual(difference.abs().max().item(), epsilon * reference_x.grad.abs().max().item())
        else:
            torch.testing.assert_close(x.grad, reference_x.grad, **tolerances)
        for (name, parameter), (reference_name, reference_parameter) in zip(
                packed.named_parameters(), reference.named_parameters(), strict=True):
            self.assertEqual(name, reference_name)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            with self.subTest(parameter=name):
                torch.testing.assert_close(parameter.grad, reference_parameter.grad, **tolerances)
        if ve is not None:
            torch.testing.assert_close(ve.grad, reference_ve.grad, **tolerances)

    def test_matches_separate_projection_outputs_and_gradients_for_multihead_attention(self):
        for layer in (0, 3):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(layer=layer, dtype=dtype):
                    self.check_projection(4, layer, dtype)

    def test_matches_separate_projection_outputs_and_gradients_for_grouped_query_attention(self):
        for layer in (0, 3):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(layer=layer, dtype=dtype):
                    self.check_projection(2, layer, dtype)


if __name__ == '__main__':
    unittest.main()
