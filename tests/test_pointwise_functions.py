import unittest

import torch
import torch.nn.functional as F
from torch._dynamo.backends.registry import lookup_backend

from tools.cpu_smoke import load_model_namespace


class PointwiseFunctionTests(unittest.TestCase):
    def test_preserves_rotary_query_key_values_and_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            for dimension, kv_heads in ((64, 5), (128, 1)):
                with self.subTest(dtype=dtype, dimension=dimension, kv_heads=kv_heads):
                    namespace = load_model_namespace()
                    torch._dynamo.reset()
                    compiled = torch.compile(namespace['prepare_qk'], backend='aot_eager',
                                             dynamic=True, fullgraph=True)
                    torch.manual_seed(42)
                    q = torch.randn(2, 17, 5, dimension, dtype=dtype, requires_grad=True)
                    k = torch.randn(2, 17, kv_heads, dimension, dtype=dtype, requires_grad=True)
                    reference_q = q.detach().clone().requires_grad_()
                    reference_k = k.detach().clone().requires_grad_()
                    angles = torch.randn(1, 17, 1, 64)
                    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
                    actual_q, actual_k = compiled(q, k, cos, sin)

                    def previous(value):
                        half = value.size(-1) // 2
                        first, second = value[..., :half], value[..., half:]
                        c, s = cos[..., :half], sin[..., :half]
                        rotated = torch.cat((first * c + second * s,
                                             first * (-s) + second * c), dim=-1)
                        return F.rms_norm(rotated, (dimension,)) * 1.2

                    expected_q, expected_k = previous(reference_q), previous(reference_k)
                    tolerance = dict(rtol=0.02, atol=0.003) if dtype == torch.bfloat16 else dict(rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(actual_q, expected_q, **tolerance)
                    torch.testing.assert_close(actual_k, expected_k, **tolerance)
                    upstream_q, upstream_k = torch.randn_like(q), torch.randn_like(k)
                    ((actual_q * upstream_q).sum() + (actual_k * upstream_k).sum()).backward()
                    ((expected_q * upstream_q).sum() + (expected_k * upstream_k).sum()).backward()
                    torch.testing.assert_close(q.grad, reference_q.grad, **tolerance)
                    torch.testing.assert_close(k.grad, reference_k.grad, **tolerance)

    def test_preserves_the_mlp_activation_values_and_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                namespace = load_model_namespace()
                torch._dynamo.reset()
                if 'gated_activation' in namespace:
                    compiled = torch.compile(namespace['gated_activation'], backend='aot_eager',
                                             dynamic=True, fullgraph=True)
                    up = torch.linspace(-3, 3, 70, dtype=dtype).reshape(2, 5, 7).requires_grad_()
                    gate = torch.linspace(3, -3, 70, dtype=dtype).reshape(2, 5, 7).requires_grad_()
                    sigmoid = torch.sigmoid(gate.detach().double())
                    expected_activation = gate.detach().double() * sigmoid
                    expected = expected_activation.to(dtype) * up.detach()
                    actual = compiled(up, gate)
                    tolerance = dict(rtol=0.02, atol=0.003) if dtype == torch.bfloat16 else dict(rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(actual, expected, **tolerance)
                    actual.sum().backward()
                    expected_gate_gradient = (up.detach().double() * sigmoid * (
                        1 + gate.detach().double() * (1 - sigmoid))).to(dtype)
                    torch.testing.assert_close(up.grad, expected_activation.to(dtype), **tolerance)
                    torch.testing.assert_close(gate.grad, expected_gate_gradient, **tolerance)
                else:
                    compiled = torch.compile(namespace['squared_relu'], backend='aot_eager',
                                             dynamic=True, fullgraph=True)
                    values = torch.linspace(-3, 3, 70, dtype=dtype).reshape(2, 5, 7).requires_grad_()
                    actual = compiled(values)
                    torch.testing.assert_close(actual, F.relu(values).square(), rtol=0, atol=0)
                    actual.sum().backward()
                    expected_gradient = torch.where(values > 0, 2 * values, torch.zeros_like(values))
                    torch.testing.assert_close(values.grad, expected_gradient, rtol=0, atol=0)

    def test_returns_float32_softcapped_logits_and_preserves_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                namespace = load_model_namespace()
                torch._dynamo.reset()
                compiled = torch.compile(namespace['softcap_logits'], backend='aot_eager',
                                         dynamic=True, fullgraph=True)
                values = torch.linspace(-90, 90, 30, dtype=dtype).reshape(2, 3, 5).requires_grad_()
                actual = compiled(values)
                expected = 15 * torch.tanh(values.float() / 15)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue((actual >= -15).all() and (actual <= 15).all())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.sum().backward()
                expected_gradient = (1 - torch.tanh(values.float() / 15).square()).to(dtype)
                tolerance = dict(rtol=0.02, atol=0.003) if dtype == torch.bfloat16 else dict(rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(values.grad, expected_gradient, **tolerance)

    def test_trains_with_compiled_functions_and_reuses_graphs_after_parameter_updates(self):
        namespace = load_model_namespace()
        torch._dynamo.reset()
        backend = lookup_backend('aot_eager')
        graphs = []

        def count_graph(graph, inputs):
            graphs.append(graph)
            return backend(graph, inputs)

        activation = 'gated_activation' if 'gated_activation' in namespace else 'squared_relu'
        for name in ('prepare_qk', activation, 'softcap_logits'):
            namespace[name] = torch.compile(namespace[name], backend=count_graph,
                                            dynamic=True, fullgraph=True)
        torch.manual_seed(42)
        config = namespace['GPTConfig'](sequence_len=32, vocab_size=64, n_layer=6,
            n_head=5, n_kv_head=1, n_embd=80, training_loss_chunk_size=7)
        with torch.device('meta'):
            model = namespace['GPT'](config)
        model.to_empty(device='cpu')
        model.init_weights()
        optimizer = model.setup_optimizer(embedding_lr=0.03, matrix_lr=0.02, weight_decay=0.14)
        compiled_model = torch.compile(model, backend=count_graph, dynamic=False, fullgraph=True)
        inputs = torch.arange(34).reshape(2, 17) % 64
        targets = (inputs + 1) % 64
        initial = model.lm_head.weight.detach().clone()
        after_first_update = None
        for _ in range(3):
            with torch.autocast('cpu', dtype=torch.bfloat16):
                loss = compiled_model(inputs, targets)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
            optimizer.step()
            model.zero_grad(set_to_none=True)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            if after_first_update is None:
                after_first_update = len(graphs)
            self.assertEqual(len(graphs), after_first_update)
        self.assertFalse(torch.equal(initial, model.lm_head.weight))

        model.eval()
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            for length in (1, 17, 33):
                batch = torch.arange(2 * length).reshape(2, length) % 64
                logits = model(batch)
                self.assertEqual(tuple(logits.shape), (2, length, 64))
                self.assertTrue(torch.isfinite(logits).all())
            changed = inputs.clone()
            changed[:, 8:] = (changed[:, 8:] + 7) % 64
            actual = model(inputs)
            perturbed = model(changed)
            torch.testing.assert_close(actual[:, :8], perturbed[:, :8], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

