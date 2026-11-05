import unittest

import torch
from torch._dynamo.backends.registry import lookup_backend

from tools.cpu_smoke import load_model_namespace


class GroupedAdamWTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_model_namespace()

    def make_pair(self):
        torch.manual_seed(42)
        shapes = ((3, 7), (9,), (), (4, 5))
        actual = [torch.nn.Parameter(torch.randn(shape)) for shape in shapes]
        reference = [torch.nn.Parameter(p.detach().clone()) for p in actual]
        settings = dict(lr=0.03, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01)
        optimizer = self.namespace['MuonAdamW']([dict(kind='adamw', params=actual, **settings)])
        oracle = torch.optim.AdamW(reference, foreach=False, fused=False, **settings)
        return actual, reference, optimizer, oracle

    def compare(self, actual, reference, optimizer, oracle):
        for parameter, expected in zip(actual, reference, strict=True):
            torch.testing.assert_close(parameter, expected, rtol=3e-5, atol=1e-6)
            if parameter not in optimizer.state:
                self.assertNotIn(expected, oracle.state)
                continue
            state, expected_state = optimizer.state[parameter], oracle.state[expected]
            self.assertEqual(state['step'], int(expected_state['step']))
            self.assertEqual(state['step_t'].item(), state['step'])
            for key in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(state[key], expected_state[key], rtol=3e-5, atol=1e-7)

    def update(self, actual, reference, optimizer, oracle, learning_rate, active):
        optimizer.param_groups[0]['lr'] = learning_rate
        oracle.param_groups[0]['lr'] = learning_rate
        for index, (parameter, expected) in enumerate(zip(actual, reference, strict=True)):
            gradient = torch.randn_like(parameter) if index in active else None
            parameter.grad = gradient
            expected.grad = gradient.clone() if gradient is not None else None
        optimizer.step()
        oracle.step()
        self.compare(actual, reference, optimizer, oracle)

    def test_updates_mixed_shapes_and_moments_with_the_adamw_objective(self):
        actual, reference, optimizer, oracle = self.make_pair()
        for rate in (0.03, 0.0, 0.001, 0.02, 0.005):
            self.update(actual, reference, optimizer, oracle, rate, set(range(4)))

    def test_counts_only_updates_with_a_gradient_for_each_parameter(self):
        actual, reference, optimizer, oracle = self.make_pair()
        for active in (set(), {0, 3}, {0, 1, 2, 3}, {1, 2}, {0, 1, 2, 3}):
            self.update(actual, reference, optimizer, oracle, 0.01, active)
        self.assertEqual([optimizer.state[p]['step'] for p in actual], [3, 3, 3, 3])

    def test_reuses_one_compiled_graph_when_learning_rates_and_steps_change(self):
        actual, reference, optimizer, oracle = self.make_pair()
        backend = lookup_backend('aot_eager')
        graphs = []

        def count_graph(graph, inputs):
            graphs.append(graph)
            return backend(graph, inputs)

        compiled = torch.compile(self.namespace['adamw_step_fused'], backend=count_graph,
                                 fullgraph=True, dynamic=False)
        original = self.namespace['adamw_step_fused']
        self.namespace['adamw_step_fused'] = compiled
        try:
            for rate in (0.03, 0.001, 0.0, 0.02):
                self.update(actual, reference, optimizer, oracle, rate, set(range(4)))
        finally:
            self.namespace['adamw_step_fused'] = original
        self.assertEqual(len(graphs), 1)

    def test_trains_with_compiled_model_and_grouped_updates_before_tail_averaging(self):
        namespace = load_model_namespace()
        namespace['adamw_step_fused'] = torch.compile(namespace['adamw_step_fused'],
            backend='aot_eager', fullgraph=True, dynamic=False)
        torch.manual_seed(42)
        config = namespace['GPTConfig'](sequence_len=16, vocab_size=64, n_layer=2,
            n_head=5, n_kv_head=1, n_embd=80, training_loss_chunk_size=7)
        with torch.device('meta'):
            model = namespace['GPT'](config)
        model.to_empty(device='cpu')
        model.init_weights()
        optimizer = model.setup_optimizer(embedding_lr=0.03, matrix_lr=0.02)
        average = namespace['TailEMA'](model, start=0.8, tau=0.075, blend=0.6)
        compiled = torch.compile(model, backend='aot_eager', fullgraph=True, dynamic=False)
        inputs = torch.arange(34).reshape(2, 17) % 64
        targets = (inputs + 1) % 64
        before = model.lm_head.weight.detach().clone()
        for progress in (0.0, 0.8, 0.95):
            for _ in range(2):
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    loss = compiled(inputs, targets) / 2
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
            optimizer.step()
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            model.zero_grad(set_to_none=True)
            average.update(progress)
        average.apply()
        compiled.eval()
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            self.assertTrue(torch.isfinite(compiled(inputs)).all())
        self.assertFalse(torch.equal(before, model.lm_head.weight))
        self.assertEqual(average.num_updates, 2)


if __name__ == '__main__':
    unittest.main()
