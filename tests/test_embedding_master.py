import unittest

import torch

from tools.cpu_smoke import load_model_namespace


class EmbeddingMasterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_model_namespace()

    def make_model(self, master=True, layers=4, kv_heads=5):
        torch.manual_seed(42)
        config = self.namespace['GPTConfig'](
            sequence_len=16, vocab_size=64, n_layer=layers, n_head=5,
            n_kv_head=kv_heads, n_embd=80, float32_embedding_master=master)
        with torch.device('meta'):
            model = self.namespace['GPT'](config)
        model.to_empty(device='cpu')
        model.init_weights()
        return model

    def test_matches_initial_predictions_with_bfloat16_lookup_values(self):
        models = [self.make_model(master=master) for master in (False, True)]
        inputs = torch.arange(42).reshape(2, 21) % 64
        outputs = []
        for model in models:
            with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
                outputs.append(model(inputs))
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)

    def test_preserves_small_master_updates_until_lookup_rounding_changes(self):
        observed = {}
        for master in (False, True):
            model = self.make_model(master=master, layers=1)
            optimizer = model.setup_optimizer(embedding_lr=0.03, matrix_lr=0.0, scalar_lr=0.0)
            weight = model.transformer.wte.weight
            for group in optimizer.param_groups:
                if any(parameter is weight for parameter in group['params']):
                    group['lr'] = 0.6
                    group['weight_decay'] = 0.001
            with torch.no_grad():
                weight.fill_(1.0)
            before = model.transformer.wte(torch.tensor([0])).bfloat16().detach().clone()
            for parameter in model.parameters():
                parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            first = model.transformer.wte(torch.tensor([0])).bfloat16().detach().clone()
            torch.testing.assert_close(first, before, rtol=0, atol=0)
            for _ in range(3):
                optimizer.step()
            observed[master] = model.transformer.wte(torch.tensor([0])).bfloat16().detach().clone()
        self.assertTrue(torch.all(observed[True] < observed[False]))

    def test_trains_with_float32_lookup_gradients_and_adam_states(self):
        for layers, kv_heads in ((1, 1), (4, 5)):
            with self.subTest(layers=layers, kv_heads=kv_heads):
                model = self.make_model(layers=layers, kv_heads=kv_heads)
                optimizer = model.setup_optimizer(embedding_lr=0.03, matrix_lr=0.02)
                compiled = torch.compile(model, backend='aot_eager', fullgraph=True, dynamic=False)
                inputs = torch.randint(0, 64, (2, 21))
                targets = torch.randint(0, 64, inputs.shape)
                lookups = [model.transformer.wte, *model.value_embeds.values()]
                for _ in range(3):
                    for _ in range(2):
                        with torch.autocast('cpu', dtype=torch.bfloat16):
                            loss = compiled(inputs, targets)
                        self.assertTrue(torch.isfinite(loss))
                        (loss / 2).backward()
                    for lookup in lookups:
                        self.assertEqual(lookup.weight.dtype, torch.float32)
                        self.assertEqual(lookup.weight.grad.dtype, torch.float32)
                        self.assertTrue(torch.isfinite(lookup.weight.grad).all())
                    optimizer.step()
                    for lookup in lookups:
                        state = optimizer.state[lookup.weight]
                        self.assertEqual(state['exp_avg'].dtype, torch.float32)
                        self.assertEqual(state['exp_avg_sq'].dtype, torch.float32)
                        self.assertTrue(torch.isfinite(state['exp_avg_sq']).all())
                    model.zero_grad(set_to_none=True)


if __name__ == '__main__':
    unittest.main()
