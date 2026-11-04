import unittest

import torch

from tools.cpu_smoke import load_model_namespace


class WideningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_model_namespace()

    def make_model(self, layers=4, kv_heads=4, narrow=True):
        torch.manual_seed(42)
        config = self.namespace['GPTConfig'](
            sequence_len=16, vocab_size=64, n_layer=layers, n_head=4, n_kv_head=kv_heads,
            n_embd=64, narrow_first_layer=narrow)
        with torch.device('meta'):
            model = self.namespace['GPT'](config)
        model.to_empty(device='cpu')
        model.init_weights()
        return model

    def test_widens_the_first_stream_and_preserves_the_prediction_interface(self):
        counts = {}
        for narrow in (False, True):
            with self.subTest(narrow=narrow):
                model = self.make_model(narrow=narrow)
                observed = []
                first = model.transformer.h[0]
                before = first.register_forward_pre_hook(lambda module, args: observed.append(args[0].shape[-1]))
                after = first.register_forward_hook(lambda module, args, output: observed.append(output.shape[-1]))
                inputs = torch.randint(0, 64, (2, 21))
                with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
                    logits = model(inputs)
                before.remove()
                after.remove()
                self.assertEqual(observed, [32 if narrow else 64, 64])
                self.assertEqual(logits.shape, (2, 21, 64))
                self.assertTrue(torch.isfinite(logits).all())
                counts[narrow] = model.num_scaling_params()['total']
                self.assertEqual(counts[narrow], sum(parameter.numel() for parameter in model.parameters()))
        self.assertLess(counts[True], counts[False])

    def test_trains_multihead_and_grouped_query_models_with_first_layer_value_embeddings(self):
        for layers in (1, 4):
            for kv_heads in (2, 4):
                with self.subTest(layers=layers, kv_heads=kv_heads):
                    model = self.make_model(layers=layers, kv_heads=kv_heads)
                    optimizer = model.setup_optimizer(embedding_lr=0.03, matrix_lr=0.02, weight_decay=0.2)
                    grouped = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
                    self.assertEqual(len(grouped), len(set(grouped)))
                    self.assertEqual(set(grouped), {id(parameter) for parameter in model.parameters()})
                    before = model.transformer.wte.weight.detach().clone()
                    inputs = torch.randint(0, 64, (2, 21))
                    targets = torch.randint(0, 64, inputs.shape)
                    for _ in range(3):
                        with torch.autocast('cpu', dtype=torch.bfloat16):
                            losses = model(inputs, targets, reduction='none')
                        self.assertEqual(losses.numel(), targets.numel())
                        self.assertTrue(torch.isfinite(losses).all())
                        losses.mean().backward()
                        for parameter in model.parameters():
                            self.assertIsNotNone(parameter.grad)
                            self.assertTrue(torch.isfinite(parameter.grad).all())
                        optimizer.step()
                        model.zero_grad(set_to_none=True)
                    self.assertFalse(torch.equal(before, model.transformer.wte.weight))


if __name__ == '__main__':
    unittest.main()
