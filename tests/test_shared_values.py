import unittest

import torch

from tools.cpu_smoke import load_model_namespace


class SharedValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_model_namespace()

    def make_model(self, shared=True, layers=8, kv_heads=5):
        torch.manual_seed(42)
        config = self.namespace['GPTConfig'](sequence_len=16, vocab_size=64, n_layer=layers,
            n_head=5, n_kv_head=kv_heads, n_embd=80, share_value_embeddings=shared)
        with torch.device('meta'):
            model = self.namespace['GPT'](config)
        model.to_empty(device='cpu')
        model.init_weights()
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.c_proj.weight.normal_(std=0.02)
                block.mlp.c_proj.weight.normal_(std=0.02)
        return model

    def test_matches_copied_layer_values_and_sums_their_parameter_gradients(self):
        for layers, kv_heads in ((1, 1), (8, 5)):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(layers=layers, kv_heads=kv_heads, dtype=dtype):
                    shared = self.make_model(layers=layers, kv_heads=kv_heads)
                    separate = self.make_model(shared=False, layers=layers, kv_heads=kv_heads)
                    with torch.no_grad():
                        for name, parameter in shared.named_parameters():
                            if not name.startswith('value_embeds.'):
                                separate.get_parameter(name).copy_(parameter)
                        for layer, key in shared.value_embed_keys.items():
                            separate.value_embeds[str(layer)].weight.copy_(shared.value_embeds[key].weight)
                    inputs = torch.arange(34).reshape(2, 17) % 64
                    with torch.autocast('cpu', dtype=dtype, enabled=dtype != torch.float32):
                        actual = shared(inputs)
                        expected = separate(inputs)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    upstream = torch.randn_like(actual)
                    (actual * upstream).sum().backward()
                    (expected * upstream).sum().backward()
                    tolerance = dict(rtol=0.02, atol=0.001) if dtype == torch.bfloat16 else dict(rtol=3e-5, atol=2e-6)
                    for name, parameter in shared.named_parameters():
                        if not name.startswith('value_embeds.'):
                            torch.testing.assert_close(parameter.grad, separate.get_parameter(name).grad, **tolerance)
                    for key, embedding in shared.value_embeds.items():
                        contributions = [separate.value_embeds[str(layer)].weight.grad
                                         for layer, layer_key in shared.value_embed_keys.items() if layer_key == key]
                        combined = torch.stack(contributions).sum(dim=0)
                        self.assertGreater(combined.norm().item(), 0)
                        if dtype == torch.bfloat16:
                            bound = 2 * torch.finfo(dtype).eps * combined.norm().item() + 1e-6
                            self.assertLessEqual((embedding.weight.grad - combined).norm().item(), bound)
                        else:
                            torch.testing.assert_close(embedding.weight.grad, combined, **tolerance)

    def test_reuses_each_lookup_within_a_forward_and_refreshes_it_for_new_inputs(self):
        model = self.make_model(layers=10).eval()
        calls = {key: 0 for key in model.value_embeds}
        observed = {}
        handles = []
        for key, embedding in model.value_embeds.items():
            def count(module, args, output, key=key):
                calls[key] += 1
            handles.append(embedding.register_forward_hook(count))
        for layer, key in model.value_embed_keys.items():
            def record(module, args, key=key):
                observed.setdefault(key, []).append(args[1])
            handles.append(model.transformer.h[layer].attn.register_forward_pre_hook(record))
        inputs = torch.arange(34).reshape(2, 17) % 64
        with torch.no_grad():
            first = model(inputs)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(count == 1 for count in calls.values()))
        for values in observed.values():
            self.assertTrue(all(value.data_ptr() == values[0].data_ptr() for value in values))
        with torch.no_grad():
            changed = model((inputs + 1) % 64)
        for handle in handles:
            handle.remove()
        self.assertTrue(all(count == 2 for count in calls.values()))
        self.assertGreater((changed - first).norm().item(), 0)

    def test_updates_each_shared_parameter_once_with_consistent_model_accounting(self):
        for layers, kv_heads in ((1, 1), (10, 5)):
            with self.subTest(layers=layers, kv_heads=kv_heads):
                model = self.make_model(layers=layers, kv_heads=kv_heads)
                optimizer = model.setup_optimizer(embedding_lr=0.03, matrix_lr=0.02)
                grouped = [parameter for group in optimizer.param_groups for parameter in group['params']]
                self.assertEqual(len(grouped), len({id(parameter) for parameter in grouped}))
                self.assertEqual({id(parameter) for parameter in grouped}, {id(parameter) for parameter in model.parameters()})
                self.assertEqual(model.num_scaling_params()['total'], sum(p.numel() for p in model.parameters()))
                inputs = torch.arange(34).reshape(2, 17) % 64
                loss = model(inputs, (inputs + 1) % 64)
                loss.backward()
                optimizer.step()
                for embedding in model.value_embeds.values():
                    self.assertEqual(optimizer.state[embedding.weight]['step'], 1)
                    self.assertTrue(torch.isfinite(embedding.weight.grad).all())
                    self.assertTrue(torch.isfinite(embedding.weight).all())

    def test_trains_compiled_pairs_then_returns_causal_logits_after_averaging(self):
        namespace = load_model_namespace()
        namespace['adamw_step_fused'] = torch.compile(namespace['adamw_step_fused'],
            backend='aot_eager', fullgraph=True, dynamic=False)
        torch.manual_seed(42)
        config = namespace['GPTConfig'](sequence_len=16, vocab_size=64, n_layer=6,
            n_head=5, n_kv_head=1, n_embd=80, training_loss_chunk_size=7)
        with torch.device('meta'):
            model = namespace['GPT'](config)
        model.to_empty(device='cpu')
        model.init_weights()
        optimizer = model.setup_optimizer(embedding_lr=0.03, matrix_lr=0.02)
        namespace['scale_adam_second_moments'](optimizer, 0.5)
        average = namespace['TailEMA'](model, start=0.8, tau=0.075, blend=0.6)
        compiled = torch.compile(model, backend='aot_eager', fullgraph=True, dynamic=False)
        inputs = torch.arange(34).reshape(2, 17) % 64
        targets = (inputs + 1) % 64
        before = [embedding.weight.detach().clone() for embedding in model.value_embeds.values()]
        for progress in (0.0, 0.8, 0.95):
            with torch.autocast('cpu', dtype=torch.bfloat16):
                loss = compiled(inputs, targets)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
            model.zero_grad(set_to_none=True)
            average.update(progress)
        average.apply()
        compiled.eval()
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            logits = compiled(inputs)
            losses = compiled(inputs, targets, reduction='none')
            changed = inputs.clone()
            changed[:, 9:] = (changed[:, 9:] + 1) % 64
            changed_logits = compiled(changed)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(losses).all())
        torch.testing.assert_close(logits[:, :9], changed_logits[:, :9], rtol=0, atol=0)
        self.assertTrue(all(not torch.equal(old, embedding.weight) for old, embedding in zip(before, model.value_embeds.values(), strict=True)))
        self.assertEqual(average.num_updates, 2)


if __name__ == '__main__':
    unittest.main()
