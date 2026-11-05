import copy
import unittest

import torch
import torch.nn.functional as F

from tools.cpu_smoke import load_model_namespace


class TrainingLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_model_namespace()

    def make_model(self, chunk_size=7):
        torch.manual_seed(42)
        config = self.namespace['GPTConfig'](
            sequence_len=16, vocab_size=64, n_layer=2, n_head=5,
            n_kv_head=1, n_embd=80, training_loss_chunk_size=chunk_size)
        with torch.device('meta'):
            model = self.namespace['GPT'](config)
        model.to_empty(device='cpu')
        model.init_weights()
        return model

    def test_matches_dense_softcapped_loss_and_gradients_across_masked_uneven_chunks(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = self.make_model()
                dense_head = copy.deepcopy(model.lm_head)
                hidden = torch.randn(2, 17, 80, requires_grad=True)
                dense_hidden = hidden.detach().clone().requires_grad_()
                targets = torch.arange(34).reshape(2, 17) % 64
                targets.reshape(-1)[:7] = -1
                targets[1, -2:] = -1
                with torch.autocast('cpu', dtype=dtype, enabled=dtype != torch.float32):
                    actual = model._training_loss(hidden, targets)
                    logits = dense_head(dense_hidden).float()
                    expected = F.cross_entropy(
                        (15 * torch.tanh(logits / 15)).reshape(-1, 64), targets.reshape(-1),
                        ignore_index=-1)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                (actual * 0.25).backward()
                (expected * 0.25).backward()
                tolerance = dict(rtol=0.02, atol=2e-5) if dtype == torch.bfloat16 else dict(rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(hidden.grad, dense_hidden.grad, **tolerance)
                torch.testing.assert_close(model.lm_head.weight.grad, dense_head.weight.grad, **tolerance)

    def test_preserves_the_mean_loss_contract_when_every_target_is_ignored(self):
        model = self.make_model()
        hidden = torch.randn(2, 17, 80, requires_grad=True)
        targets = torch.full((2, 17), -1)
        actual = model._training_loss(hidden, targets)
        logits = model.lm_head(hidden).float()
        expected = F.cross_entropy((15 * torch.tanh(logits / 15)).reshape(-1, 64),
                                   targets.reshape(-1), ignore_index=-1)
        self.assertTrue(torch.isnan(actual))
        self.assertTrue(torch.isnan(expected))

    def test_returns_consistent_logits_and_per_token_losses_for_evaluation(self):
        model = self.make_model()
        inputs = torch.arange(34).reshape(2, 17) % 64
        targets = (inputs + 1) % 64
        targets[0, :7] = -1
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                with torch.no_grad(), torch.autocast('cpu', dtype=dtype, enabled=dtype != torch.float32):
                    model.train()
                    training_logits = model(inputs)
                    model.eval()
                    logits = model(inputs)
                    per_token = model(inputs, targets, reduction='none')
                    mean = model(inputs, targets)
                    total = model(inputs, targets, reduction='sum')
                expected = F.cross_entropy(logits.reshape(-1, 64), targets.reshape(-1),
                                           ignore_index=-1, reduction='none')
                torch.testing.assert_close(logits, training_logits, rtol=0, atol=0)
                torch.testing.assert_close(per_token, expected, rtol=0, atol=0)
                torch.testing.assert_close(total, expected.sum())
                torch.testing.assert_close(mean, expected.sum() / (targets != -1).sum())

    def test_propagates_the_dense_objective_through_the_complete_model(self):
        model = self.make_model()
        dense = copy.deepcopy(model)
        dense.config = copy.copy(dense.config)
        dense.config.training_loss_chunk_size = 0
        inputs = torch.arange(34).reshape(2, 17) % 64
        targets = (inputs + 1) % 64
        targets[0, :7] = -1
        actual = model(inputs, targets)
        expected = dense(inputs, targets)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        for (name, parameter), (dense_name, dense_parameter) in zip(
                model.named_parameters(), dense.named_parameters(), strict=True):
            self.assertEqual(name, dense_name)
            self.assertIsNotNone(parameter.grad)
            torch.testing.assert_close(parameter.grad, dense_parameter.grad, rtol=2e-5, atol=1e-7)

    def test_trains_compiled_chunks_with_accumulation_and_returns_the_declared_average(self):
        model = self.make_model()
        optimizer = model.setup_optimizer(embedding_lr=0.03, matrix_lr=0.02)
        average = self.namespace['TailEMA'](model, start=0.8, tau=0.075, blend=0.6)
        compiled = torch.compile(model, backend='aot_eager', fullgraph=True, dynamic=False)
        inputs = torch.arange(34).reshape(2, 17) % 64
        targets = (inputs + 1) % 64
        targets[0, :7] = -1
        before = model.lm_head.weight.detach().clone()
        for progress in (0.0, 0.8, 0.95):
            for _ in range(2):
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    loss = compiled(inputs, targets) / 2
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
            model.zero_grad(set_to_none=True)
            average.update(progress)
        average.apply()
        compiled.eval()
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            self.assertTrue(torch.isfinite(compiled(inputs)).all())
            self.assertTrue(torch.isfinite(compiled(inputs, targets, reduction='none')).all())
        self.assertFalse(torch.equal(before, model.lm_head.weight))
        self.assertEqual(average.num_updates, 2)


if __name__ == '__main__':
    unittest.main()
