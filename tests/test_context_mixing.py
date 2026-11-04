import contextlib
import unittest

import torch

from tools.cpu_smoke import load_model_namespace


class ContextMixingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_model_namespace()

    def make_model(self, dtype=torch.float32, active_blocks=False):
        torch.manual_seed(42)
        config = self.namespace['GPTConfig'](
            sequence_len=32, vocab_size=64, n_layer=4, n_head=2, n_kv_head=2, n_embd=32)
        model = self.namespace['GPT'](config)
        model.init_weights()
        if dtype == torch.float32:
            model.float()
        if active_blocks:
            with torch.no_grad():
                for block in model.transformer.h:
                    block.attn.c_proj.weight.normal_(std=0.02)
                    block.mlp.c_proj.weight.normal_(std=0.02)
        return model

    def test_uses_previous_embedding_to_predict_the_current_token(self):
        model = self.make_model()
        with torch.no_grad():
            model.smear_gain.fill_(0.6)
            model.smear_gate.weight.zero_()
            inputs = torch.tensor([[1, 2, 3, 4]])
            changed = inputs.clone()
            changed[:, 0] = 5
            original_logits = model(inputs)
            changed_logits = model(changed)
        self.assertGreater((original_logits[:, 1] - changed_logits[:, 1]).norm().item(), 1e-6)

    def test_predicts_causally_with_active_context_mixing_and_single_token_inputs(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = self.make_model(dtype, active_blocks=True)
                with torch.no_grad():
                    model.smear_gain.fill_(0.7)
                    inputs = torch.randint(0, 64, (2, 17))
                    changed = inputs.clone()
                    changed[:, 8:] = (changed[:, 8:] + 1) % 64
                    autocast = (torch.autocast('cpu', dtype=dtype) if dtype == torch.bfloat16
                                else contextlib.nullcontext())
                    with autocast:
                        logits = model(inputs)
                        changed_logits = model(changed)
                        first_logits = model(inputs[:, :1])
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(first_logits).all())
                torch.testing.assert_close(logits[:, :8], changed_logits[:, :8], rtol=0, atol=0)
                tolerances = (dict(rtol=2 * torch.finfo(torch.bfloat16).eps, atol=2e-5)
                              if dtype == torch.bfloat16 else dict(rtol=2e-5, atol=2e-6))
                torch.testing.assert_close(first_logits[:, 0], logits[:, 0], **tolerances)

    def test_updates_context_mixing_parameters_after_gradient_accumulation(self):
        model = self.make_model(torch.bfloat16, active_blocks=True)
        with torch.no_grad():
            model.smear_gain.fill_(0.4)
        parameters = (model.smear_gate.weight, model.smear_gain, model.backout_gain)
        before = [parameter.detach().clone() for parameter in parameters]
        optimizer = model.setup_optimizer(embedding_lr=0.03, matrix_lr=0.02, weight_decay=0.2)
        grouped = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(parameter) for parameter in model.parameters()})
        self.assertEqual(model.num_scaling_params()['total'], sum(p.numel() for p in model.parameters()))
        compiled = torch.compile(model, backend='aot_eager', fullgraph=True, dynamic=False)
        for _ in range(2):
            inputs = torch.randint(0, 64, (2, 17))
            targets = torch.randint(0, 64, (2, 17))
            with torch.autocast('cpu', dtype=torch.bfloat16):
                loss = compiled(inputs, targets) / 2
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(), 0)
        optimizer.step()
        for original, parameter in zip(before, parameters, strict=True):
            self.assertTrue(torch.isfinite(parameter).all())
            self.assertGreater((parameter - original).norm().item(), 0)


if __name__ == '__main__':
    unittest.main()
