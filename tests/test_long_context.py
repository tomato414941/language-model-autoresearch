import unittest

import torch

from tools.cpu_smoke import load_model_namespace


class LongContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_model_namespace()

    def make_model(self, layers=2, pattern='SL'):
        torch.manual_seed(42)
        config = self.namespace['GPTConfig'](
            sequence_len=16, vocab_size=64, n_layer=layers, n_head=2, n_kv_head=2,
            n_embd=32, window_pattern=pattern)
        with torch.device('meta'):
            model = self.namespace['GPT'](config)
        model.to_empty(device='cpu')
        model.init_weights()
        return model.float()

    def identity_value_attention(self, model, index, inputs):
        attention = model.transformer.h[index].attn
        inputs = inputs[..., :attention.n_embd]
        with torch.no_grad():
            attention.c_q.weight.zero_()
            attention.c_k.weight.zero_()
            attention.c_v.weight.copy_(torch.eye(attention.n_embd))
            attention.c_proj.weight.copy_(torch.eye(attention.n_embd))
            rotary = (model.cos[:, :inputs.shape[1]], model.sin[:, :inputs.shape[1]])
            return attention(inputs, None, rotary, model.window_sizes[index])

    def test_long_attention_uses_distant_prefix_tokens_beyond_the_training_length(self):
        model = self.make_model()
        inputs = torch.zeros(1, 48, 32)
        inputs[:, 0, 0] = 1
        output = self.identity_value_attention(model, 1, inputs)
        self.assertAlmostEqual(output[0, -1, 0].item(), 1 / 48, places=6)

    def test_short_attention_uses_recent_tokens_within_a_long_input(self):
        model = self.make_model()
        distant = torch.zeros(1, 48, 32)
        distant[:, 0, 0] = 1
        recent = torch.zeros_like(distant)
        recent[:, -2, 0] = 1
        distant_output = self.identity_value_attention(model, 0, distant)
        recent_output = self.identity_value_attention(model, 0, recent)
        self.assertEqual(distant_output[0, -1, 0].item(), 0)
        self.assertGreater(recent_output[0, -1, 0].item(), 0)

    def test_trains_and_predicts_causally_on_inputs_longer_than_the_training_length(self):
        model = self.make_model(layers=4, pattern='SSSL')
        with torch.no_grad():
            model.smear_gain.fill_(0.5)
            for block in model.transformer.h:
                block.attn.c_proj.weight.normal_(std=0.02)
                block.mlp.c_proj.weight.normal_(std=0.02)
        inputs = torch.randint(0, 64, (2, 48))
        changed = inputs.clone()
        changed[:, 24:] = (changed[:, 24:] + 1) % 64
        targets = torch.randint(0, 64, inputs.shape)
        with torch.no_grad():
            logits = model(inputs)
            changed_logits = model(changed)
        self.assertTrue(torch.isfinite(logits).all())
        torch.testing.assert_close(logits[:, :24], changed_logits[:, :24], rtol=0, atol=0)
        losses = model(inputs, targets, reduction='none')
        self.assertEqual(losses.numel(), targets.numel())
        self.assertTrue(torch.isfinite(losses).all())
        losses.mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == '__main__':
    unittest.main()
