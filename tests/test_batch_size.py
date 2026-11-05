from gpu_config import active_gpu
import ast
import math
from pathlib import Path
import unittest

import torch

from prepare import MAX_SEQ_LEN
from tools.cpu_smoke import load_model_namespace


class BatchSizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_model_namespace()

    def make_optimizer(self, beta2):
        parameter = torch.nn.Parameter(torch.zeros(3, 7))
        optimizer = self.namespace['MuonAdamW']([dict(kind='adamw', params=[parameter],
            lr=0.0, betas=(0.8, beta2), eps=1e-10, weight_decay=0.0)])
        return parameter, optimizer

    def test_keeps_the_nominal_second_moment_half_life_in_tokens(self):
        old_tokens, new_tokens = 262144, 131072
        for beta2 in (0.96, 0.995, 0.95):
            with self.subTest(beta2=beta2):
                parameter, optimizer = self.make_optimizer(beta2)
                self.namespace['scale_adam_second_moments'](optimizer, new_tokens / old_tokens)
                beta1, adjusted = optimizer.param_groups[0]['betas']
                before = old_tokens * math.log(0.5) / math.log(beta2)
                after = new_tokens * math.log(0.5) / math.log(adjusted)
                self.assertAlmostEqual(after / before, 1.0, places=12)
                self.assertEqual(beta1, 0.8)
                self.assertGreater(adjusted, beta2)
                self.assertLess(adjusted, 1.0)

    def test_decays_existing_variance_equally_over_the_same_token_interval(self):
        for beta2 in (0.96, 0.995, 0.95):
            with self.subTest(beta2=beta2):
                parent_parameter, parent = self.make_optimizer(beta2)
                child_parameter, child = self.make_optimizer(beta2)
                self.namespace['scale_adam_second_moments'](child, 0.5)
                for parameter, optimizer in ((parent_parameter, parent), (child_parameter, child)):
                    parameter.grad = torch.zeros_like(parameter)
                    optimizer.step()
                    optimizer.state[parameter]['exp_avg_sq'].fill_(1.0)
                parent.step()
                child.step()
                child.step()
                torch.testing.assert_close(parent.state[parent_parameter]['exp_avg_sq'],
                    child.state[child_parameter]['exp_avg_sq'], rtol=1e-6, atol=1e-7)

    def test_applies_the_batch_scaled_matrix_decay_with_zero_gradients(self):
        initial = torch.linspace(-1, 1, 21).reshape(1, 3, 7)
        results = {}
        for name, decay, steps in (('parent', 0.28, 1), ('scaled', 0.14, 2), ('unscaled', 0.28, 2)):
            parameters = initial.clone()
            gradients = torch.zeros_like(parameters)
            momentum = torch.zeros_like(parameters)
            second_moment = torch.zeros(1, 1, 7)
            for _ in range(steps):
                self.namespace['muon_step_fused'](gradients, parameters, momentum, second_moment,
                    torch.tensor(0.95), torch.tensor(0.04), torch.tensor(decay), torch.tensor(0.9), 5, -2)
            coefficient = float(torch.tensor(0.04, dtype=torch.bfloat16)
                                * torch.tensor(decay, dtype=torch.bfloat16))
            torch.testing.assert_close(parameters, initial * (1 - coefficient)**steps,
                                       rtol=1e-6, atol=1e-7)
            results[name] = parameters
        scaled_error = (results['parent'] - results['scaled']).abs().max()
        unscaled_error = (results['parent'] - results['unscaled']).abs().max()
        self.assertLess(scaled_error, unscaled_error / 100)
        self.assertGreater(scaled_error, 0)

    def test_trains_one_device_batch_per_update_with_the_production_optimizer_recipe(self):
        namespace = load_model_namespace()
        tree = ast.parse(active_gpu().train_source().read_text())
        constants = {'TOTAL_BATCH_SIZE', 'BATCH_REFERENCE_SIZE', 'DEVICE_BATCH_SIZE', 'EMBEDDING_LR',
                     'UNEMBEDDING_LR', 'MATRIX_LR', 'SCALAR_LR', 'WEIGHT_DECAY',
                     'WARMUP_RATIO', 'WARMDOWN_RATIO', 'FINAL_LR_FRAC'}
        names = {'get_lr_multiplier', 'get_muon_momentum', 'get_weight_decay'}
        selected = [node for node in tree.body if
                    (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                     isinstance(node.targets[0], ast.Name) and node.targets[0].id in constants) or
                    (isinstance(node, ast.FunctionDef) and node.name in names)]
        exec(compile(ast.Module(body=selected, type_ignores=[]), 'train.py', 'exec'), namespace)
        accumulation = namespace['TOTAL_BATCH_SIZE'] // (namespace['DEVICE_BATCH_SIZE'] * MAX_SEQ_LEN)
        self.assertEqual(accumulation, 1)
        torch.manual_seed(42)
        config = namespace['GPTConfig'](sequence_len=16, vocab_size=64, n_layer=2,
            n_head=5, n_kv_head=1, n_embd=80, training_loss_chunk_size=7)
        with torch.device('meta'):
            model = namespace['GPT'](config)
        model.to_empty(device='cpu')
        model.init_weights()
        namespace['model'] = model
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and
                          any(isinstance(target, ast.Name) and target.id == 'optimizer' for target in node.targets))
        adjustment = next(node for node in tree.body if isinstance(node, ast.Expr) and
                          isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and
                          node.value.func.id == 'scale_adam_second_moments')
        exec(compile(ast.Module(body=[assignment, adjustment], type_ignores=[]), 'train.py', 'exec'), namespace)
        optimizer = namespace['optimizer']
        namespace['adamw_step_fused'] = torch.compile(namespace['adamw_step_fused'],
            backend='aot_eager', fullgraph=True, dynamic=False)
        average = namespace['TailEMA'](model, start=0.8, tau=0.075, blend=0.6)
        compiled = torch.compile(model, backend='aot_eager', fullgraph=True, dynamic=False)
        inputs = torch.arange(34).reshape(2, 17) % 64
        targets = (inputs + 1) % 64
        before = model.lm_head.weight.detach().clone()
        for step, progress in ((0, 0.0), (400, 0.8), (5000, 0.95)):
            for _ in range(accumulation):
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    loss = compiled(inputs, targets) / accumulation
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * namespace['get_lr_multiplier'](progress)
                if group['kind'] == 'muon':
                    group['momentum'] = namespace['get_muon_momentum'](step, progress)
                    group['weight_decay'] = namespace['get_weight_decay'](progress)
            optimizer.step()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertTrue(torch.isfinite(parameter).all())
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
