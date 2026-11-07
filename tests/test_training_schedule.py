from gpu_config import active_gpu
import ast
import math
import unittest
from unittest.mock import patch


class TrainingScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(active_gpu().train_source().read_text())
        constants = {'WARMUP_RATIO', 'WARMDOWN_RATIO', 'FINAL_LR_FRAC', 'WEIGHT_DECAY',
                     'TOTAL_BATCH_SIZE', 'BATCH_REFERENCE_SIZE'}
        functions = {'get_lr_multiplier', 'get_muon_momentum', 'get_weight_decay'}
        nodes = [node for node in tree.body if
                 (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                  isinstance(node.targets[0], ast.Name) and node.targets[0].id in constants) or
                 (isinstance(node, ast.FunctionDef) and node.name in functions)]
        cls.namespace = {'math': math}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'train.py', 'exec'), cls.namespace)
        cls.warmup_tokens = 104857600

    def warmup_end(self):
        return math.ceil(self.warmup_tokens / self.namespace['TOTAL_BATCH_SIZE'])

    def test_warms_momentum_before_holding_a_plateau(self):
        momentum = self.namespace['get_muon_momentum']
        warmup_end = self.warmup_end()
        values = [momentum(step, 0.1) for step in (0, 100, 200, warmup_end - 1, warmup_end)]
        self.assertEqual(values, sorted(values))
        self.assertLess(values[0], values[-1])
        self.assertEqual(values[-1], momentum(2 * warmup_end, 0.1))

    def test_keeps_momentum_warmup_at_the_same_processed_token_count(self):
        momentum = self.namespace['get_muon_momentum']
        tokens = (0, 26214400, 52428800, 78643200, 104857600, 131072000)
        for progress in (0.1, 0.8, 1.0):
            schedules = []
            for batch in (262144, 131072, self.namespace['TOTAL_BATCH_SIZE']):
                with patch.dict(self.namespace, TOTAL_BATCH_SIZE=batch):
                    schedules.append([momentum(count // batch, progress) for count in tokens])
            for schedule in schedules[1:]:
                self.assertEqual(schedules[0], schedule)

    def test_keeps_nominal_matrix_decay_exposure_per_token(self):
        decay = self.namespace['get_weight_decay']
        for progress in (0.0, 0.2, 0.35, 0.8, 1.0):
            exposures = []
            for batch in (262144, 131072, self.namespace['TOTAL_BATCH_SIZE']):
                with patch.dict(self.namespace, TOTAL_BATCH_SIZE=batch):
                    exposures.append(decay(progress) / batch)
            for exposure in exposures[1:]:
                self.assertEqual(exposures[0], exposure)

    def test_lowers_momentum_during_the_learning_rate_cooldown(self):
        momentum = self.namespace['get_muon_momentum']
        learning_rate = self.namespace['get_lr_multiplier']
        start = 1.0 - self.namespace['WARMDOWN_RATIO']
        progress = [start + (1.0 - start) * part / 10 for part in range(11)]
        values = [momentum(self.warmup_end(), point) for point in progress]
        rates = [learning_rate(point) for point in progress]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertEqual(rates, sorted(rates, reverse=True))
        self.assertLess(values[-1], values[0])
        self.assertGreater(values[-1], momentum(0, 0.0))

    def test_finishes_the_step_warmup_before_applying_a_time_cooldown(self):
        momentum = self.namespace['get_muon_momentum']
        self.assertEqual(momentum(100, 0.0), momentum(100, 1.0))

    def test_holds_momentum_when_no_cooldown_is_configured(self):
        previous = self.namespace['WARMDOWN_RATIO']
        try:
            self.namespace['WARMDOWN_RATIO'] = 0.0
            momentum = self.namespace['get_muon_momentum']
            values = [momentum(self.warmup_end(), point) for point in (0.0, 0.5, 1.0)]
            self.assertEqual(values, [0.97] * 3)
        finally:
            self.namespace['WARMDOWN_RATIO'] = previous

    def test_smoothly_reduces_weight_decay_to_zero(self):
        decay = self.namespace['get_weight_decay']
        peak = (self.namespace['WEIGHT_DECAY'] * self.namespace['TOTAL_BATCH_SIZE']
                / self.namespace['BATCH_REFERENCE_SIZE'])
        values = [decay(part / 100) for part in range(101)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertEqual(values[0], peak)
        self.assertEqual(values[-1], 0.0)
        self.assertAlmostEqual(decay(0.5), peak / 2)
        delta = 1e-5
        self.assertLess((decay(0.0) - decay(delta)) / delta, peak * 1e-3)
        self.assertLess(decay(1.0 - delta) / delta, peak * 1e-3)


if __name__ == '__main__':
    unittest.main()
