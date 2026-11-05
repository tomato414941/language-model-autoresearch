import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gpu_config
import train as dispatcher
from tools import rental


class GPUConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = copy.deepcopy(gpu_config.read_config())
        self.path = self.root / 'gpus.json'

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        self.path.write_text(json.dumps(self.config))

    def test_switches_source_history_and_budgets_from_one_active_setting(self):
        self.config['gpus']['rtx5090'].update(training_seconds=900, rental_seconds=2700, max_hourly_cost=0.8)
        expected = {'h100': (1800, 3600, 5), 'rtx5090': (900, 2700, 0.8)}
        for identifier in expected:
            self.config['active_gpu'] = identifier
            self.save()
            with patch.object(gpu_config, 'CONFIG_PATH', self.path):
                profile = gpu_config.active_gpu()
                self.assertEqual(profile.identifier, identifier)
                self.assertEqual(tuple(profile.settings().values()), expected[identifier])
                self.assertEqual(profile.train_source(self.root), self.root / 'gpus' / identifier / 'train.py')
                self.assertEqual(profile.results_directory(self.root), self.root / 'gpus' / identifier / 'results')
                self.assertIn(f'gpus/{identifier}/train.py', gpu_config.source_files())

    def test_requires_the_users_budgets_before_preparing_a_first_rtx_rental(self):
        self.config['active_gpu'] = 'rtx5090'
        self.save()
        profile = gpu_config.get_profile(path=self.path)
        with self.assertRaisesRegex(ValueError, 'user must set'):
            profile.settings()
        self.assertEqual(profile.settings(require=False), dict.fromkeys(gpu_config.SETTING_FIELDS))

    def test_uses_frozen_job_budgets_after_a_user_changes_the_active_gpu(self):
        self.config['active_gpu'] = 'rtx5090'
        self.save()
        job = dict(gpu_id='h100', gpu_name='H100 SXM',
                   gpu_settings=dict(training_seconds=1800, rental_seconds=3600, max_hourly_cost=5),
                   expected_eval_seconds=720)
        with patch.object(gpu_config, 'CONFIG_PATH', self.path):
            self.assertEqual(gpu_config.profile_for_job(job).identifier, 'h100')
            self.assertEqual(rental.execution_budget(job), 2745)
            self.assertEqual(rental.lease_seconds(job), 3600)

    def test_preserves_the_global_daily_limit_when_the_user_switches_gpu(self):
        with patch.object(rental, 'ROOT', self.root):
            rental.job_path('previous').parent.mkdir(parents=True)
            rental.save(dict(id='previous', gpu_id='h100', creation_requested_at=1000))
            self.config['active_gpu'] = 'rtx5090'
            self.save()
            with patch.object(gpu_config, 'CONFIG_PATH', self.path):
                with self.assertRaisesRegex(RuntimeError, 'last 24 hours'):
                    rental.check_daily_budget(2000)

    def test_selects_only_the_users_active_gpu_from_offers(self):
        self.config['active_gpu'] = 'rtx5090'
        self.save()
        base = dict(gpu_name='RTX 5090', num_gpus=1, cuda_max_good=12.8, gpu_ram=32607)
        offers = [dict(base, id=1, dph_total=0.7), dict(base, id=2, dph_total=0.5),
                  dict(base, id=3, gpu_name='H100 SXM', dph_total=0.1)]
        with patch.object(gpu_config, 'CONFIG_PATH', self.path):
            with patch.object(rental, 'api', return_value={'offers': offers}) as api:
                self.assertEqual(rental.choose_offer()['id'], 2)
        self.assertEqual(api.call_args.args[2]['gpu_name'], {'eq': 'RTX 5090'})

    def test_exports_the_selected_model_source_when_launched_through_the_dispatcher(self):
        import torch
        selected = self.root / 'gpus/h100/train.py'
        selected.parent.mkdir(parents=True)
        checkpoint = self.root / 'model.pt'
        selected.write_text(
            'from dataclasses import dataclass\nimport torch\n'
            'from evaluation.checkpoint import export_checkpoint\n'
            '@dataclass\nclass Config:\n    width: int = 1\n'
            'class GPT(torch.nn.Module):\n'
            '    def __init__(self):\n        super().__init__()\n'
            '        self.config = Config()\n'
            '        self.weight = torch.nn.Parameter(torch.ones(1))\n'
            f'export_checkpoint(GPT(), {str(checkpoint)!r})\n')
        profile = SimpleNamespace(settings=lambda: {}, train_source=lambda: selected)
        with patch.object(dispatcher, 'active_gpu', return_value=profile):
            dispatcher.main()
        payload = torch.load(checkpoint, weights_only=True)
        self.assertEqual(payload['source'], selected.read_text())
        self.assertEqual(payload['source_sha256'], hashlib.sha256(selected.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
