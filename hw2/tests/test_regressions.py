"""Маленькие регрессионные проверки без скачивания модели."""
import unittest
from unittest.mock import patch

import torch

from src.inspect_model import PeakMemory, forward_hooks, group_table, parameter_rows


class RegressionTests(unittest.TestCase):
    def test_tied_weight_has_two_names_but_one_contribution(self):
        model = torch.nn.Module()
        model.embed_tokens = torch.nn.Embedding(7, 3)
        model.lm_head = torch.nn.Linear(3, 7, bias=False)
        model.lm_head.weight = model.embed_tokens.weight
        groups = {r['group']: r for r in group_table(parameter_rows(model))}
        self.assertEqual(groups['embed']['params'], 21)
        self.assertEqual(groups['lm_head']['params'], 0)
        self.assertEqual(groups['lm_head']['tied_params'], 21)

    def test_hooks_cleaned_on_failure_and_existing_hook_preserved(self):
        layer = torch.nn.Identity()
        existing = layer.register_forward_hook(lambda *args: None)
        try:
            with self.assertRaisesRegex(RuntimeError, 'forward failed'):
                with forward_hooks({'layer': layer}) as values:
                    layer(torch.ones(1, 2, 3))
                    self.assertEqual(len(values['layer']), 2)
                    raise RuntimeError('forward failed')
            self.assertEqual(len(layer._forward_hooks), 1)
        finally:
            existing.remove()

    def test_mps_peak_survives_later_deallocation(self):
        with patch('torch.mps.synchronize'), patch(
            'torch.mps.driver_allocated_memory', side_effect=[100, 500, 200, 100]
        ):
            # Длинный интервал исключает фоновое чтение в коротком тесте.
            with PeakMemory(torch.device('mps'), interval=60) as peak:
                peak.sample()
                peak.sample()
            self.assertEqual(peak.used, 500)

    def test_cuda_uses_hardware_peak_counter(self):
        with patch('torch.cuda.synchronize'), patch('torch.cuda.reset_peak_memory_stats') as reset, patch(
            'torch.cuda.max_memory_allocated', return_value=123456
        ):
            with PeakMemory(torch.device('cuda')) as peak:
                pass
            self.assertEqual(peak.used, 123456)
            reset.assert_called_once()


if __name__ == '__main__':
    unittest.main()
