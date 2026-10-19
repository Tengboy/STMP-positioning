import copy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from artifact_support import checked_relative, verify_validation_selection


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.metrics = [
            {'event': 'validation', 'step': 1000, 'count': 2000,
             'raw_loss': 10., 'ema_loss': 8., 'raw_diagnostics': [], 'ema_diagnostics': []},
            {'event': 'validation', 'step': 2000, 'count': 2000,
             'raw_loss': 7., 'ema_loss': 9., 'raw_diagnostics': [], 'ema_diagnostics': []},
            {'event': 'finished', 'step': 2000}]
        self.status = {'state': 'training_finished', 'step': 2000,
                       'best_metric': 7., 'best_variant': 'raw'}
        self.checkpoint = {'step': 2000, 'best_metric': 7., 'weight_variant': 'raw'}

    def test_checkpoint_selection_uses_complete_fixed_validation(self):
        chosen = verify_validation_selection(self.metrics, self.status, self.checkpoint)
        self.assertEqual((chosen['step'], chosen['variant'], chosen['loss']), (2000, 'raw', 7.))
        bad = copy.deepcopy(self.metrics)
        bad[1]['count'] = 30
        with self.assertRaisesRegex(ValueError, 'population'):
            verify_validation_selection(bad, self.status, self.checkpoint)

    def test_selected_checkpoint_must_match_global_best_and_weight_variant(self):
        bad = dict(self.checkpoint, step=1000, best_metric=8., weight_variant='ema')
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            verify_validation_selection(self.metrics, self.status, bad)
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            verify_validation_selection(self.metrics, self.status, dict(self.checkpoint, weight_variant='ema'))

    def test_resumed_or_failed_training_cannot_be_frozen(self):
        with self.assertRaisesRegex(ValueError, 'finished'):
            verify_validation_selection(self.metrics + [{'event': 'started'}], self.status, self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'finished'):
            verify_validation_selection(self.metrics, dict(self.status, state='failed'), self.checkpoint)

    def test_export_paths_cannot_escape_or_name_partial_files(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT / 'tests')) as name:
            base = Path(name)
            self.assertEqual(checked_relative(base, 'results/raw/a.mat'), base / 'results/raw/a.mat')
            for value in ('../original.py', '/etc/passwd', 'C:/elsewhere', 'results\\x.mat',
                          'results/../x.mat', 'results/a.partial', 'results//a.mat'):
                with self.subTest(path=value), self.assertRaises(ValueError):
                    checked_relative(base, value)


if __name__ == '__main__':
    unittest.main()
