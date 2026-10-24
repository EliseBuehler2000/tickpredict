"""Fixed predictor checks using CPU tensors and temporary synthetic fixtures only."""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_predictor as evaluator
from inference import preprocess_image
from modeling import preprocessing_spec


class CornerClassifier(torch.nn.Module):
    """Orientation-sensitive logits make averaging mistakes visible."""

    def forward(self, images):
        signal = images[:, 0, 0, 0] + 2 * images[:, 0, 0, -1]
        return torch.stack([signal, torch.zeros_like(signal)], dim=1)


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='tick-predictor-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = CornerClassifier().eval()
        self.batch = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4) / 10
        self.cpu = torch.device('cpu')
        self.classes = ['adult female', 'nymph']
        self.preflight = {'source_sha256': 'synthetic-source',
                          'manifest_sha256': 'synthetic-manifest'}
        self.records = [
            {'tick_id': 'fixture-a', 'life_stage': 'adult female',
             'engorgement_level': 'unengorged'},
            {'tick_id': 'fixture-b', 'life_stage': 'nymph',
             'engorgement_level': 'semi-engorged'},
        ]
        self.files = []
        for index in range(2):
            directory = self.root / f'run{index}'
            directory.mkdir()
            classes = self.classes if index == 0 else self.classes[::-1]
            provenance = {
                'config': {'target': 'life_stage', 'architecture': 'resnet18'},
                'classes': classes, 'smoke_only': False, 'test_images_used': 0,
                'validation_n': 110, **self.preflight,
            }
            (directory / 'provenance.json').write_text(json.dumps(provenance))
            (directory / 'completion.json').write_text(json.dumps({
                'status': 'completed', 'smoke_only': False}))
            confidence = 0.9 if index == 0 else 0.7
            predictions = [
                {'tick_id': r['tick_id'], 'true': r['life_stage'],
                 'predicted': r['life_stage'],
                 'uncalibrated_probabilities': [
                     confidence if c == r['life_stage'] else 1 - confidence for c in classes]}
                for r in self.records
            ]
            path = directory / 'validation.json'
            path.write_text(json.dumps({'classes': classes, 'predictions': predictions}))
            self.files.append(str(path))
        download = patch('torch.hub.download_url_to_file',
                         side_effect=AssertionError('Tests must never download weights.'))
        download.start()
        self.addCleanup(download.stop)

    def probabilities(self, batch=None, batch_size=3, tta='d4'):
        return evaluator._view_probabilities(
            self.model, self.batch if batch is None else batch, tta, self.cpu, batch_size)

    def test_eight_unique_d4_views_and_invariant_average(self):
        views = evaluator.d4_views(self.batch)
        self.assertEqual(len(views), 8)
        self.assertEqual(len({v[0].numpy().tobytes() for v in views}), 8)
        torch.testing.assert_close(self.probabilities(), self.probabilities(views[5]),
                                   rtol=0, atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'square NCHW'):
            evaluator.d4_views(torch.zeros(1, 3, 4, 5))

    def test_averages_probabilities_not_logits(self):
        logits = torch.stack([self.model(view) for view in evaluator.d4_views(self.batch)])
        expected = logits.softmax(-1).double().mean(0)
        torch.testing.assert_close(self.probabilities(), expected, rtol=0, atol=1e-12)
        self.assertFalse(torch.allclose(self.probabilities(), logits.mean(0).softmax(-1).double()))

    def test_batch_size_does_not_change_view_weighting(self):
        expected = self.probabilities(batch_size=1)
        for size in (3, 5, 16):
            torch.testing.assert_close(self.probabilities(batch_size=size), expected,
                                       rtol=0, atol=1e-12)
        torch.testing.assert_close(self.probabilities(tta='none'),
                                   self.model(self.batch).softmax(1).double(), rtol=0, atol=1e-12)

    def test_cached_ensemble_aligns_classes_and_reports_confidence(self):
        # Two records exercise joining/alignment directly; the public evaluator
        # separately enforces the project's full 110-record manifest.
        probabilities, _ = evaluator._cached_probabilities(
            self.files, 'life_stage', self.records, self.classes, self.preflight)
        np.testing.assert_allclose(probabilities, [[0.8, 0.2], [0.2, 0.8]])
        result = evaluator._details(self.records, probabilities, self.classes, 'life_stage')
        self.assertEqual(result['correct'], 2)
        self.assertAlmostEqual(result['nll'], -math.log(0.8))
        self.assertAlmostEqual(result['ece10'], 0.2)
        self.assertAlmostEqual(result['brier'], 0.08)

    def test_cached_predictions_reject_foreign_and_duplicate_ids(self):
        path = Path(self.files[0])
        validation = json.loads(path.read_text())
        for bad_id in ('fixture-b', 'not-in-development'):
            with self.subTest(tick_id=bad_id):
                validation['predictions'][0]['tick_id'] = bad_id
                path.write_text(json.dumps(validation))
                with self.assertRaisesRegex(ValueError, 'Prediction IDs'):
                    evaluator._cached_probabilities(
                        self.files, 'life_stage', self.records, self.classes, self.preflight)

    def checkpoint(self, smoke=False):
        directory = Path(self.files[0]).parent
        spec = preprocessing_spec(64)
        provenance = json.loads((directory / 'provenance.json').read_text())
        provenance['preprocessing'] = spec
        (directory / 'provenance.json').write_text(json.dumps(provenance))
        path = directory / 'best.pt'
        torch.save({'target': 'life_stage', 'classes': self.classes, 'architecture': 'resnet18',
                    'epoch': 1, 'state_dict': {}, 'preprocessing': spec, 'smoke_only': smoke}, path)
        return path, spec

    def test_smoke_checkpoint_rejected_before_model_or_image_loading(self):
        checkpoint, _ = self.checkpoint(smoke=True)
        with patch.object(evaluator, 'build_model') as build, \
                patch.object(evaluator, 'preprocess_image') as preprocess:
            with self.assertRaisesRegex(ValueError, 'smoke checkpoint'):
                evaluator.predict_image(self.root / 'must-not-open.png', {
                    'target': 'life_stage', 'checkpoints': [str(checkpoint)], 'tta': 'd4'})
            build.assert_not_called()
            preprocess.assert_not_called()

    def test_unknown_paths_duplicates_and_weight_knobs_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing input file'):
            evaluator.load_spec({'target': 'life_stage', 'checkpoints': [str(self.root / 'absent.pt')]})
        with self.assertRaisesRegex(ValueError, 'Duplicate inputs'):
            evaluator.load_spec({'target': 'life_stage', 'prediction_files': [self.files[0]] * 2})
        with self.assertRaisesRegex(ValueError, 'Spec accepts only'):
            evaluator.load_spec({'target': 'life_stage', 'prediction_files': self.files, 'weights': [0.9, 0.1]})
        with self.assertRaisesRegex(ValueError, 'D4 requires checkpoints'):
            evaluator.load_spec({'target': 'life_stage', 'prediction_files': self.files, 'tta': 'd4'})

    def test_single_image_uses_shared_preprocessing_and_fixed_d4(self):
        checkpoint, preprocessing = self.checkpoint()
        image = self.root / 'synthetic.png'
        Image.fromarray(np.arange(32 * 40 * 3, dtype=np.uint8).reshape(32, 40, 3)).save(image)
        spec = {'target': 'life_stage', 'checkpoints': [str(checkpoint)], 'tta': 'd4'}
        with patch.object(evaluator, 'build_model', return_value=CornerClassifier()):
            result = evaluator.predict_image(image, spec, batch_size=3)
        expected = self.probabilities(preprocess_image(image, preprocessing)[None], batch_size=2)[0]
        np.testing.assert_allclose(list(result['model_scores_uncalibrated'].values()), expected.numpy())
        self.assertFalse(result['confidence_calibrated'])


if __name__ == '__main__':
    unittest.main()
