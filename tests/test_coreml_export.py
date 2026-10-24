"""Core ML wrapper checks using CPU tensors only; no conversion or images needed."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_predictor as evaluator

try:
    from export_coreml import MultiHeadPredictor, Predictor, d4_views
except ModuleNotFoundError as error:  # coremltools is installed on macOS only
    raise unittest.SkipTest(f'Core ML export unavailable: {error}')


class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(3 * 4 * 4, 3))

    def forward(self, images):
        return images.flatten(1) @ self.weights


class CoreMLExportTests(unittest.TestCase):
    def test_d4_views_match_evaluator_order(self):
        batch = torch.rand(2, 3, 5, 5)
        for ours, reference in zip(d4_views(batch), evaluator.d4_views(batch), strict=True):
            self.assertTrue(torch.equal(ours, reference))

    def test_predictor_matches_evaluator_pipeline(self):
        torch.manual_seed(0)
        model = Linear().eval()
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        pixels = torch.rand(1, 3, 4, 4) * 255
        normalized = (pixels / 255 - torch.tensor(mean).reshape(1, 3, 1, 1)) / torch.tensor(std).reshape(1, 3, 1, 1)
        for tta in ('none', 'd4'):
            with torch.inference_mode():
                ours = Predictor(model, mean, std, tta)(pixels).double()
                reference = evaluator._view_probabilities(model, normalized, tta, torch.device('cpu'), 16)
            self.assertTrue(torch.allclose(ours, reference, atol=1e-5), tta)

    def test_multihead_predictor_matches_inference_per_head(self):
        from multihead_inference import head_probabilities
        torch.manual_seed(1)
        model = Linear().eval()  # three logits, split into heads of 2 and 1
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        rows = {'a': (0, 2), 'b': (2, 3)}
        pixels = torch.rand(1, 3, 4, 4) * 255
        normalized = (pixels / 255 - torch.tensor(mean).reshape(1, 3, 1, 1)) / torch.tensor(std).reshape(1, 3, 1, 1)
        for tta in ('none', 'd4'):
            with torch.inference_mode():
                ours = MultiHeadPredictor(model, mean, std, list(rows.values()), tta)(pixels)
            reference = head_probabilities(model, {'logit_rows': rows}, normalized, tta)
            for output, head in zip(ours, rows):
                self.assertTrue(torch.allclose(output, reference[head], atol=1e-5), (tta, head))


if __name__ == '__main__':
    unittest.main()
