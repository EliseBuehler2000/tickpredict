"""v2 multi-head label handling and loss, on CPU tensors only."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from multihead_experiment import HEADS, head_rows, inat_life_stage, set_loss


class MultiHeadTests(unittest.TestCase):
    def test_rows_cover_every_class_once(self):
        rows, total = head_rows()
        self.assertEqual(total, sum(len(c) for c in HEADS.values()))
        spans = sorted(rows.values())
        self.assertEqual(spans[0][0], 0)
        for (_, end), (start, _) in zip(spans, spans[1:]):
            self.assertEqual(end, start)

    def test_inaturalist_life_stage_mapping(self):
        def stage(life_stage, sex=''):
            return inat_life_stage({'life_stage': life_stage, 'sex': sex})
        self.assertEqual(stage('nymph'), ['nymph'])
        self.assertEqual(stage('adult male', 'male'), ['adult male'])
        self.assertEqual(stage('adult'), ['adult female', 'adult male'])
        self.assertEqual(stage('', 'female'), ['adult female'])
        self.assertEqual(stage(''), [])

    def test_set_loss_is_negative_log_of_summed_probability(self):
        logits = torch.tensor([[2.0, 0.5, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        loss, n = set_loss(logits, mask, None)
        p = logits[0].softmax(0)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(float(loss), float(-(p[0] + p[1]).log()), places=5)

    def test_single_label_set_loss_matches_cross_entropy(self):
        logits = torch.randn(5, 3)
        target = torch.tensor([0, 2, 1, 1, 0])
        mask = torch.nn.functional.one_hot(target, 3).float()
        loss, _ = set_loss(logits, mask, None)
        self.assertAlmostEqual(float(loss), float(torch.nn.functional.cross_entropy(logits, target)), places=5)

    def test_unlabeled_head_contributes_nothing(self):
        logits = torch.randn(3, 3, requires_grad=True)
        loss, n = set_loss(logits, torch.zeros(3, 3), None)
        loss.backward()
        self.assertEqual(n, 0)
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.all(logits.grad == 0))


if __name__ == '__main__':
    unittest.main()
