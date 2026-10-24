"""CPU-only regressions for corrupt probability transfers and aggregates."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_predictor as evaluator


class TransferProbe:
    """An accelerator-like result whose combined transfer/cast must not run."""

    def __init__(self, values, calls):
        self.values, self.calls = values, calls

    def softmax(self, dim):
        assert dim == 1
        self.calls.append("softmax")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self.values

    def to(self, *args, **kwargs):
        raise AssertionError("Transfer to CPU before changing dtype")


class EmptyClassifier(torch.nn.Module):
    def forward(self, images):
        return torch.zeros((len(images), 2))


class ProbabilityValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.cpu = torch.device("cpu")
        self.batch = torch.zeros((2, 3, 4, 4))
        self.classes = ["adult female", "nymph"]
        self.records = [
            {"tick_id": "synthetic-a", "life_stage": "adult female", "engorgement_level": "unengorged"},
            {"tick_id": "synthetic-b", "life_stage": "nymph", "engorgement_level": "semi-engorged"},
        ]

    def test_cpu_transfer_precedes_float64_cast(self):
        calls = []
        values = torch.tensor([[0.25, 0.75], [0.8, 0.2]])
        output = evaluator._view_probabilities(
            lambda images: TransferProbe(values, calls), self.batch, "none", self.cpu, 8)
        self.assertEqual(calls, ["softmax", "cpu"])
        self.assertEqual(output.dtype, torch.float64)
        torch.testing.assert_close(output, values.double(), rtol=0, atol=0)

    def test_zero_softmax_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            evaluator._view_probabilities(
                lambda images: TransferProbe(torch.zeros((len(images), 2)), []),
                self.batch, "none", self.cpu, 8)

    def test_corrupt_view_average_is_rejected(self):
        with patch.object(torch.Tensor, "mean", return_value=torch.zeros((2, 2), dtype=torch.float64)):
            with self.assertRaisesRegex(ValueError, "View average.*sum to one"):
                evaluator._view_probabilities(EmptyClassifier(), self.batch, "d4", self.cpu, 8)

    def test_metrics_reject_invalid_probability_matrices(self):
        invalid = [
            np.zeros((2, 2)),
            np.array([[-0.1, 1.1], [0.5, 0.5]]),
            np.array([[np.nan, 1.0], [0.5, 0.5]]),
            np.array([[np.inf, 0.0], [0.5, 0.5]]),
            np.array([[0.2, 0.2], [0.5, 0.5]]),
            np.full((2, 3), 1 / 3),
            np.full((1, 2), 0.5),
            np.array([0.5, 0.5]),
        ]
        for probabilities in invalid:
            with self.subTest(probabilities=probabilities.tolist()):
                with self.assertRaises(ValueError):
                    evaluator._details(self.records, probabilities, self.classes, "life_stage")

    def test_valid_extreme_probabilities_are_preserved(self):
        result = evaluator._details(self.records, np.eye(2), self.classes, "life_stage")
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["nll"], 0)
        self.assertEqual(result["mean_confidence"], 1)

    def test_model_aggregation_rejects_corrupt_view_output(self):
        spec = {"target": "life_stage", "checkpoints": ["synthetic.pt"], "tta": "d4"}
        saved = {"state_dict": {}}
        info = {"classes": self.classes, "architecture": "resnet18", "preprocessing": {}}
        with patch.object(evaluator, "_checkpoint", return_value=(saved, info)), \
                patch.object(evaluator, "build_model", return_value=EmptyClassifier()), \
                patch.object(evaluator, "preprocess_image", return_value=self.batch[0]), \
                patch.object(evaluator, "_view_probabilities", return_value=torch.zeros((2, 2))):
            with self.assertRaisesRegex(ValueError, "sum to one"):
                evaluator._predict_paths(["synthetic-a", "synthetic-b"], spec, self.cpu, 8)

    def test_single_image_rejects_corrupt_final_output(self):
        spec = {"target": "life_stage", "checkpoints": ["synthetic.pt"], "tta": "d4"}
        with patch.object(evaluator, "load_spec", return_value=spec), \
                patch.object(evaluator, "_predict_paths", return_value=(np.zeros((1, 2)), [])):
            with self.assertRaisesRegex(ValueError, "Single-image prediction.*sum to one"):
                evaluator.predict_image("must-not-open.png", spec, device="cpu")

    def test_invalid_evaluation_fails_without_writing_metrics(self):
        spec = {"target": "life_stage", "checkpoints": ["synthetic.pt"], "tta": "d4"}
        preflight = {"source_sha256": "synthetic-source", "manifest_sha256": "synthetic-manifest"}
        records = [dict(row, source_path="must-not-open.png") for row in self.records]
        verified = ([], records, self.classes, preflight["source_sha256"], preflight["manifest_sha256"])
        with tempfile.TemporaryDirectory(prefix="tick-invalid-probability-") as directory:
            with patch.object(evaluator, "WORK", Path(directory)), \
                    patch.object(evaluator, "load_spec", return_value=spec), \
                    patch.object(evaluator, "_records", return_value=(records, self.classes, preflight)), \
                    patch.object(evaluator, "_source_metadata", return_value={}), \
                    patch.object(evaluator, "verified_records", return_value=verified), \
                    patch.object(evaluator, "_predict_paths", return_value=(np.zeros((2, 2)), [])):
                with self.assertRaisesRegex(ValueError, "Evaluation metrics.*sum to one"):
                    evaluator.evaluate_spec(spec, execute=True, device="cpu")
            outputs = list((Path(directory) / "experiments/evaluations").iterdir())
            self.assertEqual(len(outputs), 1)
            completion = json.loads((outputs[0] / "completion.json").read_text())
            self.assertEqual(completion["status"], "failed")
            self.assertFalse((outputs[0] / "validation.json").exists())
            self.assertFalse((outputs[0] / "provenance.json").exists())


    def test_dino_linear_scores_transfer_each_input_before_cast(self):
        import dino_probe
        calls = []

        class CpuFirst:
            def __init__(self, name, value):
                self.name, self.value = name, value

            def cpu(self):
                calls.append(self.name)
                return self.value

            def to(self, *args, **kwargs):
                raise AssertionError("Transfer to CPU before changing dtype")

            def double(self):
                raise AssertionError("Float64 conversion must occur on CPU")

        features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        weight = torch.tensor([[0.5, -0.25], [1.0, 2.0]])
        bias = torch.tensor([0.2, -0.1])
        result = dino_probe.raw_scores(CpuFirst("features", features),
                                      CpuFirst("weight", weight), CpuFirst("bias", bias))
        expected = torch.nn.functional.linear(features.double(), weight.double(), bias.double())
        self.assertEqual(calls, ["features", "weight", "bias"])
        self.assertEqual(result.dtype, torch.float64)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
