"""Reject misleading comparison inputs using temporary metadata only."""

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_runs import BASELINE_RUNS, collect_results


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='tick-comparison-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        records = [
            {'tick_id': str(i), 'provisional_split': 'dev',
             'life_stage': stage, 'engorgement_level': engorgement}
            for i, (stage, engorgement) in enumerate(
                (stage, engorgement)
                for stage in ['adult female', 'nymph']
                for engorgement in ['fully engorged', 'semi-engorged', 'unengorged'])
        ]
        manifest = self.root / 'data/penn_train_dev.csv'
        manifest.parent.mkdir()
        with manifest.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        hashes = {'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  'source_sha256': 'synthetic-source-hash'}
        self.write('notes/preflight.json', {**hashes, 'counts': {'dev': 6}})
        self.validations = {}
        for target, folder in BASELINE_RUNS.items():
            classes = sorted({record[target] for record in records})
            predictions = []
            for record in records:
                probabilities = [0.1 / (len(classes) - 1)] * len(classes)
                probabilities[classes.index(record[target])] = 0.9
                predictions.append({'tick_id': record['tick_id'], 'true': record[target],
                                    'predicted': record[target],
                                    'uncalibrated_probabilities': probabilities})
            self.write(f'{folder}/completion.json', {'status': 'completed', 'smoke_only': False})
            self.write(f'{folder}/provenance.json', {
                **hashes, 'config': {'target': target}, 'classes': classes,
                'smoke_only': False, 'test_images_used': 0, 'validation_n': 6,
                'selection_metric': 'accuracy'})
            self.write(f'{folder}/history.json', [{'epoch': 1, 'validation_accuracy': 1.0}])
            validation = {'classes': classes, 'predictions': predictions, 'epoch': 1}
            self.write(f'{folder}/validation.json', validation)
            self.validations[target] = validation

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_known_probabilities_and_class_metrics(self):
        results = collect_results(self.root)
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(result['correct'], 6)
            self.assertEqual(result['validation_accuracy'], 1.0)
            self.assertEqual(result['validation_balanced_accuracy'], 1.0)
            self.assertAlmostEqual(result['validation_ece10'], 0.1)
            self.assertTrue(result['promotion_eligible'])
            self.assertEqual(result['test_images_used'], 0)

    def test_rejects_duplicate_or_foreign_prediction_ids(self):
        validation = self.validations['life_stage']
        folder = BASELINE_RUNS['life_stage']
        for bad_id in ['1', 'not-a-development-id']:
            with self.subTest(bad_id=bad_id):
                validation['predictions'][0]['tick_id'] = bad_id
                self.write(f'{folder}/validation.json', validation)
                with self.assertRaisesRegex(ValueError, 'Prediction IDs differ'):
                    collect_results(self.root)

    def test_rejects_prediction_that_disagrees_with_probabilities(self):
        validation = self.validations['life_stage']
        validation['predictions'][0]['predicted'] = 'nymph'
        self.write(f"{BASELINE_RUNS['life_stage']}/validation.json", validation)
        with self.assertRaisesRegex(ValueError, 'argmax'):
            collect_results(self.root)

    def test_rejects_changed_manifest(self):
        manifest = self.root / 'data/penn_train_dev.csv'
        with manifest.open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            collect_results(self.root)


if __name__ == '__main__':
    unittest.main()
