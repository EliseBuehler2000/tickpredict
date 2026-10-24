"""Synthetic CPU checks for the training-only joint sampler.

No source photos, model weights, development images or test images are used.
"""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment import CachedImages, build_train_sampling, load_config
from modeling import preprocessing_spec
from multitask_experiment import multitask_config


def synthetic_records(counts):
    records = []
    for (stage, engorgement), count in counts.items():
        for _ in range(count):
            index = len(records)
            records.append({
                'tick_id': f'tick-{index}', 'similarity_group': f'group-{index}',
                'provisional_split': 'train', 'life_stage': stage,
                'engorgement_level': engorgement,
            })
    return records


def joint_config(**overrides):
    return load_config({
        'target': 'engorgement_level', 'train_sampling': 'joint_sqrt_inverse',
        'class_weighting': 'none', 'seed': 20261019, **overrides,
    })


class SamplingTests(unittest.TestCase):
    def test_uniform_default_has_no_sampler_or_randomness_side_effect(self):
        cfg = load_config({'target': 'engorgement_level', 'class_weighting': 'sqrt_inverse'})
        records = synthetic_records({('nymph', 'semi-engorged'): 4})
        before = torch.random.get_rng_state().clone()
        sampler, plan = build_train_sampling(records, cfg)
        self.assertEqual(cfg['train_sampling'], 'uniform')
        self.assertIsNone(sampler)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertFalse(plan['replacement'])
        self.assertEqual(plan['num_samples_per_epoch'], 4)
        self.assertEqual(plan['stage_engorgement_groups'][0]['expected_draws_per_epoch'], 4)

    def test_joint_policy_rejects_stacked_weights_and_other_targets(self):
        for weighting in ['inverse', 'sqrt_inverse']:
            with self.subTest(weighting=weighting):
                with self.assertRaisesRegex(ValueError, 'class_weighting=none'):
                    joint_config(class_weighting=weighting)
        with self.assertRaisesRegex(ValueError, 'restricted to engorgement_level'):
            joint_config(target='life_stage')
        with self.assertRaisesRegex(ValueError, 'train_sampling must be'):
            joint_config(train_sampling='unknown')

    def test_multitask_cannot_silently_ignore_joint_sampling(self):
        self.assertEqual(multitask_config({})['train_sampling'], 'uniform')
        with self.assertRaisesRegex(ValueError, 'train_sampling=uniform only'):
            multitask_config({'train_sampling': 'joint_sqrt_inverse', 'class_weighting': 'none'})

    def test_rejects_nontraining_missing_split_empty_or_duplicate_ids(self):
        records = synthetic_records({('nymph', 'semi-engorged'): 2})
        for split in ['dev', 'test', '', None]:
            bad = [dict(record) for record in records]
            bad[0]['provisional_split'] = split
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, 'train records only'):
                    build_train_sampling(bad, joint_config())
        bad = [dict(record) for record in records]
        del bad[0]['provisional_split']
        with self.assertRaisesRegex(ValueError, 'train records only'):
            build_train_sampling(bad, joint_config())
        with self.assertRaisesRegex(ValueError, 'nonempty train records'):
            build_train_sampling([], joint_config())
        bad = [dict(record) for record in records]
        bad[1]['tick_id'] = bad[0]['tick_id']
        with self.assertRaisesRegex(ValueError, 'unique tick IDs'):
            build_train_sampling(bad, joint_config())

    def test_joint_cell_probabilities_are_between_uniform_rows_and_uniform_cells(self):
        # Five synthetic rows: cell sampling should be 1/3 versus 2/3,
        # rather than 1/5 versus 4/5 (uniform rows) or 1/2 versus 1/2.
        records = synthetic_records({
            ('adult female', 'semi-engorged'): 1,
            ('nymph', 'semi-engorged'): 4,
        })
        sampler, plan = build_train_sampling(records, joint_config())
        self.assertEqual(sampler.weights.device.type, 'cpu')
        self.assertEqual(sampler.weights.dtype, torch.double)
        self.assertTrue(sampler.replacement)
        self.assertEqual(sampler.num_samples, 5)
        probabilities = sampler.weights / sampler.weights.sum()
        self.assertAlmostEqual(float(probabilities[0]), 1 / 3)
        self.assertAlmostEqual(float(probabilities[1:].sum()), 2 / 3)
        groups = plan['stage_engorgement_groups']
        self.assertAlmostEqual(groups[0]['expected_draws_per_epoch'], 5 / 3)
        self.assertAlmostEqual(groups[1]['expected_draws_per_epoch'], 10 / 3)
        self.assertAlmostEqual(sum(group['expected_draws_per_epoch'] for group in groups), 5)

    def test_full_size_draws_are_seeded_but_progress_across_epochs(self):
        # Synthetic labels matching the approved training cell counts only.
        records = synthetic_records({
            ('adult female', 'fully engorged'): 13,
            ('adult female', 'semi-engorged'): 7,
            ('adult female', 'unengorged'): 36,
            ('nymph', 'fully engorged'): 9,
            ('nymph', 'semi-engorged'): 155,
            ('nymph', 'unengorged'): 21,
        })
        before = torch.random.get_rng_state().clone()
        first, plan = build_train_sampling(records, joint_config())
        second, _ = build_train_sampling(records, joint_config())
        first_epoch, second_epoch = list(first), list(first)
        self.assertEqual(first_epoch, list(second))
        self.assertEqual(second_epoch, list(second))
        self.assertNotEqual(first_epoch, second_epoch)
        self.assertEqual(len(first_epoch), 241)
        self.assertTrue(all(0 <= index < 241 for index in first_epoch))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        adult_semi = next(group for group in plan['stage_engorgement_groups']
                          if group['life_stage'] == 'adult female'
                          and group['engorgement_level'] == 'semi-engorged')
        self.assertAlmostEqual(adult_semi['expected_draws_per_epoch'], 19.750657, places=4)
        self.assertAlmostEqual(adult_semi['expected_draws_per_image_per_epoch'], 2.821522, places=4)
        self.assertEqual(plan['similarity_groups']['unique_count'], 241)
        self.assertEqual(plan['similarity_groups']['max_multiplicity'], 1)

    def test_repeated_similarity_groups_are_reported_without_silent_reweighting(self):
        records = synthetic_records({
            ('adult female', 'semi-engorged'): 1,
            ('nymph', 'semi-engorged'): 4,
        })
        original, _ = build_train_sampling(records, joint_config())
        records[1]['similarity_group'] = records[0]['similarity_group']
        sampler, plan = build_train_sampling(records, joint_config())
        torch.testing.assert_close(original.weights, sampler.weights, rtol=0, atol=0)
        groups = plan['similarity_groups']
        self.assertEqual(groups['unique_count'], 4)
        self.assertEqual(groups['max_multiplicity'], 2)
        self.assertEqual(groups['multiplicity_histogram'], {'1': 3, '2': 1})
        self.assertEqual(groups['repeated_groups'], [{'similarity_group': 'group-0', 'count': 2}])
        self.assertEqual(groups['duplicate_group_correction'], 'none')

    def test_stage_metadata_does_not_enter_input_pixels_or_dataset_output(self):
        with tempfile.TemporaryDirectory(prefix='tick-sampling-test-') as directory:
            path = Path(directory) / 'synthetic.png'
            Image.new('RGB', (90, 80), (100, 70, 30)).save(path)
            records = synthetic_records({
                ('adult female', 'semi-engorged'): 1,
                ('nymph', 'semi-engorged'): 1,
            })
            for record in records:
                record['source_path'] = str(path)
            with patch('experiment.resolve_image_path', side_effect=Path):
                dataset = CachedImages(records, ['semi-engorged'], 'engorgement_level',
                                       preprocessing_spec(64, 0.04), augmentation=None)
            self.assertEqual(len(dataset[0]), 2)
            first_image, first_label = dataset[0]
            second_image, second_label = dataset[1]
            torch.testing.assert_close(first_image, second_image, rtol=0, atol=0)
            self.assertEqual(first_label, second_label)


if __name__ == '__main__':
    unittest.main()
