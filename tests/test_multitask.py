"""Synthetic CPU checks for shared-head export and pixel-only joint datasets."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from image_preprocessing import VERSION
from inference import preprocess_image
from modeling import ARCHITECTURES, build_model, preprocessing_spec
from multitask_experiment import (EXPECTED_CLASSES, JointImages, TASK_ROWS,
                                  export_state_dicts, multitask_config, split_logits)


class MultitaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_separate_exports_match_both_joint_heads_for_every_architecture(self):
        images = torch.randn(1, 3, 64, 64)
        with patch('torch.hub.download_url_to_file',
                   side_effect=AssertionError('No weight downloads in tests.')):
            for architecture in ARCHITECTURES:
                with self.subTest(architecture=architecture):
                    joint = build_model(architecture, 5, pretrained=False).eval()
                    with torch.inference_mode():
                        expected = split_logits(joint(images))
                    states = export_state_dicts(joint, architecture)
                    del joint
                    for task, (start, end) in TASK_ROWS.items():
                        single = build_model(architecture, end - start, pretrained=False).eval()
                        single.load_state_dict(states[task], strict=True)
                        with torch.inference_mode():
                            actual = single(images)
                        self.assertEqual(tuple(actual.shape), (1, end - start))
                        torch.testing.assert_close(actual, expected[task], rtol=1e-6, atol=1e-6)
                        del single
                    del states

    def test_joint_labels_do_not_change_input_pixels_or_crop_preprocessing(self):
        classes = {task: sorted(labels) for task, labels in EXPECTED_CLASSES.items()}
        cfg = multitask_config({'image_size': 64, 'foreground_crop': VERSION})
        spec = preprocessing_spec(cfg['image_size'], cfg['bottom_crop_fraction'], cfg['foreground_crop'])
        with tempfile.TemporaryDirectory(prefix='tickpredict-joint-test-') as directory:
            path = Path(directory) / 'synthetic.png'
            image = Image.new('RGB', (160, 140), (240, 235, 225))
            ImageDraw.Draw(image).ellipse((65, 50, 95, 90), fill=(50, 30, 20))
            image.save(path)
            records = [
                {'source_path': str(path), 'life_stage': 'adult female', 'engorgement_level': 'unengorged'},
                {'source_path': str(path), 'life_stage': 'nymph', 'engorgement_level': 'semi-engorged'},
            ]
            with patch('experiment.resolve_image_path', side_effect=Path):
                dataset = JointImages(records, classes, spec, augmentation=None)
            first, life_first, eng_first = dataset[0]
            second, life_second, eng_second = dataset[1]
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            torch.testing.assert_close(first, preprocess_image(path, spec), rtol=0, atol=0)
            self.assertEqual(classes['life_stage'][life_first], 'adult female')
            self.assertEqual(classes['life_stage'][life_second], 'nymph')
            self.assertEqual(classes['engorgement_level'][eng_first], 'unengorged')
            self.assertEqual(classes['engorgement_level'][eng_second], 'semi-engorged')


if __name__ == '__main__':
    unittest.main()
