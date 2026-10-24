"""CPU-only integration checks using synthetic images and temporary run folders.

Run with: .venv/bin/python -m unittest discover -s tests -v
No dataset files, pretrained downloads, or accelerator training are needed.
"""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageOps
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import experiment
import evaluate_predictor
import inference
from image_preprocessing import VERSION as FOREGROUND_VERSION
from modeling import ARCHITECTURES, build_model, preprocessing_spec
import run_experiments


def synthetic_image(path):
    """An asymmetric image makes orientation, cropping and padding observable."""
    image = Image.new('RGB', (93, 127), (230, 225, 210))
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((2, 5, 45, 65), fill=(30, 80, 160))
    drawing.ellipse((35, 35, 88, 115), fill=(160, 55, 20))
    drawing.rectangle((0, 118, 92, 126), fill=(255, 0, 255))
    exif = Image.Exif()
    exif[274] = 6  # Rotated camera image; preprocessing must honor EXIF.
    image.save(path, exif=exif)


def legacy_tensor(path):
    """The original pinned models' fixed 224-pixel preprocessing contract."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    image = image.crop((0, 0, image.width, round(image.height * 0.96)))
    image = ImageOps.pad(image, (224, 224), method=Image.Resampling.LANCZOS,
                         color=(240, 240, 240))
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225]),
    ])(image)


class ModelPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='tickpredict-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = self.root / 'synthetic.png'
        synthetic_image(self.image)
        # A future regression must fail rather than make a network request.
        download = patch('torch.hub.download_url_to_file',
                         side_effect=AssertionError('Tests must never download weights.'))
        download.start()
        self.addCleanup(download.stop)

    def test_default_preprocessing_exactly_matches_legacy(self):
        expected = legacy_tensor(self.image)
        self.assertEqual(tuple(expected.shape), (3, 224, 224))
        torch.testing.assert_close(inference.preprocess_image(self.image), expected,
                                   rtol=0, atol=0)
        torch.testing.assert_close(
            inference.preprocess_image(self.image, preprocessing_spec()), expected,
            rtol=0, atol=0)
        self.assertNotIn('foreground_crop', preprocessing_spec())

    def test_cached_validation_images_match_inference_at_both_resolutions(self):
        rgba_path = self.root / 'synthetic-rgba.png'
        with Image.open(self.image) as source:
            rgba = source.convert('RGBA')
            rgba.putalpha(Image.linear_gradient('L').resize(rgba.size))
            rgba.save(rgba_path, exif=source.getexif())
        classes = ['adult female', 'nymph']
        # Route the cache explicitly to temporary fixtures, never the real dataset.
        with patch('experiment.resolve_image_path', side_effect=Path):
            for path in (self.image, rgba_path):
                for size, crop, foreground in ((224, 0.04, None), (320, 0.08, None),
                                               (320, 0.04, FOREGROUND_VERSION)):
                    with self.subTest(mode=path.stem, image_size=size, foreground_crop=foreground):
                        spec = preprocessing_spec(size, crop, foreground)
                        cached = experiment.CachedImages(
                            [{'source_path': str(path), 'life_stage': 'nymph',
                              'tick_id': 'synthetic'}], classes, 'life_stage', spec,
                            augmentation=None)
                        actual, label = cached[0]
                        self.assertEqual(label, 1)
                        self.assertEqual(tuple(actual.shape), (3, size, size))
                        torch.testing.assert_close(
                            actual, inference.preprocess_image(path, spec), rtol=0, atol=0)

    def test_legacy_checkpoint_without_preprocessing_still_predicts(self):
        classes = sorted(inference.EXPECTED_CLASSES['life_stage'])
        model = build_model('resnet18', len(classes), pretrained=False).eval()
        with torch.inference_mode():
            expected = model(legacy_tensor(self.image).unsqueeze(0)).softmax(1)[0]
        checkpoint = self.root / 'legacy.pt'
        torch.save({'architecture': 'resnet18', 'classes': classes,
                    'target': 'life_stage', 'state_dict': model.state_dict(), 'epoch': 1},
                   checkpoint)
        del model
        actual = inference.predict(self.image, checkpoint, 'life_stage')
        torch.testing.assert_close(
            torch.tensor(list(actual['model_scores_uncalibrated'].values())), expected,
            rtol=1e-6, atol=1e-7)

    def test_all_architectures_roundtrip_both_task_heads(self):
        # Non-default size/crop checks that inference uses checkpoint metadata.
        spec = preprocessing_spec(image_size=64, bottom_crop_fraction=0.08,
                                   foreground_crop=FOREGROUND_VERSION)
        tensor = inference.preprocess_image(self.image, spec).unsqueeze(0)
        checkpoint = self.root / 'roundtrip.pt'
        for architecture in ARCHITECTURES:
            for task, labels in inference.EXPECTED_CLASSES.items():
                with self.subTest(architecture=architecture, task=task):
                    classes = sorted(labels)
                    model = build_model(architecture, len(classes), pretrained=False).eval()
                    self.assertFalse(model.initial_weights['pretrained'])
                    self.assertEqual(next(model.parameters()).device.type, 'cpu')
                    with torch.inference_mode():
                        logits = model(tensor)
                        self.assertEqual(tuple(logits.shape), (1, len(classes)))
                        expected = logits.softmax(1)[0]
                    torch.save({
                        'architecture': architecture, 'classes': classes, 'target': task,
                        'state_dict': model.state_dict(), 'epoch': 1,
                        'preprocessing': spec, 'image_size': 64,
                        'bottom_crop_fraction': 0.08, 'smoke_only': False,
                    }, checkpoint)
                    del model
                    actual = inference.predict(self.image, checkpoint, task)
                    scores = actual['model_scores_uncalibrated']
                    self.assertEqual(list(scores), classes)
                    torch.testing.assert_close(torch.tensor(list(scores.values())), expected,
                                               rtol=1e-6, atol=1e-7)
                    self.assertEqual(actual['prediction'], classes[int(expected.argmax())])
                    self.assertAlmostEqual(actual['confidence'], float(expected.max()), places=6)
                    self.assertFalse(actual['confidence_calibrated'])
                    self.assertEqual(actual['architecture'], architecture)

    def test_evaluator_accepts_only_the_supported_foreground_version(self):
        spec = preprocessing_spec(64, foreground_crop=FOREGROUND_VERSION)
        classes = sorted(inference.EXPECTED_CLASSES['life_stage'])
        provenance = {'config': {'target': 'life_stage', 'architecture': 'resnet18'},
                      'classes': classes, 'smoke_only': False, 'test_images_used': 0,
                      'preprocessing': spec}
        (self.root / 'completion.json').write_text(json.dumps({
            'status': 'completed', 'smoke_only': False}))
        checkpoint = self.root / 'metadata.pt'
        for version in (FOREGROUND_VERSION, 'unsupported-future-crop'):
            spec['foreground_crop'] = version
            (self.root / 'provenance.json').write_text(json.dumps(provenance))
            torch.save({'target': 'life_stage', 'classes': classes, 'architecture': 'resnet18',
                        'state_dict': {}, 'preprocessing': spec, 'smoke_only': False}, checkpoint)
            if version == FOREGROUND_VERSION:
                _, information = evaluate_predictor._checkpoint(checkpoint, 'life_stage')
                self.assertEqual(information['preprocessing'], spec)
            else:
                with self.assertRaisesRegex(ValueError, 'Unsupported foreground crop version'):
                    evaluate_predictor._checkpoint(checkpoint, 'life_stage')

    def test_wrong_task_and_smoke_rejected_before_model_or_image_loading(self):
        checkpoint = self.root / 'invalid.pt'
        missing_image = self.root / 'must-not-open.png'
        with patch('inference.build_model') as build:
            torch.save({'architecture': 'resnet18', 'target': 'engorgement_level'}, checkpoint)
            with self.assertRaisesRegex(ValueError, 'Expected a'):
                inference.predict(missing_image, checkpoint, 'life_stage')
            torch.save({'architecture': 'resnet18', 'target': 'life_stage',
                        'smoke_only': True}, checkpoint)
            with self.assertRaisesRegex(ValueError, 'smoke-test checkpoint'):
                inference.predict(missing_image, checkpoint, 'life_stage')
            build.assert_not_called()

    def test_runner_resumes_normalized_config_and_skips_incomplete_metadata(self):
        source = self.root / 'prepared_data.csv'
        source.write_text('synthetic source metadata\n')
        inputs = {
            'source_sha256': source,
            'manifest_sha256': self.root / 'data/penn_train_dev.csv',
            'script_sha256': self.root / 'scripts/experiment.py',
            'modeling_sha256': self.root / 'scripts/modeling.py',
            'image_preprocessing_sha256': self.root / 'scripts/image_preprocessing.py',
        }
        for key, path in inputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if path != source:
                path.write_text(f'Synthetic fixture for {key}\n')
        fingerprints = {key: hashlib.sha256(path.read_bytes()).hexdigest()
                        for key, path in inputs.items()}
        plan = {'tasks': ['life_stage'], 'shared': {'seed': 42},
                'recipes': [{'name': 'completed-recipe'}, {'name': 'new-recipe'}]}
        plan_path = self.root / 'plan.json'
        plan_path.write_text(json.dumps(plan))
        complete = self.root / 'experiments/runs/completed'
        complete.mkdir(parents=True)
        (complete / 'completion.json').write_text(json.dumps({
            'status': 'completed', 'smoke_only': False}))
        normalized = experiment.load_config({'target': 'life_stage', 'seed': 42,
                                             'name': 'completed-recipe'})
        (complete / 'provenance.json').write_text(json.dumps({
            **fingerprints, 'config': normalized, 'smoke_only': False}))
        (complete / 'best.pt').write_bytes(b'Existence marker; never loaded by the runner.')
        for name, status in [('unfinished', 'running'), ('failed', 'failed'),
                             ('missing-provenance', 'completed')]:
            folder = self.root / 'experiments/runs' / name
            folder.mkdir()
            (folder / 'completion.json').write_text(json.dumps({
                'status': status, 'smoke_only': False}))
        new_run = self.root / 'experiments/runs/new-result'
        with patch.object(run_experiments, 'WORK', self.root), \
                patch.object(run_experiments, 'SOURCE', source), \
                patch.object(experiment, 'run', return_value=new_run) as train:
            result = run_experiments.run_plan(plan_path, execute=True)
        self.assertEqual(result, [complete, new_run])
        train.assert_called_once_with(
            experiment.load_config({'target': 'life_stage', 'seed': 42, 'name': 'new-recipe'}),
            smoke=False)


if __name__ == '__main__':
    unittest.main()
