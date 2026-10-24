"""Synthetic CPU tests for the frozen crop and its legacy preprocessing path."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import image_preprocessing as preprocessing


def centered_shape():
    image = Image.new('RGB', (400, 300), (240, 235, 225))
    drawing = ImageDraw.Draw(image)
    drawing.ellipse((170, 115, 230, 185), fill=(65, 40, 25))
    for endpoint in ((130, 95), (270, 95), (125, 150), (275, 150), (130, 205), (270, 205)):
        drawing.line(((200, 150), endpoint), fill=(50, 35, 20), width=5)
    return image


class FrozenForegroundTests(unittest.TestCase):
    def test_uniform_images_fall_back_to_the_full_frame(self):
        for color in ((240, 240, 240), (0, 0, 0), (60, 100, 150)):
            with self.subTest(color=color):
                image = Image.new('RGB', (400, 300), color)
                box, audit = preprocessing.propose(image)
                self.assertEqual(box, (0, 0, 400, 300))
                self.assertEqual(audit['reason'], 'no_foreground')
                self.assertEqual(audit['area_fraction'], 1.0)

    def test_centered_shape_crop_is_bounded_and_includes_all_limbs(self):
        image = centered_shape()
        box, audit = preprocessing.propose(image)
        self.assertEqual(audit['reason'], 'crop')
        x0, y0, x1, y1 = box
        self.assertTrue(0 <= x0 < x1 <= image.width)
        self.assertTrue(0 <= y0 < y1 <= image.height)
        # Every deliberately drawn foreground pixel, including thin limbs, stays inside.
        pixels = np.asarray(image)
        foreground = np.any(pixels != (240, 235, 225), axis=2)
        y, x = np.where(foreground)
        self.assertLessEqual(x0, int(x.min()))
        self.assertLessEqual(y0, int(y.min()))
        self.assertGreater(x1, int(x.max()))
        self.assertGreater(y1, int(y.max()))
        self.assertLess(audit['area_fraction'], 1.0)

    def test_uncertain_edge_object_keeps_the_full_frame(self):
        image = Image.new('RGB', (400, 300), (240, 235, 225))
        ImageDraw.Draw(image).ellipse((2, 100, 70, 175), fill=(65, 40, 25))
        box, audit = preprocessing.propose(image)
        self.assertEqual(audit['reason'], 'off_center_uncertain')
        self.assertEqual(box, (0, 0, 400, 300))
        self.assertEqual(audit['area_fraction'], 1.0)

    def test_repeated_proposals_are_deterministic_and_do_not_mutate_pixels(self):
        image = centered_shape()
        original = image.tobytes()
        expected = preprocessing.propose(image)
        for _ in range(3):
            self.assertEqual(preprocessing.propose(image), expected)
        self.assertEqual(image.tobytes(), original)

    def test_without_foreground_key_matches_original_pixels_with_exif_and_rgba(self):
        with tempfile.TemporaryDirectory(prefix='tickpredict-foreground-') as directory:
            path = Path(directory) / 'synthetic.png'
            image = centered_shape().convert('RGBA')
            image.putalpha(Image.linear_gradient('L').resize(image.size))
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
            with Image.open(path) as source:
                legacy = ImageOps.exif_transpose(source).convert('RGB')
            legacy = legacy.crop((0, 0, legacy.width, round(legacy.height * 0.96)))
            for size in (224, 320):
                with self.subTest(image_size=size):
                    expected = ImageOps.pad(legacy, (size, size),
                                            method=Image.Resampling.LANCZOS,
                                            color=(240, 240, 240))
                    actual = preprocessing.prepare_image(path, {
                        'image_size': size, 'bottom_crop_fraction': 0.04,
                        'pad_color': [240, 240, 240],
                    })
                    self.assertEqual(actual.mode, 'RGB')
                    self.assertEqual(actual.size, expected.size)
                    self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_foreground_receives_oriented_bottom_masked_image_before_padding(self):
        with tempfile.TemporaryDirectory(prefix='tickpredict-foreground-') as directory:
            path = Path(directory) / 'synthetic.png'
            image = centered_shape()
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
            with Image.open(path) as source:
                expected_input = ImageOps.exif_transpose(source).convert('RGB')
            expected_input = expected_input.crop(
                (0, 0, expected_input.width, round(expected_input.height * 0.96)))
            box, _ = preprocessing.propose(expected_input)
            expected = ImageOps.pad(expected_input.crop(box), (320, 320),
                                    method=Image.Resampling.LANCZOS, color=(240, 240, 240))
            with patch.object(preprocessing, 'propose', wraps=preprocessing.propose) as propose:
                actual = preprocessing.prepare_image(path, {
                    'image_size': 320, 'bottom_crop_fraction': 0.04,
                    'pad_color': [240, 240, 240], 'foreground_crop': preprocessing.VERSION,
                })
            propose.assert_called_once()
            supplied_image = propose.call_args.args[0]
            self.assertEqual(supplied_image.size, expected_input.size)
            self.assertEqual(supplied_image.tobytes(), expected_input.tobytes())
            self.assertEqual(actual.tobytes(), expected.tobytes())


if __name__ == '__main__':
    unittest.main()
