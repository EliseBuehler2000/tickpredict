"""Shared inference for saved Penn classifiers and their recorded preprocessing."""

from pathlib import Path

import torch
import torchvision
from image_preprocessing import prepare_image
from modeling import ARCHITECTURES, build_model, preprocessing_spec


EXPECTED_CLASSES = {
    'life_stage': {'adult female', 'nymph'},
    'engorgement_level': {'fully engorged', 'semi-engorged', 'unengorged'},
}
NOTES = {
    'life_stage': 'Only adult female versus nymph. Confidence is an uncalibrated '
                  'model score. No unknown-image rejection; phone-photo '
                  'performance is not validated. Scores are not Lyme risk.',
    'engorgement_level': 'Confidence is an uncalibrated model score, not '
                        'percentage of blood consumed, attachment hours, or '
                        'Lyme risk. No unknown-image rejection; phone-photo '
                        'performance is not validated.',
}


def preprocess_image(image_path, preprocessing=None):
    """Match the saved models' Penn-specific validation preprocessing."""
    spec = preprocessing_spec() if preprocessing is None else preprocessing
    if not isinstance(spec, dict):
        raise ValueError('Checkpoint preprocessing must be a dictionary.')
    size = spec.get('image_size')
    crop = spec.get('bottom_crop_fraction')
    if type(size) is not int or size < 32 or not isinstance(crop, (int, float)) or not 0 <= crop < 0.5:
        raise ValueError('Unsupported checkpoint image size or crop fraction.')
    if spec != preprocessing_spec(size, crop, spec.get('foreground_crop')):
        raise ValueError('Unsupported preprocessing recipe; use the recorded Penn letterbox pipeline.')
    image = prepare_image(image_path, spec)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(spec['mean'], spec['std']),
    ])
    return transform(image)


def predict(image_path, checkpoint, expected_task):
    """Return a task prediction and its uncalibrated softmax confidence."""
    if expected_task not in EXPECTED_CLASSES:
        raise ValueError(f'Unsupported task: {expected_task!r}')
    checkpoint = Path(checkpoint).expanduser().resolve()
    torch.set_num_threads(2)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(saved, dict) or saved.get('architecture') not in ARCHITECTURES:
        raise ValueError(f'A saved checkpoint for one of {ARCHITECTURES} is required.')
    if saved.get('smoke_only'):
        raise ValueError('A smoke-test checkpoint is not a usable prediction model.')
    if saved.get('target') != expected_task:
        raise ValueError(
            f'Expected a {expected_task!r} checkpoint; '
            f'found target {saved.get("target")!r}.'
        )
    classes = saved.get('classes')
    expected_classes = EXPECTED_CLASSES[expected_task]
    if (not isinstance(classes, (list, tuple))
            or not all(isinstance(name, str) for name in classes)
            or len(classes) != len(expected_classes)
            or set(classes) != expected_classes):
        raise ValueError(f'Unexpected class labels for {expected_task!r}: {classes!r}')

    model = build_model(saved['architecture'], len(classes), pretrained=False)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    with torch.inference_mode():
        scores = model(preprocess_image(image_path, saved.get('preprocessing')).unsqueeze(0)).softmax(dim=1)[0]
    if not torch.isfinite(scores).all():
        raise ValueError('The model returned non-finite scores.')
    predicted_index = int(scores.argmax())
    return {
        'task': expected_task,
        'prediction': classes[predicted_index],
        'confidence': float(scores[predicted_index]),
        'confidence_calibrated': False,
        'model_scores_uncalibrated': {
            name: float(score) for name, score in zip(classes, scores)
        },
        'checkpoint': str(checkpoint),
        'architecture': saved['architecture'],
        'note': NOTES[expected_task],
    }
