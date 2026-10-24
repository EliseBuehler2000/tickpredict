"""Load a v2 multi-head checkpoint and predict species, life stage and engorgement.

Uses the checkpoint's recorded preprocessing (the same letterbox as v1 and
the iOS app). `tta="d4"` averages softmax scores over the eight fixed
rotations/reflections used by the v1 engorgement predictor. Scores are
uncalibrated.

    .venv/bin/python scripts/multihead_inference.py CHECKPOINT IMAGE [--tta d4]
"""

import argparse
import json
from pathlib import Path

import torch
import torchvision

from evaluate_predictor import d4_views
from image_preprocessing import prepare_image
from modeling import ARCHITECTURES, build_model


def load_multihead(checkpoint, device='cpu'):
    saved = torch.load(Path(checkpoint).expanduser(), map_location='cpu', weights_only=True)
    if saved.get('target') != 'multihead' or saved.get('architecture') not in ARCHITECTURES:
        raise ValueError('Not a v2 multi-head checkpoint.')
    rows = {head: tuple(span) for head, span in saved['logit_rows'].items()}
    total = max(end for _, end in rows.values())
    for head, (start, end) in rows.items():
        if end - start != len(saved['heads'][head]):
            raise ValueError(f'Logit rows and classes disagree for {head}.')
    model = build_model(saved['architecture'], total, pretrained=False)
    model.load_state_dict(saved['state_dict'])
    return model.eval().to(device), saved


def image_tensor(image_path, preprocessing):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(preprocessing['mean'], preprocessing['std']),
    ])
    return transform(prepare_image(image_path, preprocessing))


def head_probabilities(model, saved, batch, tta='none'):
    """Per-head softmax for a normalized NCHW batch, averaged over views per head."""
    views = d4_views(batch) if tta == 'd4' else [batch]
    with torch.inference_mode():
        logits = torch.stack([model(view).float() for view in views])
    return {head: logits[..., start:end].softmax(-1).mean(0)
            for head, (start, end) in ((h, tuple(s)) for h, s in saved['logit_rows'].items())}


def predict(checkpoint, image_path, tta='none', device='cpu'):
    model, saved = load_multihead(checkpoint, device)
    batch = image_tensor(image_path, saved['preprocessing']).unsqueeze(0).to(device)
    probabilities = head_probabilities(model, saved, batch, tta)
    result = {}
    for head, classes in saved['heads'].items():
        scores = probabilities[head][0].cpu().tolist()
        best = max(range(len(classes)), key=scores.__getitem__)
        result[head] = {'prediction': classes[best], 'confidence': scores[best],
                        'scores_uncalibrated': dict(zip(classes, scores))}
    return {'checkpoint': str(checkpoint), 'tta': tta, 'epoch': saved['epoch'],
            'smoke_only': saved['smoke_only'], 'confidence_calibrated': False, **result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint')
    parser.add_argument('image')
    parser.add_argument('--tta', choices=('none', 'd4'), default='none')
    parser.add_argument('--device', default='cpu')
    arguments = parser.parse_args()
    print(json.dumps(predict(arguments.checkpoint, arguments.image, arguments.tta, arguments.device), indent=2))
