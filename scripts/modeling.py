"""Model definitions and explicit ImageNet initialization for Penn experiments."""

from pathlib import Path
from urllib.parse import urlparse

import torch
from torchvision import models

from common import WEIGHTS, digest
from image_preprocessing import VERSION as FOREGROUND_VERSION


ARCHITECTURES = ('resnet18', 'resnet50', 'efficientnet_v2_s', 'convnext_tiny')
RESNET18_WEIGHTS_SHA256 = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'


def pretrained_weights(architecture):
    """Pin concrete weight versions rather than torchvision's changing DEFAULT."""
    choices = {
        'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
        'resnet50': models.ResNet50_Weights.IMAGENET1K_V2,
        'efficientnet_v2_s': models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
        'convnext_tiny': models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
    }
    if architecture not in choices:
        raise ValueError(f'Unsupported architecture: {architecture!r}')
    return choices[architecture]


def pretrained_weights_name(architecture):
    return str(pretrained_weights(architecture))


def classifier_head(model, architecture):
    pretrained_weights(architecture)  # Validate before inspecting the model.
    return model.fc if architecture.startswith('resnet') else model.classifier


def last_block(model, architecture):
    """Last residual stage, EfficientNet stage + projection, or ConvNeXt stage."""
    pretrained_weights(architecture)
    if architecture.startswith('resnet'):
        return model.layer4
    if architecture == 'efficientnet_v2_s':
        return model.features[-2:]
    return model.features[-1]


def set_trainable(model, architecture, fine_tune):
    if fine_tune not in {'head', 'last_block', 'all'}:
        raise ValueError(f'Unsupported fine_tune mode: {fine_tune!r}')
    model.requires_grad_(fine_tune == 'all')
    if fine_tune == 'last_block':
        last_block(model, architecture).requires_grad_(True)
    classifier_head(model, architecture).requires_grad_(True)


def preprocessing_spec(image_size=224, bottom_crop_fraction=0.04, foreground_crop=None):
    if foreground_crop not in (None, FOREGROUND_VERSION):
        raise ValueError(f'Unsupported foreground crop version: {foreground_crop!r}')
    spec = {
        'exif_transpose': True,
        'image_size': int(image_size),
        'bottom_crop_fraction': float(bottom_crop_fraction),
        'resize': 'letterbox',
        'resampling': 'lanczos',
        'pad_color': [240, 240, 240],
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    }
    if foreground_crop is not None:
        spec['foreground_crop'] = foreground_crop
    return spec


def build_model(architecture, num_classes, pretrained=False):
    """Build a task head; inference passes pretrained=False and needs no download.

    The optional local ResNet18 weights are the same ImageNet V1 checkpoint
    used by the baseline. Other weights use the pinned torchvision enum.
    Initialization details are attached as ``model.initial_weights``.
    """
    weights = pretrained_weights(architecture)
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError('num_classes must be an integer of at least two.')
    constructor = getattr(models, architecture)
    local = architecture == 'resnet18' and pretrained and WEIGHTS.is_file()
    source = None
    weight_hash = None
    if local:
        weight_hash = digest(WEIGHTS)
        if weight_hash != RESNET18_WEIGHTS_SHA256:
            raise ValueError(f'Unexpected ResNet18 ImageNet weights: {WEIGHTS}')
        model = constructor(weights=None)
        model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
        source = str(WEIGHTS)
    else:
        model = constructor(weights=weights if pretrained else None)
        if pretrained:
            source = weights.url
            cached = Path(torch.hub.get_dir()) / 'checkpoints' / Path(urlparse(weights.url).path).name
            if cached.is_file():
                weight_hash = digest(cached)
    if architecture.startswith('resnet'):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        original = model.classifier[-1]
        model.classifier[-1] = torch.nn.Linear(original.in_features, num_classes)
    model.initial_weights = {
        'pretrained': bool(pretrained),
        'enum': str(weights) if pretrained else None,
        'source': source,
        'sha256': weight_hash,
    }
    return model
