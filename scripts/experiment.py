"""Bounded training comparisons on the fixed Penn train/dev split only.

There is deliberately no test loader. Smoke runs validate execution and must
never be interpreted as model performance. The historic training algorithm is unchanged.
"""

import argparse
from collections import Counter
from datetime import datetime
from functools import partial
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision

from common import WORK, SOURCE, digest, resolve_image_path, rows, write_json
from image_preprocessing import VERSION as FOREGROUND_VERSION, prepare_image
from modeling import (ARCHITECTURES, build_model, classifier_head,
                      preprocessing_spec, set_trainable)


DEFAULTS = {
    'seed': 20261012,
    'name': 'comparison',
    'architecture': 'resnet18',
    'image_size': 224,
    'bottom_crop_fraction': 0.04,
    'foreground_crop': None,
    'epochs': 24,
    'patience': 6,
    'min_epochs': 0,
    'batch_size': 16,
    'backbone_lr': 0.00002,
    'head_lr': 0.0003,
    'weight_decay': 0.001,
    'fine_tune': 'all',
    'warmup_epochs': 2,
    'class_weighting': 'none',
    'train_sampling': 'uniform',
    'label_smoothing': 0.05,
    'augmentation': 'moderate',
    'selection_metric': 'accuracy',
    'scheduler': 'cosine',
}


def load_config(config):
    if not isinstance(config, dict):
        config = json.loads(Path(config).expanduser().read_text())
    unknown = set(config) - set(DEFAULTS) - {'target'}
    if unknown:
        raise ValueError(f'Unknown experiment settings: {sorted(unknown)}')
    cfg = {**DEFAULTS, **config}
    if cfg.get('target') not in {'life_stage', 'engorgement_level'}:
        raise ValueError('target must be life_stage or engorgement_level.')
    choices = {
        'architecture': ARCHITECTURES,
        'fine_tune': ('head', 'last_block', 'all'),
        'class_weighting': ('none', 'inverse', 'sqrt_inverse'),
        'train_sampling': ('uniform', 'joint_sqrt_inverse'),
        'augmentation': ('baseline', 'moderate'),
        'selection_metric': ('accuracy', 'balanced_accuracy'),
        'scheduler': ('none', 'cosine'),
        'foreground_crop': (None, FOREGROUND_VERSION),
    }
    for key, allowed in choices.items():
        if cfg[key] not in allowed:
            raise ValueError(f'{key} must be one of {allowed}.')
    if cfg['train_sampling'] != 'uniform':
        if cfg['class_weighting'] != 'none':
            raise ValueError('Nonuniform train_sampling requires class_weighting=none.')
        if cfg['target'] != 'engorgement_level':
            raise ValueError('Nonuniform train_sampling is restricted to engorgement_level.')
    for key in ('epochs', 'patience', 'batch_size', 'image_size'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    for key in ('seed', 'warmup_epochs', 'min_epochs'):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer.')
    if cfg['seed'] >= 2**32:
        raise ValueError('seed must be less than 2**32.')
    if cfg['image_size'] < 32:
        raise ValueError('image_size must be at least 32.')
    if cfg['warmup_epochs'] >= cfg['epochs']:
        raise ValueError('warmup_epochs must be less than epochs.')
    if cfg['min_epochs'] > cfg['epochs']:
        raise ValueError('min_epochs must not exceed epochs.')
    for key in ('backbone_lr', 'head_lr', 'weight_decay',
                'bottom_crop_fraction', 'label_smoothing'):
        value = cfg[key]
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be a finite nonnegative number.')
    if cfg['head_lr'] == 0 or cfg['backbone_lr'] == 0:
        raise ValueError('Learning rates must be positive.')
    if not 0 <= cfg['bottom_crop_fraction'] < 0.5:
        raise ValueError('bottom_crop_fraction must be between 0 and 0.5 (exclusive).')
    if not 0 <= cfg['label_smoothing'] < 1:
        raise ValueError('label_smoothing must be between 0 and 1 (exclusive).')
    if not isinstance(cfg['name'], str) or not cfg['name'].strip():
        raise ValueError('name must be a nonempty string.')
    return cfg


def verified_records(target):
    """Read/hash only the approved train/dev image manifest and its photos."""
    manifest = WORK / 'data/penn_train_dev.csv'
    preflight = json.loads((WORK / 'notes/preflight.json').read_text())
    if not SOURCE.is_file():
        raise FileNotFoundError(f'Missing original prepared_data.csv: {SOURCE}')
    source_hash, manifest_hash = digest(SOURCE), digest(manifest)
    if source_hash != preflight['source_sha256']:
        raise ValueError('Source CSV changed; review data before training.')
    if manifest_hash != preflight['manifest_sha256']:
        raise ValueError('Train/dev manifest changed; review data before training.')
    records = rows(manifest)
    counts = Counter(r['provisional_split'] for r in records)
    if counts != {'train': 241, 'dev': 110}:
        raise ValueError(f'Expected the fixed 241 train / 110 dev split, found {dict(counts)}.')
    if len({r['tick_id'] for r in records}) != len(records):
        raise ValueError('Duplicate tick IDs in the approved train/dev manifest.')
    for record in records:
        image_path = resolve_image_path(record['source_path'])
        if not image_path.is_file():
            raise FileNotFoundError(f'Missing train/dev photo: {image_path}')
        if digest(image_path) != record['sha256']:
            raise ValueError(f'Photo SHA256 changed: {image_path}')
    train = [r for r in records if r['provisional_split'] == 'train']
    dev = [r for r in records if r['provisional_split'] == 'dev']
    classes = sorted({r[target] for r in train})
    if {r[target] for r in dev} != set(classes):
        raise ValueError('Train/dev classes differ.')
    return train, dev, classes, source_hash, manifest_hash


def build_train_sampling(records, cfg):
    """Plan row sampling from training metadata only; never load an image.

    Joint weights are n(stage, engorgement)^(-1/2) per row. A full run
    draws 241 rows with replacement; a smoke run draws its smaller train
    subset length. Stage is used only to select training rows, never as a
    model input. Similarity-group multiplicity is reported without silently
    changing the approved row-based policy.
    """
    if not records:
        raise ValueError('Training sampling requires nonempty train records.')
    if any(record.get('provisional_split') != 'train' for record in records):
        raise ValueError('Training sampling accepts train records only.')
    if any(not record.get('tick_id') or not record.get('similarity_group')
           or not record.get('life_stage') or not record.get('engorgement_level')
           for record in records):
        raise ValueError('Training sampling requires IDs, groups and both labels.')
    if len({record['tick_id'] for record in records}) != len(records):
        raise ValueError('Training sampling requires unique tick IDs.')
    policy = cfg['train_sampling']
    if policy not in {'uniform', 'joint_sqrt_inverse'}:
        raise ValueError(f'Unsupported train_sampling: {policy!r}')
    if policy != 'uniform' and cfg['class_weighting'] != 'none':
        raise ValueError('Nonuniform train_sampling requires class_weighting=none.')
    if policy != 'uniform' and cfg['target'] != 'engorgement_level':
        raise ValueError('Nonuniform train_sampling is restricted to engorgement_level.')
    cells = [(record['life_stage'], record['engorgement_level']) for record in records]
    counts = Counter(cells)
    weights = [1.0 if policy == 'uniform' else counts[cell] ** -0.5 for cell in cells]
    total_weight = sum(weights)
    draws = len(records)
    group_sizes = Counter(record['similarity_group'] for record in records)
    groups = []
    for (stage, engorgement), count in sorted(counts.items()):
        row_weight = 1.0 if policy == 'uniform' else count ** -0.5
        probability = count * row_weight / total_weight
        groups.append({
            'life_stage': stage, 'engorgement_level': engorgement,
            'train_count': count, 'sampling_probability': probability,
            'expected_draws_per_epoch': draws * probability,
            'expected_draws_per_image_per_epoch': draws * row_weight / total_weight,
        })
    provenance = {
        'policy': policy, 'sampling_unit': 'approved training manifest row',
        'count_source': 'train only', 'num_samples_per_epoch': draws,
        'replacement': policy != 'uniform', 'generator_seed': cfg['seed'],
        'weight_definition': ('uniform shuffle without replacement' if policy == 'uniform'
                              else 'per-row n(life_stage, engorgement_level)^(-1/2)'),
        'weights_device': None if policy == 'uniform' else 'cpu',
        'weights_dtype': None if policy == 'uniform' else 'float64',
        'stage_engorgement_groups': groups,
        'similarity_groups': {
            'unique_count': len(group_sizes),
            'max_multiplicity': max(group_sizes.values()),
            'multiplicity_histogram': {
                str(size): count for size, count in sorted(Counter(group_sizes.values()).items())},
            'repeated_groups': [
                {'similarity_group': group, 'count': count}
                for group, count in sorted(group_sizes.items()) if count > 1],
            'duplicate_group_correction': 'none',
        },
        'inference': 'image only; sampling uses stage labels only during training',
    }
    if policy == 'uniform':
        return None, provenance
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double, device='cpu'),
        num_samples=draws, replacement=True,
        generator=torch.Generator(device='cpu').manual_seed(cfg['seed']))
    return sampler, provenance


def image_transform(preprocessing, augmentation=None):
    """Training augmentation (when given) followed by tensor conversion and normalization."""
    size = preprocessing['image_size']
    transforms = torchvision.transforms
    operations = []
    if augmentation is not None:
        operations += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        if augmentation == 'baseline':
            operations.append(transforms.RandomRotation(20, fill=(240, 240, 240)))
        else:
            operations += [
                # Expand before padding so edge-positioned ticks retain their legs.
                transforms.RandomRotation(35, expand=True, fill=(240, 240, 240),
                                          interpolation=transforms.InterpolationMode.BILINEAR),
                partial(ImageOps.pad, size=(size, size), method=Image.Resampling.LANCZOS,
                        color=tuple(preprocessing['pad_color'])),
                transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                       saturation=0.1, hue=0.02),
            ]
    return transforms.Compose(operations + [
        transforms.ToTensor(),
        transforms.Normalize(preprocessing['mean'], preprocessing['std']),
    ])


class CachedImages(Dataset):
    """Decode/resize each approved image once; augment cached PIL images anew."""

    def __init__(self, records, classes, target, preprocessing, augmentation=None):
        self.images = []
        self.labels = [classes.index(r[target]) for r in records]
        for record in records:
            self.images.append(prepare_image(resolve_image_path(record['source_path']), preprocessing))
        self.transform = image_transform(preprocessing, augmentation)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.labels[index]


def classification_metrics(truth, predicted, classes):
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for actual, guess in zip(truth, predicted):
        matrix[actual, guess] += 1
    support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    correct = matrix.diagonal()
    recalls = np.divide(correct, support, out=np.zeros(len(classes)), where=support > 0)
    precisions = np.divide(correct, predicted_support, out=np.zeros(len(classes)),
                           where=predicted_support > 0)
    f1 = np.divide(2 * correct, support + predicted_support, out=np.zeros(len(classes)),
                   where=(support + predicted_support) > 0)
    return {
        'n': len(truth),
        'correct': int(correct.sum()),
        'accuracy': float(correct.sum() / len(truth)),
        'balanced_accuracy': float(recalls[support > 0].mean()),
        'macro_f1': float(f1.mean()),
        'confusion_matrix': matrix.tolist(),
        'classes': classes,
        'per_class': [
            {'class': name, 'support': int(support[i]), 'correct': int(correct[i]),
             'recall': float(recalls[i]), 'precision': float(precisions[i]), 'f1': float(f1[i])}
            for i, name in enumerate(classes)
        ],
    }


def validation_details(records, truth, predicted, probabilities, classes, loss):
    result = classification_metrics(truth, predicted, classes)
    result['loss'] = loss
    result['predictions'] = [
        {'tick_id': record['tick_id'], 'true': classes[actual],
         'predicted': classes[guess], 'uncalibrated_probabilities': probability,
         'life_stage': record['life_stage'], 'engorgement_level': record['engorgement_level']}
        for record, actual, guess, probability in zip(records, truth, predicted, probabilities)
    ]
    result['stage_engorgement_subgroups'] = []
    groups = sorted({(r['life_stage'], r['engorgement_level']) for r in records})
    for life_stage, engorgement in groups:
        indices = [i for i, r in enumerate(records)
                   if (r['life_stage'], r['engorgement_level']) == (life_stage, engorgement)]
        group = classification_metrics([truth[i] for i in indices],
                                       [predicted[i] for i in indices], classes)
        group.update(life_stage=life_stage, engorgement_level=engorgement)
        result['stage_engorgement_subgroups'].append(group)
    return result


def run(config, smoke=False):
    """Train a single bounded experiment; return its directory as a Path."""
    cfg = load_config(config)
    target, seed, architecture = cfg['target'], cfg['seed'], cfg['architecture']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train, dev, classes, source_hash, manifest_hash = verified_records(target)
    if smoke:
        train = [next(r for r in train if r[target] == name) for name in classes]
        dev = [next(r for r in dev if r[target] == name) for name in classes]
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    name = re.sub(r'[^A-Za-z0-9_-]+', '-', cfg['name']).strip('-') or 'comparison'
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out = WORK / 'experiments/runs' / f'{stamp}_{target}_{name}{"_smoke" if smoke else ""}'
    out.mkdir(parents=True)
    preprocessing = preprocessing_spec(cfg['image_size'], cfg['bottom_crop_fraction'], cfg['foreground_crop'])
    started = time.time()
    history = []
    best_score, best_result, best_epoch, stale = -1.0, None, None, 0
    selection_metric = cfg['selection_metric']
    write_json(out / 'completion.json', {'status': 'running', 'smoke_only': smoke})
    print(f'Run: {out}\nDevice: {device}; architecture: {architecture}; selected by: {selection_metric}',
          flush=True)
    try:
        model = build_model(architecture, len(classes), pretrained=True)
        set_trainable(model, architecture, cfg['fine_tune'])
        model.to(device)
        head = list(classifier_head(model, architecture).parameters())
        head_ids = {id(parameter) for parameter in head}
        backbone = [parameter for parameter in model.parameters()
                    if parameter.requires_grad and id(parameter) not in head_ids]
        groups = [{'params': head, 'lr': cfg['head_lr'], 'name': 'head'}]
        if backbone:
            groups.insert(0, {'params': backbone, 'lr': cfg['backbone_lr'], 'name': 'backbone'})
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
        total_epochs = 1 if smoke else cfg['epochs']
        warmup_epochs = 0 if smoke else cfg['warmup_epochs']
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs - warmup_epochs))
            if cfg['scheduler'] == 'cosine' else None)
        counts = np.array([sum(r[target] == name for r in train) for name in classes])
        class_weights = None
        if cfg['class_weighting'] != 'none':
            weights = len(train) / (len(classes) * counts)
            if cfg['class_weighting'] == 'sqrt_inverse':
                weights = np.sqrt(weights)
            class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        training_loss = torch.nn.CrossEntropyLoss(weight=class_weights,
                                                  label_smoothing=cfg['label_smoothing'])
        validation_loss = torch.nn.CrossEntropyLoss(reduction='sum')
        train_dataset = CachedImages(train, classes, target, preprocessing, cfg['augmentation'])
        dev_dataset = CachedImages(dev, classes, target, preprocessing)
        train_sampler, sampling_provenance = build_train_sampling(train, cfg)
        if train_sampler is None:
            # Keep the original uniform loader and its random sequence unchanged.
            train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True,
                                      num_workers=0, generator=torch.Generator().manual_seed(seed))
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=cfg['batch_size'], shuffle=False,
                sampler=train_sampler, num_workers=0,
                generator=torch.Generator().manual_seed(seed))
        dev_loader = DataLoader(dev_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
        provenance = {
            'config': cfg, 'classes': classes, 'train_n': len(train), 'validation_n': len(dev),
            'full_split_counts': {'train': 241, 'dev': 110},
            'device': str(device), 'smoke_only': smoke,
            'weights_sha256': model.initial_weights['sha256'],
            'initial_weights': model.initial_weights,
            'manifest_sha256': manifest_hash, 'source_sha256': source_hash,
            'script_sha256': digest(__file__),
            'modeling_sha256': digest(Path(__file__).with_name('modeling.py')),
            'image_preprocessing_sha256': digest(Path(__file__).with_name('image_preprocessing.py')),
            'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
            'architecture': architecture, 'test_images_used': 0,
            'preprocessing': preprocessing, 'selection_metric': selection_metric,
            'validation_loss_definition': 'Unweighted cross entropy without label smoothing.',
            'class_weights': None if class_weights is None else class_weights.cpu().tolist(),
            'training_sampling': sampling_provenance,
            'batch_norm': 'Running statistics frozen in every training phase.',
            'randomness': 'Seeded Python, NumPy, Torch and train-loader generator; GPU kernels may vary across platforms.',
            'uncertainty': 'Softmax scores are uncalibrated; no clinical or phone validation.',
        }
        write_json(out / 'provenance.json', provenance)
        for epoch in range(1, total_epochs + 1):
            epoch_started = time.time()
            warming_up = epoch <= warmup_epochs
            set_trainable(model, architecture, 'head' if warming_up else cfg['fine_tune'])
            if epoch == warmup_epochs + 1:
                stale = 0
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            loss_sum, seen, train_correct = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = training_loss(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += float(loss.item()) * len(labels)
                seen += len(labels)
                train_correct += int((logits.argmax(1) == labels).sum().item())
            model.eval()
            truth, predicted, probabilities = [], [], []
            dev_loss_sum = 0.0
            with torch.inference_mode():
                for images, labels in dev_loader:
                    logits = model(images.to(device))
                    loss = validation_loss(logits, labels.to(device))
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite validation loss.')
                    dev_loss_sum += float(loss.item())
                    probability = logits.softmax(1).cpu().numpy()
                    truth.extend(labels.tolist())
                    predicted.extend(probability.argmax(1).tolist())
                    probabilities.extend(probability.tolist())
            result = validation_details(dev, truth, predicted, probabilities,
                                        classes, dev_loss_sum / len(dev))
            entry = {
                'epoch': epoch, 'phase': 'head_warmup' if warming_up else cfg['fine_tune'],
                'training_loss': loss_sum / seen, 'training_accuracy': train_correct / seen,
                'validation_loss': result['loss'], 'validation_accuracy': result['accuracy'],
                'validation_balanced_accuracy': result['balanced_accuracy'],
                'validation_macro_f1': result['macro_f1'],
                'learning_rates': {group['name']: group['lr'] for group in optimizer.param_groups},
                'seconds': time.time() - epoch_started,
            }
            history.append(entry)
            write_json(out / 'history.json', history)
            print(json.dumps(entry), flush=True)
            score = result[selection_metric]
            if score > best_score:
                best_score, best_result, best_epoch, stale = score, result, epoch, 0
                checkpoint = {
                    'state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    'classes': classes, 'target': target, 'epoch': epoch,
                    'architecture': architecture, 'preprocessing': preprocessing,
                    'image_size': cfg['image_size'], 'bottom_crop_fraction': cfg['bottom_crop_fraction'],
                    'selection_metric': selection_metric, 'selection_score': score,
                    'smoke_only': smoke,
                }
                temporary = out / 'best.pt.tmp'
                torch.save(checkpoint, temporary)
                temporary.replace(out / 'best.pt')
                result.update(epoch=epoch, selection_metric=selection_metric, selection_score=score,
                              smoke_only=smoke)
                write_json(out / 'validation.json', result)
            elif not warming_up:
                stale += 1
            if scheduler is not None and not warming_up:
                scheduler.step()
            if not warming_up and epoch >= cfg['min_epochs'] and stale >= cfg['patience']:
                break
        completion = {
            'status': 'completed', 'epochs': len(history), 'smoke_only': smoke,
            'selection_metric': selection_metric, 'best_selection_score': best_score,
            'best_epoch': best_epoch,
            'best_validation_accuracy': best_result['accuracy'],
            'best_validation_balanced_accuracy': best_result['balanced_accuracy'],
            'best_validation_macro_f1': best_result['macro_f1'],
            'elapsed_seconds': time.time() - started, 'test_images_used': 0,
        }
        write_json(out / 'completion.json', completion)
    except BaseException as error:
        write_json(out / 'completion.json', {
            'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
            'error': f'{type(error).__name__}: {error}', 'epochs': len(history),
            'smoke_only': smoke, 'test_images_used': 0,
        })
        raise
    print(f'Saved: {out}', flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--smoke', action='store_true')
    arguments = parser.parse_args()
    run(arguments.config, smoke=arguments.smoke)
