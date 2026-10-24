"""v2 experiment: one image -> species, life stage and engorgement.

One backbone with three classification heads (sliced rows of one final linear
layer). Each source supervises only the heads it labels:

    iNaturalist (dataset/inaturalist)  species; life stage where annotated
    Penn (fixed v1 train/dev split)     life stage and engorgement
                                        (species only with penn_species=true,
                                        off by default: every Penn photo is a
                                        microscope image of a deer tick, so the
                                        model could learn "microscope = deer tick")

Labels may be sets: an iNaturalist "adult" without a recorded sex accepts
either adult class, so its loss is -log(p(adult female) + p(adult male)).

Images use the same letterbox preprocessing as v1 and the iOS app. Only train
and dev splits are loaded; the Penn test images and the iNaturalist test
photographers are never read. The checkpoint is selected by the mean of
iNaturalist species balanced accuracy, Penn life-stage accuracy and Penn
engorgement accuracy on dev. Softmax scores are uncalibrated.

    .venv/bin/python scripts/multihead_experiment.py --config configs/v2_multihead.json            # smoke
    .venv/bin/python scripts/multihead_experiment.py --config configs/v2_multihead.json --execute  # train
"""

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from multiprocessing import Pool
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision

from common import WORK, digest, resolve_image_path, write_json
from download_inaturalist import OUT as INAT, SPECIES
from experiment import classification_metrics, image_transform, verified_records
from image_preprocessing import prepare_image
from modeling import ARCHITECTURES, build_model, classifier_head, preprocessing_spec, set_trainable

HEADS = {
    'species': [name for _, name, _, _ in SPECIES],
    # No larva class: the selected photos contain no larva annotations to learn from.
    'life_stage': ['adult female', 'adult male', 'nymph'],
    'engorgement_level': ['fully engorged', 'semi-engorged', 'unengorged'],
}
DEER_TICK = 'Ixodes scapularis'

DEFAULTS = {
    'name': 'v2_multihead', 'seed': 20261019, 'architecture': 'convnext_tiny',
    'image_size': 384, 'bottom_crop_fraction': 0.04,
    'epochs': 20, 'patience': 5, 'min_epochs': 6, 'warmup_epochs': 1,
    'batch_size': 32, 'backbone_lr': 3e-05, 'head_lr': 3e-04, 'weight_decay': 0.01,
    'fine_tune': 'all', 'scheduler': 'cosine', 'augmentation': 'moderate',
    'class_weighting': 'sqrt_inverse', 'penn_repeat': 8, 'penn_species': False,
    'head_weights': {'species': 1.0, 'life_stage': 1.0, 'engorgement_level': 1.0},
    'num_workers': 8, 'mixed_precision': True,
}


def load_config(path):
    config = json.loads(Path(path).expanduser().read_text())
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown settings: {sorted(unknown)}')
    cfg = {**DEFAULTS, **config, 'head_weights': {**DEFAULTS['head_weights'], **config.get('head_weights', {})}}
    if cfg['architecture'] not in ARCHITECTURES:
        raise ValueError(f'architecture must be one of {ARCHITECTURES}')
    if set(cfg['head_weights']) != set(HEADS):
        raise ValueError(f'head_weights must cover exactly {sorted(HEADS)}')
    if cfg['class_weighting'] not in ('none', 'sqrt_inverse', 'inverse'):
        raise ValueError('class_weighting must be none, sqrt_inverse or inverse')
    return cfg


def head_rows():
    rows, start = {}, 0
    for head, classes in HEADS.items():
        rows[head] = (start, start + len(classes))
        start += len(classes)
    return rows, start


# ---------------------------------------------------------------- records

def _record(source, record_id, path, split, **labels):
    """labels: head -> list of allowed class names (empty or missing = unlabeled)."""
    for head, allowed in labels.items():
        unknown = set(allowed) - set(HEADS[head])
        if unknown:
            raise ValueError(f'{source} {record_id}: unknown {head} labels {unknown}')
    return {'source': source, 'id': record_id, 'path': str(path), 'split': split,
            'labels': {head: list(labels.get(head) or []) for head in HEADS}}


def penn_records(cfg):
    train, dev, _, source_hash, manifest_hash = verified_records('life_stage')
    records = [
        _record('penn', r['tick_id'], resolve_image_path(r['source_path']), r['provisional_split'],
                life_stage=[r['life_stage']], engorgement_level=[r['engorgement_level']],
                species=[DEER_TICK] if cfg['penn_species'] else [])
        for r in train + dev
    ]
    return records, {'penn_source_sha256': source_hash, 'penn_manifest_sha256': manifest_hash}


def inat_life_stage(row):
    stage, sex = row['life_stage'], row['sex']
    if stage in ('nymph', 'adult female', 'adult male'):
        return [stage]
    if stage == 'adult':
        return ['adult female', 'adult male']
    if not stage and sex in ('female', 'male'):
        return [f'adult {sex}']  # only adult ticks can be sexed
    return []  # unannotated, or larva (no larva class)


def inat_records():
    manifest = INAT / 'manifest.csv'
    with manifest.open() as file:
        rows = list(csv.DictReader(file))
    records, missing = [], 0
    for row in rows:
        if row['split'] == 'test':
            continue  # held-out photographers; never read
        path = INAT / 'images' / f"{row['photo_id']}.jpg"
        if not path.is_file():
            missing += 1
            continue
        records.append(_record('inaturalist', row['observation_id'], path, row['split'],
                               species=[row['species']], life_stage=inat_life_stage(row)))
    return records, {'inaturalist_manifest_sha256': digest(manifest), 'inaturalist_missing_photos': missing}


def label_counts(records):
    counts = {}
    for split in ('train', 'dev'):
        for head in HEADS:
            counter = Counter()
            for r in records:
                if r['split'] == split and r['labels'][head]:
                    allowed = r['labels'][head]
                    counter[allowed[0] if len(allowed) == 1 else ' or '.join(allowed)] += 1
            counts[f'{split}/{head}'] = dict(sorted(counter.items()))
    return counts


# ---------------------------------------------------------------- data

def _prepare(arguments):
    path, preprocessing = arguments
    return prepare_image(path, preprocessing)


def prepare_all(records, preprocessing, workers):
    """Decode and letterbox every image once, in parallel."""
    with Pool(workers) as pool:
        return pool.map(_prepare, [(r['path'], preprocessing) for r in records], chunksize=32)


class MultiHeadImages(Dataset):
    def __init__(self, images, records, indices, preprocessing, augmentation=None):
        self.images, self.records, self.indices = images, records, indices
        self.transform = image_transform(preprocessing, augmentation)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        masks = []
        for head, classes in HEADS.items():
            mask = torch.zeros(len(classes))
            for name in self.records[index]['labels'][head]:
                mask[classes.index(name)] = 1
            masks.append(mask)
        return self.transform(self.images[index]), torch.cat(masks), index


def set_loss(logits, mask, class_weights):
    """Mean of -log P(label set) over rows that have a label for this head."""
    labeled = mask.sum(1) > 0
    if not labeled.any():
        return logits.sum() * 0, 0
    logits, mask = logits[labeled].float(), mask[labeled]
    log_p = logits.log_softmax(1)
    loss = -(log_p.masked_fill(mask == 0, float('-inf')).logsumexp(1))
    if class_weights is not None:
        weights = (mask * class_weights).sum(1) / mask.sum(1)
        loss = loss * weights
        return loss.sum() / weights.sum(), int(labeled.sum())
    return loss.mean(), int(labeled.sum())


def class_weights(records, head, mode, device):
    if mode == 'none':
        return None
    counts = Counter(r['labels'][head][0] for r in records if len(r['labels'][head]) == 1)
    present = [counts.get(name, 0) for name in HEADS[head]]
    total = sum(present)
    weights = [total / (len(present) * c) if c else 0.0 for c in present]
    if mode == 'sqrt_inverse':
        weights = [w ** 0.5 for w in weights]
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------- evaluation

def evaluate(model, loader, records, rows, device, autocast):
    """Per head and source: metrics on exactly labeled rows, plus set accuracy on all labeled rows."""
    model.eval()
    outputs = {}
    with torch.inference_mode(), autocast():
        for images, _, indices in loader:
            logits = model(images.to(device)).float().cpu()
            for row, index in zip(logits, indices.tolist()):
                outputs[index] = row
    results = {}
    for head, (start, end) in rows.items():
        classes = HEADS[head]
        for source in ('inaturalist', 'penn'):
            chosen = [i for i in outputs if records[i]['source'] == source and records[i]['labels'][head]]
            if not chosen:
                continue
            probabilities = {i: outputs[i][start:end].softmax(0) for i in chosen}
            exact = [i for i in chosen if len(records[i]['labels'][head]) == 1]
            entry = {'labeled': len(chosen),
                     'set_accuracy': float(np.mean([classes[int(probabilities[i].argmax())]
                                                    in records[i]['labels'][head] for i in chosen]))}
            if exact:
                truth = [classes.index(records[i]['labels'][head][0]) for i in exact]
                predicted = [int(probabilities[i].argmax()) for i in exact]
                metrics = classification_metrics(truth, predicted, classes)
                nll = float(np.mean([-np.log(max(float(probabilities[i][t]), 1e-12))
                                     for i, t in zip(exact, truth)]))
                entry.update(metrics, nll=nll)
            results[f'{source}/{head}'] = entry
    return results


def selection_score(results):
    parts = [results.get('inaturalist/species', {}).get('balanced_accuracy'),
             results.get('penn/life_stage', {}).get('accuracy'),
             results.get('penn/engorgement_level', {}).get('accuracy')]
    parts = [p for p in parts if p is not None]
    return float(np.mean(parts)) if parts else 0.0


# ---------------------------------------------------------------- training

def run(config, smoke=True):
    cfg = load_config(config)
    seed, architecture = cfg['seed'], cfg['architecture']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    use_amp = cfg['mixed_precision'] and device.type == 'cuda'

    def autocast():
        return torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp)

    penn, penn_info = penn_records(cfg)
    inat, inat_info = inat_records()
    records = inat + penn
    if smoke:
        rng = random.Random(seed)
        keep = []
        for split in ('train', 'dev'):
            for source in ('inaturalist', 'penn'):
                pool = [r for r in records if r['split'] == split and r['source'] == source]
                keep += rng.sample(pool, min(len(pool), 24))
        records = keep
    counts = label_counts(records)

    rows, total = head_rows()
    preprocessing = preprocessing_spec(cfg['image_size'], cfg['bottom_crop_fraction'])
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    name = re.sub(r'[^A-Za-z0-9_-]+', '-', cfg['name']).strip('-') or 'v2'
    out = WORK / 'experiments/multihead' / f'{stamp}_{name}{"_smoke" if smoke else ""}'
    out.mkdir(parents=True)
    write_json(out / 'completion.json', {'status': 'running', 'smoke_only': smoke})
    print(f'Run: {out}\nDevice: {device}; architecture: {architecture}; images: {len(records)}', flush=True)
    print(json.dumps(counts, indent=1), flush=True)

    started = time.time()
    images = prepare_all(records, preprocessing, workers=min(16, max(2, len(records) // 64 or 2)))
    print(f'Prepared {len(images)} images in {time.time() - started:.0f} s', flush=True)
    train_idx = [i for i, r in enumerate(records) if r['split'] == 'train']
    dev_idx = [i for i, r in enumerate(records) if r['split'] == 'dev']
    repeat = 1 if smoke else cfg['penn_repeat']
    epoch_idx = [i for i in train_idx for _ in range(repeat if records[i]['source'] == 'penn' else 1)]

    workers = 0 if smoke else cfg['num_workers']
    train_loader = DataLoader(MultiHeadImages(images, records, epoch_idx, preprocessing, cfg['augmentation']),
                              batch_size=cfg['batch_size'], shuffle=True, num_workers=workers,
                              persistent_workers=workers > 0, drop_last=len(epoch_idx) > cfg['batch_size'],
                              generator=torch.Generator().manual_seed(seed))
    dev_loader = DataLoader(MultiHeadImages(images, records, dev_idx, preprocessing),
                            batch_size=cfg['batch_size'] * 2, shuffle=False, num_workers=workers)

    history, best_score, best_epoch, stale = [], -1.0, None, 0
    try:
        model = build_model(architecture, total, pretrained=not smoke)
        set_trainable(model, architecture, cfg['fine_tune'])
        model.to(device)
        head = list(classifier_head(model, architecture).parameters())
        head_ids = {id(p) for p in head}
        backbone = [p for p in model.parameters() if id(p) not in head_ids]
        optimizer = torch.optim.AdamW([
            {'params': backbone, 'lr': cfg['backbone_lr'], 'name': 'backbone'},
            {'params': head, 'lr': cfg['head_lr'], 'name': 'head'},
        ], weight_decay=cfg['weight_decay'])
        epochs, warmup = (1, 0) if smoke else (cfg['epochs'], cfg['warmup_epochs'])
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup))
                     if cfg['scheduler'] == 'cosine' else None)
        train_records = [records[i] for i in train_idx]
        weights = {h: class_weights(train_records, h, cfg['class_weighting'], device) for h in HEADS}
        write_json(out / 'provenance.json', {
            'config': cfg, 'smoke_only': smoke, 'device': str(device), 'heads': HEADS, 'logit_rows': rows,
            'preprocessing': preprocessing, 'label_counts': counts,
            'class_weights': {h: None if w is None else w.cpu().tolist() for h, w in weights.items()},
            'train_n': len(train_idx), 'dev_n': len(dev_idx), 'epoch_n': len(epoch_idx),
            'test_images_used': 0, **penn_info, **inat_info,
            'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'weights_sha256': model.initial_weights['sha256'], 'initial_weights': model.initial_weights,
            'script_sha256': digest(__file__),
            'selection': 'mean of iNaturalist species balanced accuracy, Penn life-stage accuracy, '
                         'Penn engorgement accuracy (dev)',
            'objective': 'sum over heads of head_weight * class-weighted -log P(label set); unlabeled heads skipped',
            'batch_norm': 'Running statistics frozen during training, as in v1.',
            'uncertainty': 'Softmax scores are uncalibrated.',
        })
        for epoch in range(1, epochs + 1):
            epoch_started = time.time()
            warming = epoch <= warmup
            set_trainable(model, architecture, 'head' if warming else cfg['fine_tune'])
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            sums, labeled = Counter(), Counter()
            for images_batch, masks, _ in train_loader:
                images_batch, masks = images_batch.to(device, non_blocking=True), masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    logits = model(images_batch)
                loss = 0
                for h, (start, end) in rows.items():
                    head_loss, n = set_loss(logits[:, start:end], masks[:, start:end], weights[h])
                    loss = loss + cfg['head_weights'][h] * head_loss
                    sums[h] += float(head_loss.detach()) * n
                    labeled[h] += n
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            results = evaluate(model, dev_loader, records, rows, device, autocast)
            score = selection_score(results)
            entry = {'epoch': epoch, 'phase': 'head_warmup' if warming else cfg['fine_tune'],
                     'training_loss': {h: sums[h] / labeled[h] for h in HEADS if labeled[h]},
                     'selection_score': score,
                     'dev': {k: {m: v[m] for m in ('accuracy', 'balanced_accuracy', 'set_accuracy', 'nll', 'labeled')
                                 if m in v} for k, v in results.items()},
                     'learning_rates': {g['name']: g['lr'] for g in optimizer.param_groups},
                     'seconds': time.time() - epoch_started}
            history.append(entry)
            write_json(out / 'history.json', history)
            print(json.dumps(entry), flush=True)
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
                checkpoint = {
                    'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'architecture': architecture, 'heads': HEADS, 'logit_rows': rows,
                    'preprocessing': preprocessing, 'image_size': cfg['image_size'],
                    'bottom_crop_fraction': cfg['bottom_crop_fraction'], 'epoch': epoch,
                    'selection_score': score, 'smoke_only': smoke, 'target': 'multihead',
                }
                torch.save(checkpoint, out / 'best.pt.tmp')
                (out / 'best.pt.tmp').replace(out / 'best.pt')
                write_json(out / 'validation.json', {'epoch': epoch, 'selection_score': score,
                                                     'smoke_only': smoke, 'results': results})
            elif not warming:
                stale += 1
            if scheduler is not None and not warming:
                scheduler.step()
            if not warming and epoch >= cfg['min_epochs'] and stale >= cfg['patience']:
                break
        write_json(out / 'completion.json', {
            'status': 'completed', 'epochs': len(history), 'best_epoch': best_epoch,
            'best_selection_score': best_score, 'smoke_only': smoke, 'test_images_used': 0,
            'elapsed_seconds': time.time() - started})
    except BaseException as error:
        write_json(out / 'completion.json', {
            'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
            'error': f'{type(error).__name__}: {error}', 'epochs': len(history),
            'smoke_only': smoke, 'test_images_used': 0})
        raise
    print(f'Saved: {out}', flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--execute', action='store_true', help='Full pretrained training; default is a smoke run.')
    arguments = parser.parse_args()
    run(arguments.config, smoke=not arguments.execute)
