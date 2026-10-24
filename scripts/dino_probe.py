"""One fixed, frozen DINOv2 engorgement probe; default is a dry run.

--prepare downloads the pinned official source/weights into torch.hub's cache.
--execute runs the approved 241/110 train/dev experiment. --execute --smoke
uses CPU synthetic images only and is excluded from performance comparisons.
Offline prediction needs the exact cached source directory (including LICENSE)
as well as best.pt; no timm/transformers/xformers install is required.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
import warnings

os.environ['XFORMERS_DISABLED'] = '1'
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn.functional as F

from common import WORK, digest, resolve_image_path, write_json
from experiment import verified_records, validation_details

COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
MODEL = 'dinov2_vits14'
WEIGHT_URL = f'https://dl.fbaipublicfiles.com/dinov2/{MODEL}/{MODEL}_pretrain.pth'
WEIGHT_SHA256 = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
SOURCE_TREE_SHA256 = '8fac617d529511bf20a8e09c788f88b9d3de8835761c0fb0164fe9007cd43c31'
CLASSES = ['fully engorged', 'semi-engorged', 'unengorged']
CONFIG = {
    'name': 'dinov2_vits14_392_frozen_d4_linear', 'target': 'engorgement_level',
    'architecture': MODEL, 'seed': 20261019, 'image_size': 392,
    'bottom_crop_fraction': .04, 'resize': 'letterbox', 'resampling': 'lanczos',
    'pad_color': [240, 240, 240], 'mean': [.485, .456, .406], 'std': [.229, .224, .225],
    'features': 'concat(l2(final_cls),l2(mean(final_patch_tokens)))/sqrt(2)',
    'train_views': 'rot90_k_then_rot90_k_of_horizontal_flip;k=0,1,2,3',
    'dev_views': 'identity', 'view_sample_weight': .125, 'fine_tune': 'frozen',
    'head': 'multinomial_logistic_regression', 'head_C': 10., 'head_l1_ratio': 0.,
    'head_solver': 'lbfgs', 'head_max_iter': 2000, 'head_tol': 1e-6,
    'class_weighting': 'none', 'selection_metric': 'accuracy', 'feature_batch_size': 8,
}


def source_directory():
    return Path(torch.hub.get_dir()) / f'facebookresearch_dinov2_{COMMIT}'


def source_digest(directory):
    directory = Path(directory)
    files = sorted(p for p in directory.rglob('*') if p.is_file()
                   and (p.suffix == '.py' or p.name.startswith('LICENSE')))
    if not files or not (directory / 'hubconf.py').is_file() or not (directory / 'LICENSE').is_file():
        raise FileNotFoundError(f'Missing pinned DINOv2 source/ license: {directory}. Run --prepare.')
    value = hashlib.sha256()
    for path in files:
        value.update(path.relative_to(directory).as_posix().encode() + b'\0')
        value.update(bytes.fromhex(digest(path)))
    return value.hexdigest()


def prepare():
    """Download/cache official assets and construct on CPU; no images or GPU."""
    torch.set_num_threads(2)
    weight_path = Path(torch.hub.get_dir()) / 'checkpoints' / f'{MODEL}_pretrain.pth'
    if source_directory().is_dir() and source_digest(source_directory()) != SOURCE_TREE_SHA256:
        raise ValueError('Pinned DINOv2 source cache changed.')
    if weight_path.is_file() and digest(weight_path) != WEIGHT_SHA256:
        raise ValueError('Pinned DINOv2 weight cache changed.')
    encoder = torch.hub.load(f'facebookresearch/dinov2:{COMMIT}', MODEL,
                             pretrained=True, trust_repo=True)
    encoder.eval().requires_grad_(False)
    metadata = {
        'source_repository': 'https://github.com/facebookresearch/dinov2',
        'source_commit': COMMIT, 'source_directory': str(source_directory()),
        'source_tree_sha256': source_digest(source_directory()),
        'weights_url': WEIGHT_URL, 'weights_path': str(weight_path),
        'weights_sha256': digest(weight_path), 'weights_identifier': 'Weights.LVD142M',
        'license': 'Apache-2.0', 'offline_requires_pinned_source_directory': True,
    }
    if metadata['weights_sha256'] != WEIGHT_SHA256 or metadata['source_tree_sha256'] != SOURCE_TREE_SHA256:
        raise ValueError('Downloaded DINOv2 assets differ from the pinned fingerprints.')
    folder = WORK / 'experiments/dino'
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / 'preparation.json', metadata)
    return encoder, metadata


def image_tensor(image):
    image = ImageOps.exif_transpose(image).convert('RGB')
    if min(image.size) < 2:
        raise ValueError('Image is too small.')
    image = image.crop((0, 0, image.width, round(image.height * .96)))
    image = ImageOps.pad(image, (392, 392), method=Image.Resampling.LANCZOS,
                         color=tuple(CONFIG['pad_color']))
    value = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    return (value - torch.tensor(CONFIG['mean'])[:, None, None]) / torch.tensor(CONFIG['std'])[:, None, None]


@torch.inference_mode()
def normalized_features(encoder, batch):
    """Return CPU float32 normalized CLS+mean-patch embeddings; Tensor batches only."""
    result = encoder.forward_features(batch)
    value = torch.cat((F.normalize(result['x_norm_clstoken'], dim=1),
                       F.normalize(result['x_norm_patchtokens'].mean(1), dim=1)), dim=1) / math.sqrt(2)
    if value.shape[1] != 768 or not torch.isfinite(value).all():
        raise ValueError('Invalid DINOv2 features.')
    return value.cpu()


def raw_scores(features, weight, bias):
    """CPU float64 linear scores; softmax remains uncalibrated."""
    return F.linear(features.cpu().double(), weight.cpu().double(), bias.cpu().double())


def _encode(encoder, records, split, device, synthetic=None):
    values = []
    for index, record in enumerate(records):
        if record['provisional_split'] != split or split not in {'train', 'dev'}:
            raise ValueError('Feature extraction permits the expected train/dev split only.')
        if synthetic is None:
            with Image.open(resolve_image_path(record['source_path'])) as source:
                value = image_tensor(source)
        else:
            value = image_tensor(synthetic[index])
        views = [value] if split == 'dev' else (
            [torch.rot90(value, k, (1, 2)) for k in range(4)]
            + [torch.rot90(value.flip(2), k, (1, 2)) for k in range(4)])
        values.append(normalized_features(encoder, torch.stack(views).to(device)))
        if (index + 1) % 40 == 0:
            print(f'{split}: encoded {index+1}/{len(records)} source images', flush=True)
    return torch.cat(values)


def _features(encoder, train, dev, device, integrity, synthetic=None):
    specification = {**integrity, 'config': CONFIG, 'feature_device': str(device),
                     'torch': str(torch.__version__), 'numpy': np.__version__}
    key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    folder = WORK / 'experiments/dino/features'
    path = folder / f'{key}.pt'
    identities = {'train_ids': [r['tick_id'] for r in train for _ in range(8)],
                  'dev_ids': [r['tick_id'] for r in dev]}
    labels = torch.tensor([CLASSES.index(r[CONFIG['target']]) for r in train for _ in range(8)])
    if synthetic is None and path.is_file():
        sidecar = json.loads(path.with_suffix('.json').read_text())
        if digest(path) != sidecar['sha256']:
            raise ValueError('Feature-cache hash mismatch.')
        cache = torch.load(path, map_location='cpu', weights_only=True)
        if cache['specification'] != specification or any(cache[k] != v for k, v in identities.items()):
            raise ValueError('Feature-cache identity/settings mismatch.')
        if not torch.equal(cache['labels'], labels):
            raise ValueError('Feature-cache training-label mismatch.')
        print('Reusing frozen features:', path, flush=True)
    else:
        cache = {'specification': specification, **identities, 'labels': labels}
        for split, records in [('train', train), ('dev', dev)]:
            cache[split] = _encode(encoder, records, split, device,
                                   None if synthetic is None else synthetic[split])
        if synthetic is None:
            folder.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            torch.save(cache, temporary)
            temporary.replace(path)
            write_json(path.with_suffix('.json'), {'sha256': digest(path), 'specification': specification})
    for split, count in [('train', len(train)*8), ('dev', len(dev))]:
        if cache[split].shape != (count, 768) or not torch.isfinite(cache[split]).all():
            raise ValueError(f'Invalid cached {split} features.')
    return cache, {'key': key, 'path': str(path) if synthetic is None else None,
                   'sha256': digest(path) if synthetic is None else None}


def _synthetic_records():
    records, images = {}, {}
    for split in ['train', 'dev']:
        records[split], images[split] = [], []
        for i, label in enumerate(CLASSES):
            image = Image.new('RGB', (224, 168), [(80, 130, 170), (170, 80, 130), (130, 170, 80)][i])
            ImageDraw.Draw(image).ellipse((30+i*8, 35, 120+i*15, 125), fill=(20, 30, 40))
            records[split].append({'tick_id': f'synthetic-{split}-{i}', 'provisional_split': split,
                                   'life_stage': 'synthetic', 'engorgement_level': label})
            images[split].append(image)
    return records['train'], records['dev'], images


def run(execute=False, smoke=False, device=None):
    if not execute:
        return {'config': CONFIG, 'source_commit': COMMIT, 'weights_url': WEIGHT_URL,
                'downloads': '--prepare or --execute', 'smoke_only': smoke}
    torch.set_num_threads(2)
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    if smoke:
        train, dev, synthetic = _synthetic_records()
        source_hash = manifest_hash = None
        device = 'cpu'
    else:
        train, dev, classes, source_hash, manifest_hash = verified_records(CONFIG['target'])
        if classes != CLASSES or len(train) != 241 or len(dev) != 110:
            raise ValueError('Fixed engorgement classes or 241/110 split changed.')
        synthetic = None
        device = device or ('mps' if torch.backends.mps.is_available() else
                            'cuda' if torch.cuda.is_available() else 'cpu')
    started = time.time()
    out = WORK / 'experiments/dino' / (datetime.now().strftime('%Y%m%d-%H%M%S-%f') + ('_smoke' if smoke else ''))
    out.mkdir(parents=True)
    write_json(out/'completion.json', {'status': 'running', 'smoke_only': smoke, 'test_images_used': 0})
    try:
        encoder, assets = prepare()
        integrity = {'source_sha256': source_hash, 'manifest_sha256': manifest_hash,
                     'script_sha256': digest(__file__), 'helper_sha256': digest(WORK/'scripts/experiment.py'),
                     'common_sha256': digest(WORK/'scripts/common.py'),
                     'weights_sha256': assets['weights_sha256'], 'source_commit': COMMIT,
                     'source_tree_sha256': assets['source_tree_sha256']}
        provenance = {**integrity, 'config': CONFIG, 'assets': assets, 'classes': CLASSES,
                      'train_n': len(train), 'validation_n': len(dev), 'device': str(device),
                      'smoke_only': smoke, 'synthetic_only': smoke, 'test_images_used': 0,
                      'torch': str(torch.__version__), 'sklearn': sklearn.__version__,
                      'selection_metric': 'accuracy', 'head_fits': 1,
                      'uncertainty': 'Uncalibrated scores. Development reused for model selection; no test evaluation.'}
        write_json(out/'provenance.json', provenance)
        encoder.to(device)
        cache, cache_metadata = _features(encoder, train, dev, device, integrity, synthetic)
        encoder.cpu()
        classifier = LogisticRegression(C=10., l1_ratio=0., solver='lbfgs', max_iter=2000,
                                         tol=1e-6, class_weight=None, random_state=CONFIG['seed'])
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            classifier.fit(cache['train'].numpy(), cache['labels'].numpy(),
                           sample_weight=np.full(len(cache['labels']), 1/8))
        if classifier.classes_.tolist() != [0, 1, 2]:
            raise ValueError('Unexpected linear-head class order.')
        weight = torch.tensor(classifier.coef_, dtype=torch.float64)
        bias = torch.tensor(classifier.intercept_, dtype=torch.float64)
        logits = raw_scores(cache['dev'], weight, bias)
        probabilities = logits.softmax(1)
        if not np.allclose(probabilities.numpy(), classifier.predict_proba(cache['dev'].numpy()), atol=1e-10):
            raise ValueError('Exported tensor head differs from sklearn.')
        truth = [CLASSES.index(r[CONFIG['target']]) for r in dev]
        predicted = probabilities.argmax(1).tolist()
        loss = float(F.cross_entropy(logits, torch.tensor(truth)))
        result = validation_details(dev, truth, predicted, probabilities.tolist(), CLASSES, loss)
        result.update(epoch=1, checkpoint_epoch=1, smoke_only=smoke, confidence_calibrated=False)
        for prediction, scores in zip(result['predictions'], logits.tolist()):
            prediction['raw_scores'] = scores
        checkpoint = {'architecture': MODEL, 'target': CONFIG['target'], 'classes': CLASSES,
                      'epoch': 1, 'smoke_only': smoke, 'config': CONFIG, 'assets': assets,
                      'encoder_state_dict': encoder.state_dict(), 'head_weight': weight,
                      'head_bias': bias, 'head_dtype_device': 'float64_cpu'}
        torch.save(checkpoint, out/'best.pt')
        write_json(out/'validation.json', result)
        write_json(out/'history.json', [{'epoch': 1, 'validation_accuracy': result['accuracy'],
                   'validation_balanced_accuracy': result['balanced_accuracy'],
                   'validation_macro_f1': result['macro_f1'], 'validation_loss': loss,
                   'seconds': time.time()-started}])
        provenance.update(feature_cache=cache_metadata, head_iterations=classifier.n_iter_.tolist(),
                          checkpoint_sha256=digest(out/'best.pt'))
        write_json(out/'provenance.json', provenance)
        write_json(out/'completion.json', {'status': 'completed', 'epochs': 1, 'smoke_only': smoke,
                   'test_images_used': 0, 'best_epoch': 1, 'best_validation_accuracy': result['accuracy'],
                   'best_validation_balanced_accuracy': result['balanced_accuracy'],
                   'elapsed_seconds': time.time()-started})
        print(json.dumps({'run': str(out), 'smoke_only': smoke, 'accuracy': result['accuracy']}), flush=True)
        return out
    except BaseException as error:
        write_json(out/'completion.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   'smoke_only': smoke, 'test_images_used': 0, 'error': f'{type(error).__name__}: {error}'})
        raise


def predict_image(image_path, checkpoint, device='cpu', source_dir=None):
    """Offline single-image prediction; exact pinned Hub source must be present."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (saved.get('smoke_only') is not False or saved.get('config') != CONFIG
            or saved.get('classes') != CLASSES or saved.get('architecture') != MODEL
            or saved.get('target') != CONFIG['target']
            or saved.get('assets', {}).get('source_commit') != COMMIT):
        raise ValueError('Expected a completed non-smoke checkpoint for this fixed recipe.')
    folder = Path(source_dir) if source_dir is not None else source_directory()
    if source_digest(folder) != saved['assets']['source_tree_sha256'] or source_digest(folder) != SOURCE_TREE_SHA256:
        raise ValueError('Offline DINOv2 source differs from the recorded source snapshot.')
    encoder = torch.hub.load(str(folder), MODEL, source='local', pretrained=False)
    encoder.load_state_dict(saved['encoder_state_dict'], strict=True)
    encoder.eval().requires_grad_(False).to(device)
    with Image.open(image_path) as source:
        features = normalized_features(encoder, image_tensor(source).unsqueeze(0).to(device))
    logits = raw_scores(features, saved['head_weight'], saved['head_bias'])[0]
    if not torch.isfinite(logits).all():
        raise ValueError('DINOv2 linear head returned non-finite scores.')
    probabilities = logits.softmax(0)
    if not torch.isfinite(probabilities).all():
        raise ValueError('DINOv2 linear head returned non-finite probabilities.')
    return {'task': CONFIG['target'], 'prediction': CLASSES[int(probabilities.argmax())],
            'confidence': float(probabilities.max()), 'confidence_calibrated': False,
            'model_scores_uncalibrated': dict(zip(CLASSES, probabilities.tolist())),
            'raw_scores': dict(zip(CLASSES, logits.tolist())), 'checkpoint': str(checkpoint)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
    args = parser.parse_args()
    if args.prepare:
        if args.execute or args.smoke:
            parser.error('--prepare is a separate CPU-only download step.')
        _, metadata = prepare()
        print(json.dumps(metadata, indent=2))
    else:
        output = run(args.execute, args.smoke, args.device)
        if not args.execute:
            print(json.dumps(output, indent=2))
