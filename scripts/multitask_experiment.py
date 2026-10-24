"""One auxiliary-task ablation: shared pixels -> life stage and engorgement.

The objective is always L_engorgement + 0.25 * L_life_stage. The checkpoint
for both tasks is selected by engorgement development accuracy only. Labels
are supervision, never model inputs. There is no protected-test loader.

CLI defaults to a one-epoch CPU smoke run with random weights; --execute is
required for full pretrained training. Exported heads are ordinary inference
checkpoints. This experiment does not replace or promote any existing model.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision

from common import WORK, digest, write_json
from experiment import CachedImages, load_config, validation_details, verified_records
from modeling import build_model, classifier_head, preprocessing_spec, set_trainable


PRIMARY_TASK = 'engorgement_level'
LIFE_LOSS_WEIGHT = 0.25
TASK_ROWS = {'life_stage': (0, 2), 'engorgement_level': (2, 5)}
EXPECTED_CLASSES = {
    'life_stage': {'adult female', 'nymph'},
    'engorgement_level': {'fully engorged', 'semi-engorged', 'unengorged'},
}


def multitask_config(config):
    if not isinstance(config, dict):
        config = json.loads(Path(config).expanduser().read_text())
    if config.get('target', PRIMARY_TASK) != PRIMARY_TASK:
        raise ValueError('This ablation selects the shared model for engorgement_level.')
    if config.get('selection_metric', 'accuracy') != 'accuracy':
        raise ValueError('This ablation always selects ordinary engorgement accuracy.')
    if config.get('train_sampling', 'uniform') != 'uniform':
        raise ValueError('Multitask training supports train_sampling=uniform only.')
    return load_config({'name': 'multitask_aux025', **config, 'target': PRIMARY_TASK,
                        'selection_metric': 'accuracy'})


class JointImages(CachedImages):
    """Apply one image augmentation and return both independent target labels."""

    def __init__(self, records, classes, preprocessing, augmentation=None):
        super().__init__(records, classes[PRIMARY_TASK], PRIMARY_TASK,
                         preprocessing, augmentation)
        self.life_labels = [classes['life_stage'].index(r['life_stage']) for r in records]

    def __getitem__(self, index):
        image, engorgement = super().__getitem__(index)
        return image, self.life_labels[index], engorgement


def split_logits(logits):
    if logits.ndim != 2 or logits.shape[1] != 5:
        raise ValueError('A joint classifier must produce exactly five logits per image.')
    return {task: logits[:, start:end] for task, (start, end) in TASK_ROWS.items()}


def export_state_dicts(model, architecture):
    """Slice only the final linear rows; shared features remain identical.

    Each returned state dictionary loads strictly into build_model with that
    task's 2/3 classes, and its logits equal the corresponding joint slice.
    """
    head = classifier_head(model, architecture)
    linear = head if isinstance(head, torch.nn.Linear) else head[-1]
    if not isinstance(linear, torch.nn.Linear) or linear.out_features != 5:
        raise ValueError('Expected a five-output final linear layer.')
    name = next(name for name, module in model.named_modules() if module is linear)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    exports = {}
    for task, (start, end) in TASK_ROWS.items():
        exports[task] = dict(state)
        for suffix in ('weight', 'bias'):
            key = f'{name}.{suffix}'
            if key in state:
                exports[task][key] = state[key][start:end].clone()
    return exports


def _smoke_records(records):
    # First example of each observed joint subgroup covers both label spaces.
    chosen = {}
    for record in records:
        chosen.setdefault((record['life_stage'], record[PRIMARY_TASK]), record)
    return list(chosen.values())


def _class_weights(records, classes, task, mode, device):
    if mode == 'none':
        return None
    counts = np.array([sum(r[task] == name for r in records) for name in classes])
    weights = len(records) / (len(classes) * counts)
    if mode == 'sqrt_inverse':
        weights = np.sqrt(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run(config, smoke=True):
    """Return summary/life_stage/engorgement_level directories; smoke is default."""
    cfg = multitask_config(config)
    seed, architecture = cfg['seed'], cfg['architecture']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2 if smoke else 4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train, dev, _, source_hash, manifest_hash = verified_records(PRIMARY_TASK)
    classes = {task: sorted({r[task] for r in train}) for task in TASK_ROWS}
    for task in TASK_ROWS:
        if set(classes[task]) != EXPECTED_CLASSES[task] or {r[task] for r in dev} != set(classes[task]):
            raise ValueError(f'Unexpected train/dev labels for {task}.')
    if smoke:
        train, dev = _smoke_records(train), _smoke_records(dev)
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'mps' if torch.backends.mps.is_available() else 'cpu')
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    name = re.sub(r'[^A-Za-z0-9_-]+', '-', cfg['name']).strip('-') or 'aux025'
    run_id = f'{stamp}_multitask_{name}{"_smoke" if smoke else ""}'
    paths = {'summary': WORK / 'experiments/multitask' / run_id}
    paths.update({task: WORK / 'experiments/runs' / f'{run_id}_{task}' for task in TASK_ROWS})
    for folder in paths.values():
        folder.mkdir(parents=True)
        write_json(folder / 'completion.json', {'status': 'running', 'smoke_only': smoke})
    preprocessing = preprocessing_spec(cfg['image_size'], cfg['bottom_crop_fraction'], cfg['foreground_crop'])
    history = []
    task_history = {task: [] for task in TASK_ROWS}
    best_score, best_epoch, best_results, stale = -1.0, None, None, 0
    started = time.time()
    print(f'Multitask run: {paths["summary"]}\nDevice: {device}; objective: L_eng + 0.25 L_life',
          flush=True)
    try:
        model = build_model(architecture, 5, pretrained=not smoke)
        set_trainable(model, architecture, cfg['fine_tune'])
        model.to(device)
        head = list(classifier_head(model, architecture).parameters())
        head_ids = {id(parameter) for parameter in head}
        backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        groups = [{'params': head, 'lr': cfg['head_lr'], 'name': 'head'}]
        if backbone:
            groups.insert(0, {'params': backbone, 'lr': cfg['backbone_lr'], 'name': 'backbone'})
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
        epochs, warmup = (1, 0) if smoke else (cfg['epochs'], cfg['warmup_epochs'])
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup))
                     if cfg['scheduler'] == 'cosine' else None)
        weights = {task: _class_weights(train, classes[task], task, cfg['class_weighting'], device)
                   for task in TASK_ROWS}
        criteria = {task: torch.nn.CrossEntropyLoss(weight=weights[task],
                                                     label_smoothing=cfg['label_smoothing'])
                    for task in TASK_ROWS}
        validation_loss = torch.nn.CrossEntropyLoss(reduction='sum')
        train_loader = DataLoader(JointImages(train, classes, preprocessing, cfg['augmentation']),
                                  batch_size=cfg['batch_size'], shuffle=True, num_workers=0,
                                  generator=torch.Generator().manual_seed(seed))
        dev_loader = DataLoader(JointImages(dev, classes, preprocessing),
                                batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
        shared = {
            'architecture': architecture, 'train_n': len(train), 'validation_n': len(dev),
            'full_split_counts': {'train': 241, 'dev': 110}, 'device': str(device),
            'smoke_only': smoke, 'test_images_used': 0,
            'source_sha256': source_hash, 'manifest_sha256': manifest_hash,
            'script_sha256': digest(__file__),
            'experiment_sha256': digest(Path(__file__).with_name('experiment.py')),
            'modeling_sha256': digest(Path(__file__).with_name('modeling.py')),
            'image_preprocessing_sha256': digest(Path(__file__).with_name('image_preprocessing.py')),
            'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
            'weights_sha256': model.initial_weights['sha256'], 'initial_weights': model.initial_weights,
            'preprocessing': preprocessing, 'selection_task': PRIMARY_TASK,
            'training_mode': 'multitask_auxiliary_life_stage', 'multitask_run_id': run_id,
            'objective': 'L_engorgement + 0.25 * L_life_stage',
            'life_stage_loss_coefficient': LIFE_LOSS_WEIGHT,
            'objective_components': 'Per-task cross entropy with configured class weighting and label smoothing.',
            'validation_loss_definition': 'Per-task unweighted cross entropy without label smoothing.',
            'model_inputs': 'Image pixels only; labels are training supervision and evaluation targets.',
            'batch_norm': 'Running statistics frozen in every training phase.',
            'randomness': 'Seeded Python, NumPy, Torch and train-loader generator; GPU kernels may vary across platforms.',
            'uncertainty': 'Softmax scores are uncalibrated; no clinical or phone validation.',
            'summary_folder': str(paths['summary'].relative_to(WORK)),
        }
        write_json(paths['summary'] / 'provenance.json', {
            **shared, 'config': cfg, 'classes_by_task': classes, 'logit_rows': TASK_ROWS,
            'selection_metric': 'engorgement_accuracy',
            'exports': {task: str(paths[task].relative_to(WORK)) for task in TASK_ROWS},
        })
        for task in TASK_ROWS:
            metric = 'accuracy' if task == PRIMARY_TASK else 'engorgement_accuracy'
            write_json(paths[task] / 'provenance.json', {
                **shared, 'config': {**cfg, 'target': task, 'selection_metric': metric},
                'classes': classes[task], 'selection_metric': metric,
                'class_weights': None if weights[task] is None else weights[task].cpu().tolist(),
            })
        for epoch in range(1, epochs + 1):
            epoch_started = time.time()
            warming_up = epoch <= warmup
            set_trainable(model, architecture, 'head' if warming_up else cfg['fine_tune'])
            if epoch == warmup + 1:
                stale = 0
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            train_loss = {task: 0.0 for task in TASK_ROWS}
            train_correct = {task: 0 for task in TASK_ROWS}
            total_loss, seen = 0.0, 0
            for images, life, engorgement in train_loader:
                labels = {'life_stage': life.to(device), PRIMARY_TASK: engorgement.to(device)}
                optimizer.zero_grad(set_to_none=True)
                logits = split_logits(model(images.to(device)))
                losses = {task: criteria[task](logits[task], labels[task]) for task in TASK_ROWS}
                loss = losses[PRIMARY_TASK] + LIFE_LOSS_WEIGHT * losses['life_stage']
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite multitask training loss.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                seen += len(life)
                total_loss += float(loss.item()) * len(life)
                for task in TASK_ROWS:
                    train_loss[task] += float(losses[task].item()) * len(life)
                    train_correct[task] += int((logits[task].argmax(1) == labels[task]).sum().item())
            model.eval()
            truth = {task: [] for task in TASK_ROWS}
            predicted = {task: [] for task in TASK_ROWS}
            probabilities = {task: [] for task in TASK_ROWS}
            dev_loss = {task: 0.0 for task in TASK_ROWS}
            with torch.inference_mode():
                for images, life, engorgement in dev_loader:
                    labels = {'life_stage': life, PRIMARY_TASK: engorgement}
                    logits = split_logits(model(images.to(device)))
                    for task in TASK_ROWS:
                        loss = validation_loss(logits[task], labels[task].to(device))
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Non-finite multitask validation loss.')
                        dev_loss[task] += float(loss.item())
                        probability = logits[task].softmax(1).cpu().numpy()
                        truth[task].extend(labels[task].tolist())
                        predicted[task].extend(probability.argmax(1).tolist())
                        probabilities[task].extend(probability.tolist())
            results = {task: validation_details(dev, truth[task], predicted[task], probabilities[task],
                                                classes[task], dev_loss[task] / len(dev))
                       for task in TASK_ROWS}
            entry = {'epoch': epoch, 'phase': 'head_warmup' if warming_up else cfg['fine_tune'],
                     'joint_training_loss': total_loss / seen,
                     'engorgement_validation_accuracy': results[PRIMARY_TASK]['accuracy'],
                     'learning_rates': {group['name']: group['lr'] for group in optimizer.param_groups},
                     'seconds': time.time() - epoch_started}
            task_summaries = {}
            for task in TASK_ROWS:
                task_entry = {
                    **entry, 'training_loss': train_loss[task] / seen,
                    'training_accuracy': train_correct[task] / seen,
                    'validation_loss': results[task]['loss'],
                    'validation_accuracy': results[task]['accuracy'],
                    'validation_balanced_accuracy': results[task]['balanced_accuracy'],
                    'validation_macro_f1': results[task]['macro_f1'],
                }
                task_history[task].append(task_entry)
                write_json(paths[task] / 'history.json', task_history[task])
                task_summaries[task] = {key: value for key, value in task_entry.items()
                                        if key.startswith(('training_', 'validation_'))}
            entry.update(task_summaries)
            history.append(entry)
            write_json(paths['summary'] / 'history.json', history)
            print(json.dumps(entry), flush=True)
            score = results[PRIMARY_TASK]['accuracy']
            if score > best_score:
                best_score, best_epoch, best_results, stale = score, epoch, results, 0
                exports = export_state_dicts(model, architecture)
                for task in TASK_ROWS:
                    metric = 'accuracy' if task == PRIMARY_TASK else 'engorgement_accuracy'
                    checkpoint = {
                        'state_dict': exports[task], 'target': task, 'classes': classes[task],
                        'architecture': architecture, 'epoch': epoch, 'preprocessing': preprocessing,
                        'image_size': cfg['image_size'], 'bottom_crop_fraction': cfg['bottom_crop_fraction'],
                        'selection_metric': metric, 'selection_score': score,
                        'selection_task': PRIMARY_TASK, 'smoke_only': smoke,
                        'multitask_run_id': run_id,
                    }
                    temporary = paths[task] / 'best.pt.tmp'
                    torch.save(checkpoint, temporary)
                    temporary.replace(paths[task] / 'best.pt')
                    results[task].update(epoch=epoch, selection_metric=metric, selection_score=score,
                                         selection_task=PRIMARY_TASK, smoke_only=smoke)
                    write_json(paths[task] / 'validation.json', results[task])
                write_json(paths['summary'] / 'validation.json', {
                    'epoch': epoch, 'selection_metric': 'engorgement_accuracy',
                    'selection_score': score, 'smoke_only': smoke, 'tasks': results,
                })
            elif not warming_up:
                stale += 1
            if scheduler is not None and not warming_up:
                scheduler.step()
            if not warming_up and epoch >= cfg['min_epochs'] and stale >= cfg['patience']:
                break
        completed = {
            'status': 'completed', 'epochs': len(history), 'smoke_only': smoke,
            'best_epoch': best_epoch, 'best_selection_score': best_score,
            'selection_task': PRIMARY_TASK, 'test_images_used': 0,
            'elapsed_seconds': time.time() - started,
        }
        for task in TASK_ROWS:
            write_json(paths[task] / 'completion.json', {
                **completed, 'selection_metric': 'accuracy' if task == PRIMARY_TASK else 'engorgement_accuracy',
                'best_validation_accuracy': best_results[task]['accuracy'],
                'best_validation_balanced_accuracy': best_results[task]['balanced_accuracy'],
                'best_validation_macro_f1': best_results[task]['macro_f1'],
            })
        write_json(paths['summary'] / 'completion.json', {
            **completed, 'selection_metric': 'engorgement_accuracy',
            'task_accuracy_at_selected_checkpoint': {task: best_results[task]['accuracy'] for task in TASK_ROWS},
        })
    except BaseException as error:
        failure = {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   'error': f'{type(error).__name__}: {error}', 'epochs': len(history),
                   'smoke_only': smoke, 'test_images_used': 0}
        for folder in paths.values():
            write_json(folder / 'completion.json', failure)
        raise
    print('Saved multitask exports:', json.dumps({task: str(path) for task, path in paths.items()}), flush=True)
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Run full pretrained training on the available device.')
    mode.add_argument('--smoke', action='store_true', help='One CPU epoch with random weights (the default).')
    arguments = parser.parse_args()
    run(arguments.config, smoke=not arguments.execute)
