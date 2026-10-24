"""Run a predefined development comparison; completed matching runs are reused."""
import argparse
import hashlib
import json
from pathlib import Path

from common import WORK, SOURCE, digest


def planned_configs(plan_path, tasks=None, recipes=None):
    plan = json.loads(Path(plan_path).read_text())
    tasks = list(plan['tasks'] if tasks is None else tasks)
    names = [recipe['name'] for recipe in plan['recipes']]
    recipes = list(names if recipes is None else recipes)
    if not tasks or not set(tasks) <= set(plan['tasks']) or len(set(tasks)) != len(tasks):
        raise ValueError('Choose unique tasks from the experiment plan.')
    if not recipes or not set(recipes) <= set(names) or len(set(recipes)) != len(recipes):
        raise ValueError('Choose unique recipe names from the experiment plan.')
    configurations = []
    for task in tasks:
        for recipe in plan['recipes']:
            if recipe['name'] in recipes:
                configurations.append({**plan['shared'], **recipe, 'target': task})
    return configurations


def _key(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def run_plan(plan_path, tasks=None, recipes=None, execute=False, smoke=False):
    configurations = planned_configs(plan_path, tasks, recipes)
    if not execute:
        return configurations
    # Validate availability before importing the GPU training module or downloading weights.
    if not SOURCE.is_file():
        raise FileNotFoundError(f'Missing source CSV: {SOURCE}')
    fingerprints = {'source_sha256': digest(SOURCE),
                    'manifest_sha256': digest(WORK / 'data/penn_train_dev.csv'),
                    'script_sha256': digest(WORK / 'scripts/experiment.py'),
                    'modeling_sha256': digest(WORK / 'scripts/modeling.py'),
                    'image_preprocessing_sha256': digest(WORK / 'scripts/image_preprocessing.py')}
    from experiment import load_config, run
    configurations = [load_config(config) for config in configurations]
    completed = {}
    if not smoke:
        for folder in (WORK / 'experiments/runs').glob('*'):
            if not (folder / 'completion.json').is_file():
                continue
            status = json.loads((folder / 'completion.json').read_text())
            if (status.get('status') != 'completed' or status.get('smoke_only')
                    or not (folder / 'provenance.json').is_file()):
                continue
            provenance = json.loads((folder / 'provenance.json').read_text())
            if (status.get('status') == 'completed'
                    and not status.get('smoke_only') and not provenance.get('smoke_only')
                    and (folder / 'best.pt').is_file()
                    and all(provenance.get(k) == v for k, v in fingerprints.items())):
                completed[_key(provenance['config'])] = folder
    results = []
    for config in configurations:
        previous = completed.get(_key(config))
        if previous is not None:
            print('Reusing completed run:', previous, flush=True)
            folder = previous
        else:
            print(f"Starting {config['target']} / {config['name']}", flush=True)
            folder = run(config, smoke=smoke)
        results.append(folder)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=WORK / 'configs/experiments.json')
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--recipes', nargs='+')
    parser.add_argument('--execute', action='store_true', help='Actually start training; default only prints planned configs')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    results = run_plan(args.plan, args.tasks, args.recipes, args.execute, args.smoke)
    if not args.execute:
        print(json.dumps(results, indent=2))
    else:
        print(json.dumps([str(path) for path in results], indent=2))
