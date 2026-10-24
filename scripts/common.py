"""Project paths and data helpers; model inference needs no training data."""
from pathlib import Path
import csv
import hashlib
import json
import os

WORK = Path(__file__).resolve().parents[1]
LOCAL_SOURCE = WORK / 'dataset/Penn_Dataset/prepared_data.csv'
LOCAL_IMAGES = WORK / 'dataset/Penn Data/433 tick photos'
SOURCE = Path(os.environ.get('TICKPREDICT_SOURCE_CSV', LOCAL_SOURCE if LOCAL_SOURCE.is_file() else WORK / 'data/prepared_data.csv')).expanduser()
WEIGHTS = Path(os.environ.get('TICKPREDICT_WEIGHTS', WORK / 'data/weights/resnet18-f37072fd.pth')).expanduser()


def resolve_image_path(source_path):
    """Relocate photos without changing the fingerprinted manifest or its splits."""
    image_dir = os.environ.get('TICKPREDICT_IMAGE_DIR') or (LOCAL_IMAGES if LOCAL_IMAGES.is_dir() else None)
    if image_dir:
        return Path(image_dir).expanduser() / Path(source_path).name
    path = Path(source_path).expanduser()
    return path if path.is_absolute() else WORK / path


def require_training_files():
    for path, variable in [(SOURCE, 'TICKPREDICT_SOURCE_CSV'), (WEIGHTS, 'TICKPREDICT_WEIGHTS')]:
        if not path.is_file():
            raise FileNotFoundError(f'Missing training input: {path}. Set {variable} to the original file; see README.md.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')
