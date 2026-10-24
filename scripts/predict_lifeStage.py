"""Predict adult female or nymph with an uncalibrated confidence score."""

import argparse
import json
from pathlib import Path

from common import WORK
from inference import predict


DEFAULT_CHECKPOINT = WORK / 'runs/20261013-125510-220265_life_stage/best.pt'


def predict_lifeStage(image_path, checkpoint=DEFAULT_CHECKPOINT):
    return predict(image_path, checkpoint, expected_task='life_stage')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    print(json.dumps(predict_lifeStage(args.image, args.checkpoint), indent=2))
