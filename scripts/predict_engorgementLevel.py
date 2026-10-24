"""Predict a Penn engorgement category with an uncalibrated confidence score."""

import argparse
import json
from pathlib import Path

from common import WORK
from inference import predict


RUN = WORK / 'runs/20261013-125824-749104_engorgement_level'
DEFAULT_CHECKPOINT = RUN / 'best.pt'


def predict_engorgementLevel(image_path, checkpoint=DEFAULT_CHECKPOINT):
    return predict(image_path, checkpoint, expected_task='engorgement_level')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    print(json.dumps(predict_engorgementLevel(args.image, args.checkpoint), indent=2))
