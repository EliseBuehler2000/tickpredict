"""Write reference inputs for checking the iOS app's Swift preprocessing.

The app runs the v2.0 multi-head model (species, life stage) and the v1.0
engorgement model. For every development image, saves the exact
Python-prepared input of each (raw RGB bytes) and their PyTorch probabilities. Also writes JPEG
copies carrying each EXIF orientation 2-8, with their Python-prepared inputs,
because no development photo is rotated. Development images only.

    .venv/bin/python scripts/ios_parity_fixtures.py OUTPUT_DIR
    swiftc -O ... ios/Tools/ParityCheck/main.swift  (see ios/README.md)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import WORK, digest, resolve_image_path
from evaluate_predictor import _records
from export_coreml import RELEASE, TASKS
from image_preprocessing import prepare_image
from multihead_inference import head_probabilities, image_tensor, load_multihead

ORIENTATION_SOURCES = 3
MULTIHEAD = WORK / "models" / "v2.0" / "multihead" / "best.pt"
MULTIHEAD_HEADS = ("species", "life_stage")


def _spec(task):
    from export_coreml import load_predictor
    manifest = json.loads((RELEASE / "manifest.json").read_text())
    _, saved, _, _ = load_predictor(task, manifest)
    return saved["preprocessing"]


def _multihead_references(records):
    """The v2 model's eight-view probabilities, as the exported Core ML model computes them."""
    model, saved = load_multihead(MULTIHEAD)
    references = {}
    for record in records:
        batch = image_tensor(resolve_image_path(record["source_path"]), saved["preprocessing"]).unsqueeze(0)
        probabilities = head_probabilities(model, saved, batch, tta="d4")
        references[record["tick_id"]] = {
            head: dict(zip(saved["heads"][head], probabilities[head][0].tolist())) for head in MULTIHEAD_HEADS}
    return saved["preprocessing"], references


def _save_prepared(image_path, specs, directory, stem):
    files = {}
    for task, spec in specs.items():
        prepared = np.asarray(prepare_image(image_path, spec), dtype=np.uint8)
        target = directory / f"{stem}_{task}.rgb"
        target.write_bytes(prepared.tobytes())
        files[task] = str(target)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RELEASE / "manifest.json").read_text())
    task = "engorgement_level"
    validation = json.loads((WORK / manifest["models"][task]["development_result"]["validation_file"]).read_text())
    engorgement = {p["tick_id"]: dict(zip(validation["classes"], p["uncalibrated_probabilities"]))
                   for p in validation["predictions"]}

    records, _, _ = _records("life_stage")
    torch.set_grad_enabled(False)
    multihead_spec, multihead = _multihead_references(records)
    specs = {"multihead": multihead_spec, task: _spec(task)}
    entries = []
    for record in records:
        path = resolve_image_path(record["source_path"])
        assert digest(path) == record["sha256"], path
        entries.append({
            "id": record["tick_id"], "image": str(path),
            "prepared": _save_prepared(path, specs, output, record["tick_id"]),
            "reference": {**multihead[record["tick_id"]], task: engorgement[record["tick_id"]]},
            # Penn photos are all blacklegged (deer) ticks.
            "truth": {**{t: record[t] for t in TASKS}, "species": "Ixodes scapularis"},
        })

    for record in records[:ORIENTATION_SOURCES]:
        source = Image.open(resolve_image_path(record["source_path"])).convert("RGB")
        for orientation in range(2, 9):
            stem = f"{record['tick_id']}_exif{orientation}"
            path = output / f"{stem}.jpg"
            exif = Image.Exif()
            exif[0x0112] = orientation
            source.save(path, quality=95, exif=exif)
            entries.append({"id": stem, "image": str(path),
                            "prepared": _save_prepared(path, specs, output, stem)})

    (output / "fixtures.json").write_text(json.dumps({
        "release": "v2.0 + engorgement v1.0", "specs": specs, "entries": entries}, indent=1))
    print(f"Wrote {len(entries)} fixtures to {output}")


if __name__ == "__main__":
    main()
