"""Check the Android ONNX models against PyTorch on the development photos.

Feeds each Python-prepared input from scripts/ios_parity_fixtures.py to
ONNX Runtime and compares every head with the PyTorch reference.
Development images only.

    .venv/bin/python scripts/ios_parity_fixtures.py FIXTURES
    .venv/bin/python scripts/verify_onnx.py FIXTURES
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from export_onnx import OUTPUT


def session(name):
    model = ort.InferenceSession(str(OUTPUT / name), providers=["CPUExecutionProvider"])
    return model, model.get_modelmeta().custom_metadata_map


def tensor(path, size):
    pixels = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8).reshape(size, size, 3)
    return pixels.transpose(2, 0, 1)[None].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixtures", type=Path)
    args = parser.parse_args()
    fixtures = json.loads((args.fixtures / "fixtures.json").read_text())
    multihead, multihead_meta = session("TickPredictV2.onnx")
    engorgement, engorgement_meta = session("EngorgementLevel.onnx")
    heads = json.loads(multihead_meta["heads"])
    classes = json.loads(engorgement_meta["classes"])
    size = json.loads(multihead_meta["preprocessing"])["image_size"]

    results = {head: {"agree": 0, "correct": 0, "max_dp": 0.0} for head in ("species", "life_stage", "engorgement_level")}
    entries = [entry for entry in fixtures["entries"] if "reference" in entry]
    for entry in entries:
        outputs = dict(zip([o.name for o in multihead.get_outputs()],
                           multihead.run(None, {"image": tensor(entry["prepared"]["multihead"], size)})))
        scores = {head: dict(zip(heads[head], outputs[head][0].tolist())) for head in ("species", "life_stage")}
        (probabilities,) = engorgement.run(None, {"image": tensor(entry["prepared"]["engorgement_level"], size)})
        scores["engorgement_level"] = dict(zip(classes, probabilities[0].tolist()))
        for head, result in results.items():
            ours, reference = scores[head], entry["reference"][head]
            label = max(ours, key=ours.get)
            result["agree"] += label == max(reference, key=reference.get)
            result["correct"] += label == entry["truth"][head]
            result["max_dp"] = max(result["max_dp"], max(abs(ours[c] - reference[c]) for c in reference))
    for head, result in results.items():
        print(f"{head}: {result['agree']}/{len(entries)} agree with PyTorch, {result['correct']} correct, "
              f"max |dp| {result['max_dp']:.4f}")


if __name__ == "__main__":
    main()
