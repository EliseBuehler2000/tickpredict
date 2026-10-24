"""Check the exported Core ML predictors against the locked v1.0 PyTorch results.

Runs every development image through the same Python preprocessing used in
training, predicts with each .mlpackage, and compares labels and probabilities
with the release validation files. Development images only; no test images.
"""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np

from common import digest, resolve_image_path, write_json
from evaluate_predictor import _records
from export_coreml import OUTPUT, RELEASE, TASKS, WORK, _require
from image_preprocessing import prepare_image

COMPUTE_UNITS = {"all": ct.ComputeUnit.ALL, "cpu": ct.ComputeUnit.CPU_ONLY}


def verify(task, manifest, compute_units):
    records, classes, _ = _records(task)
    path = OUTPUT / f"{TASKS[task]}.mlpackage"
    model = ct.models.MLModel(str(path), compute_units=COMPUTE_UNITS[compute_units])
    metadata = model.user_defined_metadata
    _require(json.loads(metadata["classes"]) == classes, "Core ML class order differs.")
    spec = json.loads(metadata["preprocessing"])
    reference = json.loads((WORK / manifest["models"][task]["development_result"]["validation_file"]).read_text())
    _require(reference["classes"] == classes, "Reference class order differs.")
    lookup = {p["tick_id"]: p for p in reference["predictions"]}

    rows = []
    for record in records:
        image_path = resolve_image_path(record["source_path"])
        _require(digest(image_path) == record["sha256"], f"Image hash mismatch: {image_path}")
        output = model.predict({"image": prepare_image(image_path, spec)})
        probabilities = np.array([output["classLabel_probs"][c] for c in classes])
        expected = np.array(lookup[record["tick_id"]]["uncalibrated_probabilities"])
        rows.append({
            "tick_id": record["tick_id"],
            "true": record[task],
            "pytorch": classes[int(expected.argmax())],
            "coreml": output["classLabel"],
            "coreml_probabilities": probabilities.tolist(),
            "max_abs_difference": float(np.abs(probabilities - expected).max()),
        })
    differences = [r["max_abs_difference"] for r in rows]
    return {
        "task": task,
        "model": str(path.relative_to(WORK)),
        "precision": metadata.get("precision"),
        "compute_units": compute_units,
        "n": len(rows),
        "label_agreement": sum(r["coreml"] == r["pytorch"] for r in rows),
        "coreml_correct": sum(r["coreml"] == r["true"] for r in rows),
        "pytorch_correct": reference["correct"],
        "max_abs_probability_difference": max(differences),
        "mean_abs_probability_difference": float(np.mean(differences)),
        "disagreements": [r for r in rows if r["coreml"] != r["pytorch"]],
        "predictions": rows,
    }


def verify_multihead(checkpoint, package, compute_units, tta):
    """Compare a v2 Core ML export with its PyTorch checkpoint on the Penn dev photos."""
    import torch
    from multihead_inference import head_probabilities, image_tensor, load_multihead

    model, saved = load_multihead(checkpoint)
    mlmodel = ct.models.MLModel(str(package), compute_units=COMPUTE_UNITS[compute_units])
    _require(json.loads(mlmodel.user_defined_metadata["heads"]) == saved["heads"], "Head classes differ.")
    spec = saved["preprocessing"]
    records, _, _ = _records("life_stage")
    agreement = {head: 0 for head in saved["heads"]}
    largest = {head: 0.0 for head in saved["heads"]}
    for record in records:
        path = resolve_image_path(record["source_path"])
        reference = head_probabilities(model, saved, image_tensor(path, spec).unsqueeze(0), tta)
        output = mlmodel.predict({"image": prepare_image(path, spec)})
        for head in saved["heads"]:
            expected = reference[head][0].numpy()
            actual = np.asarray(output[head]).reshape(-1)
            agreement[head] += int(expected.argmax() == actual.argmax())
            largest[head] = max(largest[head], float(np.abs(expected - actual).max()))
    for head in saved["heads"]:
        print(f"{head}: {agreement[head]}/{len(records)} top classes match PyTorch, max |dp| {largest[head]:.4f}")
    return {"n": len(records), "agreement": agreement, "max_abs_probability_difference": largest,
            "compute_units": compute_units, "tta": tta, "test_images_used": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-units", choices=tuple(COMPUTE_UNITS), default="all")
    parser.add_argument("--multihead", type=Path, help="v2 checkpoint to compare with --package.")
    parser.add_argument("--package", type=Path, help="Exported v2 .mlpackage.")
    parser.add_argument("--tta", choices=("none", "d4"), default="d4")
    args = parser.parse_args()
    if args.multihead:
        _require(args.package, "--multihead needs --package.")
        result = verify_multihead(args.multihead, args.package, args.compute_units, args.tta)
        write_json(args.package.with_name(f"{args.package.stem}_verification_{args.compute_units}.json"), result)
        return
    manifest = json.loads((RELEASE / "manifest.json").read_text())
    results = {"release": manifest["release"], "split": "dev", "test_images_used": 0,
               "tasks": [verify(task, manifest, args.compute_units) for task in TASKS]}
    destination = OUTPUT / f"verification_{args.compute_units}.json"
    write_json(destination, results)
    for result in results["tasks"]:
        print(f"{result['task']}: {result['label_agreement']}/{result['n']} labels match PyTorch; "
              f"Core ML {result['coreml_correct']}/{result['n']} correct "
              f"(PyTorch {result['pytorch_correct']}); max |dp| "
              f"{result['max_abs_probability_difference']:.4f}, mean "
              f"{result['mean_abs_probability_difference']:.4f}")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
