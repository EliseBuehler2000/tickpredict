"""Preview or evaluate an explicitly specified predictor on development data only.

Each checkpoint and each optional D4 view receives equal probability weight.
No model subsets, ensemble weights, or confidence thresholds are fitted. Paths
in a spec are project-relative or absolute. Saved-prediction mode reads no images.
"""

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from common import WORK, digest, resolve_image_path, rows, write_json
from experiment import validation_details, verified_records
from inference import EXPECTED_CLASSES, NOTES, preprocess_image
from modeling import ARCHITECTURES, build_model, preprocessing_spec


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_probabilities(probabilities, *, expected_rows=None, expected_classes=None,
                            context="Predictions"):
    """Reject invalid distributions; never repair or normalize a corrupt result."""
    if isinstance(probabilities, torch.Tensor):
        _require(probabilities.device.type == "cpu", f"{context}: validation requires CPU probabilities.")
        values = probabilities.detach().numpy()
    else:
        values = np.asarray(probabilities)
    _require(values.ndim == 2 and values.shape[0] > 0 and values.shape[1] > 1,
             f"{context}: probabilities must be a nonempty rows-by-classes matrix.")
    _require(expected_rows is None or values.shape[0] == expected_rows,
             f"{context}: probability row count mismatch.")
    _require(expected_classes is None or values.shape[1] == expected_classes,
             f"{context}: probability class count mismatch.")
    _require(bool(np.isfinite(values).all()), f"{context}: nonfinite probabilities.")
    _require(bool((values >= 0).all() and (values <= 1).all()),
             f"{context}: probabilities must lie in [0, 1].")
    _require(bool(np.allclose(values.sum(axis=1, dtype=np.float64), 1.0,
                              rtol=0, atol=1e-5)),
             f"{context}: probability rows must sum to one.")


def _json(path):
    return json.loads(Path(path).read_text())


def load_spec(spec):
    """Normalize an explicit dict or JSON file; reject accidental weighting knobs."""
    spec = dict(spec) if isinstance(spec, dict) else _json(Path(spec).expanduser())
    _require(not set(spec) - {"target", "checkpoints", "prediction_files", "tta"},
             "Spec accepts only target, checkpoints or prediction_files, and tta.")
    _require(spec.get("target") in EXPECTED_CLASSES, "Unsupported predictor target.")
    keys = [key for key in ("checkpoints", "prediction_files") if key in spec]
    _require(len(keys) == 1, "Specify exactly one of checkpoints or prediction_files.")
    key = keys[0]
    _require(isinstance(spec[key], list) and spec[key]
             and all(isinstance(path, str) and path for path in spec[key]),
             "Inputs must be an explicit nonempty list of paths.")
    paths = []
    for value in spec[key]:
        path = Path(value).expanduser()
        path = (path if path.is_absolute() else WORK / path).resolve()
        _require(path.is_file(), f"Missing input file: {path}")
        paths.append(str(path))
    _require(len(set(paths)) == len(paths), "Duplicate inputs would change equal model weights.")
    tta = spec.get("tta", "none")
    _require(tta in ("none", "d4"), "tta must be none or d4.")
    _require(key != "prediction_files" or tta == "none",
             "D4 requires checkpoints and images; it cannot be added to cached predictions.")
    return {"target": spec["target"], key: paths, "tta": tta}


def _records(target):
    """Check only the approved train/dev manifest; do not open source CSV/images."""
    preflight = _json(WORK / "notes/preflight.json")
    manifest = WORK / "data/penn_train_dev.csv"
    _require(digest(manifest) == preflight["manifest_sha256"], "Manifest fingerprint changed.")
    records = rows(manifest)
    _require(Counter(r["provisional_split"] for r in records) == {"train": 241, "dev": 110},
             "Expected exactly 241 train and 110 dev records, with no test records.")
    _require(len({r["tick_id"] for r in records}) == len(records), "Duplicate manifest IDs.")
    dev = [r for r in records if r["provisional_split"] == "dev"]
    classes = sorted(EXPECTED_CLASSES[target])
    _require({r[target] for r in dev} == set(classes), "Development labels differ from task.")
    return dev, classes, preflight


def _source_metadata(path, target, preflight=None):
    """Require completed non-smoke sources and matching data provenance."""
    directory = Path(path).parent
    provenance = _json(directory / "provenance.json")
    completion = _json(directory / "completion.json")
    _require(completion.get("status") == "completed", f"Source is not completed: {directory}")
    _require(completion.get("smoke_only") is False and provenance.get("smoke_only") is False,
             f"Smoke or unspecified source is not evaluable: {directory}")
    _require(provenance.get("config", {}).get("target") == target, "Source task mismatch.")
    _require(provenance.get("test_images_used") == 0, "Source must record zero test images used.")
    if preflight is not None:
        for key in ("source_sha256", "manifest_sha256"):
            _require(provenance.get(key) == preflight[key], f"Source {key} mismatch.")
        _require(provenance.get("validation_n") == 110, "Source is not a full development run.")
    classes = provenance.get("classes")
    _require(isinstance(classes, list) and len(classes) == len(EXPECTED_CLASSES[target])
             and all(isinstance(c, str) for c in classes)
             and set(classes) == EXPECTED_CLASSES[target], "Invalid source class labels.")
    return provenance


def _checkpoint(path, target, preflight=None):
    provenance = _source_metadata(path, target, preflight)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(saved, dict) and not saved.get("smoke_only"), "Invalid or smoke checkpoint.")
    _require(saved.get("target") == target and saved.get("classes") == provenance["classes"],
             "Checkpoint task/classes differ from provenance.")
    architecture = saved.get("architecture")
    expected_architecture = provenance.get("config", {}).get(
        "architecture", provenance.get("architecture", "resnet18"))
    _require(architecture in ARCHITECTURES and architecture == expected_architecture,
             "Checkpoint architecture differs from supported provenance.")
    preprocessing = saved.get("preprocessing", preprocessing_spec())
    _require(isinstance(preprocessing, dict), "Missing checkpoint preprocessing recipe.")
    size, crop = preprocessing.get("image_size"), preprocessing.get("bottom_crop_fraction")
    _require(type(size) is int and size >= 32 and type(crop) in (int, float)
             and math.isfinite(crop) and 0 <= crop < 0.5,
             "Unsupported checkpoint image size or strip crop.")
    _require(preprocessing == preprocessing_spec(size, crop, preprocessing.get("foreground_crop")),
             "Unsupported preprocessing pipeline.")
    if isinstance(provenance.get("preprocessing"), dict):
        _require(preprocessing == provenance["preprocessing"], "Preprocessing differs from provenance.")
    else:
        _require(architecture == "resnet18" and preprocessing == preprocessing_spec(),
                 "Only historic ResNet18 permits the legacy preprocessing default.")
    information = {"checkpoint": str(Path(path).resolve()), "sha256": digest(path),
                   "architecture": architecture, "epoch": saved.get("epoch"),
                   "classes": saved["classes"], "preprocessing": preprocessing}
    return saved, information


def d4_views(batch):
    """All four rotations times two reflection states, without interpolation."""
    _require(batch.ndim == 4 and batch.shape[-1] == batch.shape[-2],
             "D4 expects square NCHW tensors after the saved preprocessing.")
    return [torch.rot90(batch, k, dims=(-2, -1)).flip(-1) if reflected
            else torch.rot90(batch, k, dims=(-2, -1))
            for reflected in (False, True) for k in range(4)]


def _view_probabilities(model, batch, tta, device, batch_size):
    views = d4_views(batch) if tta == "d4" else [batch]
    inputs = torch.cat(views)
    outputs = []
    with torch.inference_mode():
        for chunk in inputs.split(batch_size):
            # Keep transfer and cast separate: the combined MPS-to-CPU float64
            # conversion can silently return zeros on the current backend.
            probabilities = model(chunk.to(device)).softmax(1).cpu().to(dtype=torch.float64)
            _validate_probabilities(probabilities, expected_rows=len(chunk), context="Model view output")
            outputs.append(probabilities)
    averaged = torch.cat(outputs).reshape(len(views), len(batch), -1).mean(0)
    _validate_probabilities(averaged, expected_rows=len(batch), context="View average")
    return averaged


def _device(value):
    _require(value in ("cpu", "mps", "cuda"), "device must be cpu, mps, or cuda.")
    _require(value != "mps" or torch.backends.mps.is_available(), "MPS is not available.")
    _require(value != "cuda" or torch.cuda.is_available(), "CUDA is not available.")
    return torch.device(value)


def _predict_paths(paths, spec, device, batch_size, preflight=None):
    classes = sorted(EXPECTED_CLASSES[spec["target"]])
    total = torch.zeros((len(paths), len(classes)), dtype=torch.float64)
    model_information = []
    # Process one model at a time so ensembles do not multiply GPU model memory.
    for checkpoint in spec["checkpoints"]:
        saved, information = _checkpoint(checkpoint, spec["target"], preflight)
        model = build_model(information["architecture"], len(classes), pretrained=False)
        model.load_state_dict(saved["state_dict"])
        del saved
        model.eval().to(device)
        order = [information["classes"].index(c) for c in classes]
        for start in range(0, len(paths), batch_size):
            batch = torch.stack([preprocess_image(path, information["preprocessing"])
                                 for path in paths[start:start + batch_size]])
            probability = _view_probabilities(model, batch, spec["tta"], device, batch_size)
            _validate_probabilities(probability, expected_rows=len(batch),
                                    expected_classes=len(classes), context="Model prediction")
            total[start:start + len(batch)] += probability[:, order]
        model_information.append(information)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    probabilities = (total / len(spec["checkpoints"])).numpy()
    _validate_probabilities(probabilities, expected_rows=len(paths),
                            expected_classes=len(classes), context="Ensemble prediction")
    return probabilities, model_information


def predict_image(image_path, spec, device="cpu", batch_size=16):
    """Apply the exact checkpoint/D4 spec to one image; no dataset is required."""
    spec = load_spec(spec)
    _require("checkpoints" in spec, "Single-image inference requires a checkpoint spec.")
    _require(type(batch_size) is int and batch_size > 0, "batch_size must be positive.")
    probabilities, information = _predict_paths([image_path], spec, _device(device), batch_size)
    classes = sorted(EXPECTED_CLASSES[spec["target"]])
    _validate_probabilities(probabilities, expected_rows=1, expected_classes=len(classes),
                            context="Single-image prediction")
    index = int(probabilities[0].argmax())
    return {"task": spec["target"], "prediction": classes[index],
            "confidence": float(probabilities[0, index]), "confidence_calibrated": False,
            "model_scores_uncalibrated": dict(zip(classes, probabilities[0].tolist())),
            "predictor_spec": spec, "models": information, "note": NOTES[spec["target"]]}


def _cached_probabilities(files, target, records, classes, preflight):
    ids = [r["tick_id"] for r in records]
    totals = np.zeros((len(records), len(classes)), dtype=np.float64)
    information = []
    for filename in files:
        path = Path(filename)
        provenance = _source_metadata(path, target, preflight)
        validation = _json(path)
        _require(not validation.get("smoke_only"), "Smoke predictions cannot be evaluated.")
        source_classes = validation.get("classes")
        _require(source_classes == provenance["classes"], "Prediction class order differs from provenance.")
        predictions = validation.get("predictions", [])
        _require(len(predictions) == len(ids), "Expected all 110 development predictions.")
        lookup = {str(p["tick_id"]): p for p in predictions}
        _require(len(lookup) == len(predictions) and set(lookup) == set(ids),
                 "Prediction IDs must match development IDs exactly, with no duplicates.")
        ordered = []
        for record in records:
            prediction = lookup[record["tick_id"]]
            _require(prediction["true"] == record[target], "Prediction truth differs from the manifest.")
            probability = np.asarray(prediction["uncalibrated_probabilities"], dtype=np.float64)
            _require(probability.shape == (len(classes),) and np.isfinite(probability).all()
                     and (probability >= 0).all() and (probability <= 1).all()
                     and abs(float(probability.sum()) - 1) <= 1e-5, "Invalid source probabilities.")
            _require(prediction["predicted"] == source_classes[int(probability.argmax())],
                     "Source prediction disagrees with probability argmax.")
            ordered.append(probability[[source_classes.index(c) for c in classes]])
        totals += np.asarray(ordered)
        entry = {"prediction_file": str(path), "sha256": digest(path),
                 "source_provenance_sha256": digest(path.parent / "provenance.json")}
        checkpoint = path.parent / "best.pt"
        if checkpoint.is_file():
            entry["checkpoint"] = str(checkpoint)
            entry["checkpoint_sha256"] = digest(checkpoint)
        else:
            entry["upstream_models"] = provenance.get("inputs", [])
        information.append(entry)
    probabilities = totals / len(files)
    _validate_probabilities(probabilities, expected_rows=len(records),
                            expected_classes=len(classes), context="Cached ensemble prediction")
    return probabilities, information


def _details(records, probabilities, classes, target):
    _validate_probabilities(probabilities, expected_rows=len(records),
                            expected_classes=len(classes), context="Evaluation metrics")
    truth = np.array([classes.index(r[target]) for r in records])
    predicted = probabilities.argmax(1)
    confidence = probabilities.max(1)
    correct = predicted == truth
    nll = float(-np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-12, 1)).mean())
    result = validation_details(records, truth.tolist(), predicted.tolist(),
                                probabilities.tolist(), classes, nll)
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(abs(float(correct[bins == i].sum() - confidence[bins == i].sum()))
              for i in range(10)) / len(truth)
    result.update(evaluation_split="dev", confidence_calibrated=False, nll=nll,
                  brier=float(((probabilities - np.eye(len(classes))[truth]) ** 2).sum(1).mean()),
                  ece10=ece, mean_confidence=float(confidence.mean()),
                  mean_confidence_when_wrong=float(confidence[~correct].mean()) if (~correct).any() else None,
                  meets_development_target=result["accuracy"] > 0.95,
                  smoke_only=False, test_images_used=0)
    return result


def evaluate_spec(spec, execute=False, device="cpu", batch_size=16):
    """Preview by default; execution writes separate development evaluation artifacts."""
    spec = load_spec(spec)
    _require(type(batch_size) is int and batch_size > 0, "batch_size must be positive.")
    selected_device = _device(device)
    records, classes, preflight = _records(spec["target"])
    key = "checkpoints" if "checkpoints" in spec else "prediction_files"
    for path in spec[key]:
        _source_metadata(path, spec["target"], preflight)
    plan = {"predictor_spec": spec, "evaluation_split": "dev", "validation_n": len(records),
            "device": "none (cached probabilities)" if key == "prediction_files" else device,
            "batch_size": batch_size, "equal_input_weight": 1 / len(spec[key]),
            "views_per_model": 8 if spec["tta"] == "d4" else 1,
            "test_images_used": 0, "status": "preview", "execute_required": True}
    if not execute:
        return plan
    torch.set_num_threads(4)
    started = time.time()
    if key == "checkpoints":
        _, verified_dev, verified_classes, source_hash, manifest_hash = verified_records(spec["target"])
        _require(verified_dev == records and verified_classes == classes
                 and source_hash == preflight["source_sha256"]
                 and manifest_hash == preflight["manifest_sha256"], "Verified data changed.")
    out = WORK / "experiments/evaluations" / (
        datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "_" + spec["target"] + "_" + spec["tta"])
    out.mkdir(parents=True)
    write_json(out / "predictor_spec.json", spec)
    write_json(out / "completion.json", {"status": "running", "smoke_only": False, "test_images_used": 0})
    try:
        if key == "checkpoints":
            probabilities, inputs = _predict_paths(
                [resolve_image_path(r["source_path"]) for r in records],
                spec, selected_device, batch_size, preflight)
        else:
            probabilities, inputs = _cached_probabilities(spec[key], spec["target"], records, classes, preflight)
        provenance = {
            "config": {"target": spec["target"]}, "predictor_spec": spec, "inputs": inputs,
            "classes": classes, "validation_n": len(records), "evaluation_split": "dev",
            "source_sha256": preflight["source_sha256"], "manifest_sha256": preflight["manifest_sha256"],
            "device": plan["device"], "batch_size": batch_size, "torch": str(torch.__version__),
            "script_sha256": digest(__file__), "inference_sha256": digest(Path(__file__).with_name("inference.py")),
            "modeling_sha256": digest(Path(__file__).with_name("modeling.py")),
            "image_preprocessing_sha256": digest(Path(__file__).with_name("image_preprocessing.py")),
            "smoke_only": False, "test_images_used": 0, "weights_fitted": False,
            "aggregation": "Equal probability average over views within each model, then over explicit inputs.",
            "d4_definition": "After saved crop/letterbox/normalization: rot90 k=0,1,2,3; each with/without horizontal flip.",
            "caveat": "Development data reused for selection; not an independent test estimate. Confidence remains uncalibrated.",
        }
        result = _details(records, probabilities, classes, spec["target"])
        write_json(out / "provenance.json", provenance)
        write_json(out / "validation.json", result)
        write_json(out / "completion.json", {"status": "completed", "smoke_only": False,
                   "validation_accuracy": result["accuracy"], "elapsed_seconds": time.time() - started,
                   "test_images_used": 0})
    except BaseException as error:
        write_json(out / "completion.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   "error": f"{type(error).__name__}: {error}", "smoke_only": False, "test_images_used": 0})
        raise
    return {"output": str(out), "evaluation_split": "dev", "correct": result["correct"],
            "n": result["n"], "accuracy": result["accuracy"], "balanced_accuracy": result["balanced_accuracy"],
            "nll": result["nll"], "ece10": result["ece10"], "test_images_used": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--spec", type=Path)
    inputs.add_argument("--checkpoints", nargs="+")
    inputs.add_argument("--prediction-files", nargs="+")
    parser.add_argument("--target", choices=sorted(EXPECTED_CLASSES))
    parser.add_argument("--tta", choices=("none", "d4"))
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--execute", action="store_true", help="Run evaluation; default only previews the explicit spec.")
    args = parser.parse_args()
    if args.spec:
        if args.target or args.tta:
            parser.error("--spec cannot be combined with --target or --tta overrides.")
        spec = args.spec
    else:
        if not args.target:
            parser.error("--target is required with explicit input paths.")
        key = "checkpoints" if args.checkpoints else "prediction_files"
        spec = {"target": args.target, key: args.checkpoints or args.prediction_files, "tta": args.tta or "none"}
    print(json.dumps(evaluate_spec(spec, execute=args.execute, device=args.device, batch_size=args.batch_size), indent=2))


if __name__ == "__main__":
    main()
