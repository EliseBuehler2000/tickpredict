"""Compare completed runs on the unchanged Penn development split.

This module reads metadata and predictions only. It never opens images, model
checkpoints, or the source CSV containing protected test labels. All rates,
including ECE and mean confidence, are fractions between zero and one.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
BASELINE_RUNS = {
    "life_stage": "runs/20261013-125510-220265_life_stage",
    "engorgement_level": "runs/20261013-125824-749104_engorgement_level",
}
EXPECTED_CLASSES = {
    "life_stage": {"adult female", "nymph"},
    "engorgement_level": {"fully engorged", "semi-engorged", "unengorged"},
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(path.read_text())


def _finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def _checkpoint_epoch(validation, history, selection_metric, measured):
    for key in ("checkpoint_epoch", "epoch", "best_epoch"):
        if key in validation:
            epoch = validation[key]
            _require(type(epoch) is int and epoch > 0,
                     f"Invalid validation {key}.")
            _require(any(item.get("epoch") == epoch for item in history),
                     "Saved checkpoint epoch is absent from history.")
            return epoch, "validation_metadata"
    metric = "validation_" + selection_metric.removeprefix("validation_")
    _require(metric in measured, f"Unsupported selection metric: {selection_metric}")
    _require(history and all(_finite(item.get(metric)) for item in history),
             f"History is missing numeric {metric} values.")
    # Historic training saved only strict improvements, so first tied maximum
    # identifies its checkpoint. New runs should save the exact epoch directly.
    selected = max(history, key=lambda item: item[metric])
    _require(math.isclose(selected[metric], measured[metric], abs_tol=1e-9),
             "History selection does not match saved validation predictions.")
    epoch = selected.get("epoch")
    _require(type(epoch) is int and epoch > 0, "Invalid history epoch.")
    return epoch, "history_selection_metric"


def _run_result(run, root, preflight, dev):
    completion = _read(run / "completion.json")
    if completion.get("status") != "completed" or completion.get("smoke_only") is True:
        return None
    provenance = _read(run / "provenance.json")
    if provenance.get("smoke_only") is True:
        return None
    _require(completion.get("smoke_only") is False
             and provenance.get("smoke_only") is False,
             "Run must explicitly identify itself as non-smoke.")
    for key in ("source_sha256", "manifest_sha256"):
        _require(provenance.get(key) == preflight[key], f"Mismatched {key}.")
    _require(provenance.get("test_images_used") == 0,
             "Only runs recording zero test images used may be compared.")
    config = provenance.get("config", {})
    target = config.get("target")
    _require(target in EXPECTED_CLASSES, f"Unsupported target: {target!r}")
    validation = _read(run / "validation.json")
    classes = validation.get("classes")
    _require(isinstance(classes, list)
             and all(isinstance(label, str) for label in classes)
             and len(classes) == len(set(classes))
             and set(classes) == EXPECTED_CLASSES[target],
             "Validation classes do not match the task.")
    _require(classes == provenance.get("classes"),
             "Validation/provenance class order differs.")
    predictions = validation.get("predictions")
    _require(isinstance(predictions, list) and len(predictions) == len(dev),
             "Saved predictions must cover every development record exactly once.")
    ids = [str(item.get("tick_id")) for item in predictions]
    _require(len(ids) == len(set(ids)) and set(ids) == set(dev),
             "Prediction IDs differ from the exact development IDs.")
    _require(provenance.get("validation_n") == len(dev),
             "Provenance validation count differs from the development manifest.")

    confusion = [[0 for _ in classes] for _ in classes]
    bins = defaultdict(list)
    subgroups = defaultdict(list)
    confidences, wrong_confidences = [], []
    nll = brier = 0.0
    for item, tick_id in zip(predictions, ids):
        record = dev[tick_id]
        truth, predicted = item.get("true"), item.get("predicted")
        _require(truth == record[target] and predicted in classes,
                 f"Labels differ from the manifest or task for {tick_id}.")
        probabilities = item.get("uncalibrated_probabilities")
        _require(isinstance(probabilities, list) and len(probabilities) == len(classes)
                 and all(_finite(p) and 0 <= p <= 1 for p in probabilities)
                 and math.isclose(sum(probabilities), 1.0, abs_tol=1e-5),
                 f"Invalid probability vector for {tick_id}.")
        _require(classes[max(range(len(classes)), key=probabilities.__getitem__)] == predicted,
                 f"Predicted class disagrees with probability argmax for {tick_id}.")
        true_index, predicted_index = classes.index(truth), classes.index(predicted)
        confusion[true_index][predicted_index] += 1
        correct = truth == predicted
        confidence = max(probabilities)
        confidences.append(confidence)
        if not correct:
            wrong_confidences.append(confidence)
        bins[min(int(confidence * 10), 9)].append((correct, confidence))
        subgroups[(record["life_stage"], record["engorgement_level"])].append(correct)
        nll -= math.log(max(probabilities[true_index], 1e-12))
        brier += sum((p - (i == true_index)) ** 2 for i, p in enumerate(probabilities))

    if "confusion_matrix" in validation:
        _require(confusion == validation["confusion_matrix"],
                 "Saved confusion matrix differs from predictions.")
    n = len(predictions)
    correct = sum(confusion[i][i] for i in range(len(classes)))
    per_class = {}
    f1_values = []
    for i, label in enumerate(classes):
        support = sum(confusion[i])
        _require(support > 0, f"Development split has no {label} observations.")
        predicted_count = sum(row[i] for row in confusion)
        per_class[label] = {"n": support, "correct": confusion[i][i],
                            "recall": confusion[i][i] / support}
        f1_values.append(2 * confusion[i][i] / (support + predicted_count))
    measured = {
        "validation_accuracy": correct / n,
        "validation_balanced_accuracy": sum(c["recall"] for c in per_class.values()) / len(classes),
        "validation_macro_f1": sum(f1_values) / len(classes),
        "validation_nll": nll / n,
        "validation_brier": brier / n,
        "validation_ece10": sum(abs(sum(a for a, _ in group)
                                    - sum(c for _, c in group)) for group in bins.values()) / n,
    }
    history = _read(run / "history.json")
    _require(isinstance(history, list) and history, "Epoch history is missing.")
    selection_metric = provenance.get("selection_metric", config.get("selection_metric", "balanced_accuracy"))
    _require(isinstance(selection_metric, str), "Invalid selection metric.")
    selection_metric = selection_metric.removeprefix("validation_")
    epoch, epoch_source = _checkpoint_epoch(validation, history, selection_metric, measured)
    relative = run.relative_to(root).as_posix()
    return {
        "run": relative,
        "target": target,
        "architecture": config.get("architecture", provenance.get("architecture", "resnet18")),
        "seed": config.get("seed"),
        "selection_metric": selection_metric,
        "checkpoint_epoch": epoch,
        "checkpoint_epoch_source": epoch_source,
        "evaluation_split": "dev",
        "validation_n": n,
        "correct": correct,
        **measured,
        "mean_confidence": sum(confidences) / n,
        "mean_confidence_when_wrong": (sum(wrong_confidences) / len(wrong_confidences)
                                       if wrong_confidences else None),
        "per_class": per_class,
        "joint_subgroups": [
            {"life_stage": stage, "engorgement_level": engorgement,
             "n": len(values), "correct": sum(values), "accuracy": sum(values) / len(values)}
            for (stage, engorgement), values in sorted(subgroups.items())
        ],
        "is_pinned_baseline": relative == BASELINE_RUNS[target],
        "meets_development_target": correct / n > 0.95,
        "test_images_used": 0,
    }


def collect_results(root=WORK):
    """Return validated metrics for completed, non-smoke development runs.

    Unfinished runs and nonexistent experiment directories are skipped. Invalid
    completed runs raise ValueError instead of silently entering a comparison.
    Promotion eligibility additionally requires baseline accuracy or better and
    no class losing more than one correct development image versus baseline.
    """
    root = Path(root).expanduser().resolve()
    preflight = _read(root / "notes/preflight.json")
    manifest = root / "data/penn_train_dev.csv"
    _require(hashlib.sha256(manifest.read_bytes()).hexdigest() == preflight["manifest_sha256"],
             "Current train/development manifest differs from preflight.")
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    _require(records and all(r["provisional_split"] in ("train", "dev") for r in records),
             "Comparison requires a train/development-only manifest.")
    dev_rows = [r for r in records if r["provisional_split"] == "dev"]
    dev = {r["tick_id"]: r for r in dev_rows}
    _require(dev and len(dev) == len(dev_rows), "Development IDs must be nonempty and unique.")
    _require(len(dev) == preflight["counts"]["dev"], "Development count differs from preflight.")
    results = []
    for directory in ("runs", "experiments/runs", "experiments/reproductions", "experiments/dino"):
        for completion in sorted((root / directory).glob("*/completion.json")):
            try:
                result = _run_result(completion.parent, root, preflight, dev)
                if result is not None:
                    results.append(result)
            except (ValueError, KeyError, TypeError, OSError) as error:
                raise ValueError(f"{completion.parent.relative_to(root)}: {error}") from error
    baselines = {r["target"]: r for r in results if r["is_pinned_baseline"]}
    _require(set(baselines) == set(BASELINE_RUNS), "Both completed pinned baselines are required.")
    for result in results:
        baseline = baselines[result["target"]]
        floors = {label: max(0, values["correct"] - 1)
                  for label, values in baseline["per_class"].items()}
        result["promotion_class_correct_floors"] = floors
        result["promotion_eligible"] = (
            result["correct"] >= baseline["correct"]
            and all(result["per_class"][label]["correct"] >= floor for label, floor in floors.items())
        )
    return sorted(results, key=lambda r: (r["target"], -r["validation_accuracy"],
                                         -r["validation_balanced_accuracy"], r["validation_nll"], r["run"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=WORK)
    parser.add_argument("--output", type=Path, help="Optional JSON results file.")
    args = parser.parse_args()
    results = collect_results(args.root)
    print("DEVELOPMENT / VALIDATION ONLY — reused for model selection; no test evaluation.")
    print(f"{'Task':<17} {'Architecture':<19} {'Correct':>8} {'Accuracy':>9} {'Balanced':>9} "
          f"{'NLL':>7} {'ECE10':>7} {'Epoch':>5} {'Eligible':>8} {'>95%':>5} Run")
    for result in results:
        print(f"{result['target']:<17} {result['architecture']:<19} "
              f"{result['correct']:>3}/{result['validation_n']:<4} "
              f"{result['validation_accuracy']:>8.2%} {result['validation_balanced_accuracy']:>8.2%} "
              f"{result['validation_nll']:>7.3f} {result['validation_ece10']:>7.3f} "
              f"{result['checkpoint_epoch']:>5} {str(result['promotion_eligible']):>8} "
              f"{str(result['meets_development_target']):>5} {result['run']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
