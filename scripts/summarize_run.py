"""Create RESULTS.md from one completed run without loading images or models."""

import argparse
import json
import math
from pathlib import Path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cell(value):
    """Keep source labels and IDs inside their Markdown table cells."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def summarize_run(run_path):
    """Validate saved metadata and write a report; return its path."""
    run_path = Path(run_path).expanduser().resolve()
    completion = json.loads((run_path / "completion.json").read_text())
    provenance = json.loads((run_path / "provenance.json").read_text())
    validation = json.loads((run_path / "validation.json").read_text())
    history = json.loads((run_path / "history.json").read_text())

    _require(completion.get("status") == "completed", "Run is not completed.")
    _require(completion.get("smoke_only") is False, "A full, non-smoke run is required.")
    _require(provenance.get("smoke_only") is False, "Provenance must identify a non-smoke run.")
    target = provenance.get("config", {}).get("target")
    _require(isinstance(target, str) and target, "Provenance is missing its target.")
    classes = validation.get("classes", [])
    _require(isinstance(classes, list) and classes and all(isinstance(c, str) for c in classes),
             "Validation classes must be a nonempty list of labels.")
    _require(len(set(classes)) == len(classes), "Validation classes must be unique.")
    _require(classes == provenance.get("classes"), "Validation and provenance classes differ.")

    cm = validation.get("confusion_matrix", [])
    _require(isinstance(cm, list) and len(cm) == len(classes)
             and all(isinstance(row, list) and len(row) == len(classes) for row in cm),
             "Confusion-matrix dimensions do not match the classes.")
    _require(all(type(value) is int and value >= 0 for row in cm for value in row),
             "Confusion-matrix counts must be nonnegative integers.")
    class_counts = [sum(row) for row in cm]
    _require(all(class_counts), "Every class must have validation observations.")
    n = sum(class_counts)
    predictions = validation.get("predictions", [])
    _require(isinstance(predictions, list) and len(predictions) == n,
             "Prediction count does not match the confusion matrix.")
    _require(provenance.get("validation_n") == n, "Provenance validation count differs.")
    reconstructed = [[0 for _ in classes] for _ in classes]
    for prediction in predictions:
        _require(prediction.get("true") in classes and prediction.get("predicted") in classes,
                 "Prediction contains an unsupported class label.")
        reconstructed[classes.index(prediction["true"])][classes.index(prediction["predicted"])] += 1
    _require(reconstructed == cm, "Saved predictions and confusion matrix disagree.")

    correct = sum(cm[i][i] for i in range(len(classes)))
    recalls = [cm[i][i] / count for i, count in enumerate(class_counts)]
    balanced_accuracy = sum(recalls) / len(recalls)
    for label, recorded in [
        ("validation", validation.get("balanced_accuracy")),
        ("completion", completion.get("best_validation_balanced_accuracy")),
    ]:
        _require(isinstance(recorded, (int, float))
                 and math.isclose(recorded, balanced_accuracy, rel_tol=1e-9, abs_tol=1e-12),
                 f"Computed balanced accuracy differs from {label} metadata.")

    _require(isinstance(history, list) and history, "Epoch history is missing.")
    _require(completion.get("epochs") == len(history), "Completion epoch count differs from history.")
    for index, epoch in enumerate(history, 1):
        score = epoch.get("validation_balanced_accuracy")
        _require(epoch.get("epoch") == index, "History must contain consecutive epochs starting at 1.")
        _require(isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1,
                 "History contains an invalid balanced accuracy.")
    # Training replaces the checkpoint only on strict improvement. With tied
    # scores, max returns the first occurrence, matching the saved checkpoint.
    best = max(history, key=lambda epoch: epoch["validation_balanced_accuracy"])
    _require(math.isclose(best["validation_balanced_accuracy"], balanced_accuracy,
                         rel_tol=1e-9, abs_tol=1e-12),
             "History's best score differs from the saved validation result.")
    test_used = provenance.get("test_images_used")
    _require(type(test_used) is int and test_used >= 0,
             "Provenance must record a nonnegative test_images_used count.")

    title = target.replace("_", " ")
    lines = [
        f"# TickPredict — {title} results",
        "",
        "Completed research baseline on Penn validation images.",
        "",
        f"Run: {run_path.name}. Supported labels: {', '.join(classes)}.",
        "",
        "| Result | Value |",
        "|---|---:|",
        f"| Correct predictions | {correct} of {n} |",
        f"| Overall accuracy | {correct / n:.2%} |",
        f"| Balanced accuracy | {balanced_accuracy:.2%} |",
        f"| Best saved epoch | {best['epoch']} |",
        f"| Total epochs run | {len(history)} |",
        f"| Protected test images used (recorded provenance) | {test_used} |",
        "",
        "## Confusion matrix",
        "",
        "Rows are recorded labels; columns are model predictions. Balanced accuracy is the mean of the per-class recalls below.",
        "",
        "| Recorded label | " + " | ".join(f"Predicted {_cell(c)}" for c in classes) + " | Recall |",
        "|---|" + "---:|" * (len(classes) + 1),
    ]
    for i, label in enumerate(classes):
        lines.append(f"| {_cell(label)} | " + " | ".join(str(value) for value in cm[i])
                     + f" | {cm[i][i]}/{class_counts[i]} = {recalls[i]:.2%} |")
    lines += [
        "",
        f"The model made {n - correct} errors among {n} validation observations.",
        "",
        "## Learning history",
        "",
        f"The saved model was selected at epoch {best['epoch']} using validation balanced accuracy. "
        f"Training ran for {len(history)} epochs. The same validation observations were reused across epochs.",
    ]
    if (run_path / "validation_curve.png").is_file():
        lines += ["", "![Validation balanced accuracy by epoch](validation_curve.png)"]
    lines += ["", "## Individual errors", ""]
    errors = [p for p in predictions if p["true"] != p["predicted"]]
    if errors:
        lines += ["| Tick ID | Recorded label | Model prediction |", "|---|---|---|"]
        lines += [f"| {_cell(p.get('tick_id', 'Not recorded'))} | {_cell(p['true'])} | {_cell(p['predicted'])} |"
                  for p in errors]
    else:
        lines.append("No incorrect predictions were recorded in this validation set.")
    lines += [
        "",
        "## Interpretation",
        "",
        "These validation observations were used for checkpoint selection, so these figures are not independent test results. "
        f"They describe {title} classification for the listed labels in this Penn cohort. "
        "These saved results do not evaluate smartphone photographs, unsupported classes, or human Lyme risk. "
        "Softmax model scores are uncalibrated and must not be read as validated probabilities of correctness. "
        "Person-level separation cannot be checked because person identifiers are unavailable.",
        "",
        f"The run provenance records {test_used} protected test images used. "
        "Generating this report reads saved JSON only and performs no model evaluation.",
        "",
        "## Saved files",
        "",
        "- best.pt: selected model checkpoint.",
        "- validation.json: confusion matrix, recorded labels, predictions and uncalibrated model scores.",
        "- history.json: results for each training epoch.",
        "- provenance.json: cohort fingerprints, classes, software versions and configuration.",
        "- completion.json: completion status and best recorded validation score.",
        "",
        "This report was generated from the saved run; its checkpoints and JSON records were left unchanged.",
        "",
    ]
    destination = run_path / "RESULTS.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Directory containing a completed non-smoke run")
    arguments = parser.parse_args()
    print(summarize_run(arguments.run))
