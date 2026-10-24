"""Export the locked v1.0 predictors to Core ML for on-device iOS inference.

Each exported model takes the already letterboxed RGB image (EXIF orientation,
bottom 4% removed, Lanczos pad to a square of light grey) and returns class
probabilities. ImageNet normalization, softmax, and the engorgement predictor's
fixed D4 averaging are built into the model, so the app runs one prediction per
task. Confidence remains the uncalibrated softmax score.
"""

import argparse
import json
from pathlib import Path

import coremltools as ct
import torch

from common import WORK, digest
from inference import EXPECTED_CLASSES
from modeling import build_model

RELEASE = WORK / "models/v1.0"
OUTPUT = RELEASE / "coreml"
TASKS = {
    "life_stage": "LifeStage",
    "engorgement_level": "EngorgementLevel",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def d4_views(batch):
    """Same views and order as evaluate_predictor.d4_views, using flip/transpose only."""
    views = []
    for reflected in (False, True):
        view = batch
        for k in range(4):
            out = view.flip(-1) if reflected else view
            views.append(out)
            # rot90 by one quarter turn counter-clockwise over (H, W).
            view = view.transpose(-2, -1).flip(-2)
    return views


class Predictor(torch.nn.Module):
    """0-255 RGB input -> normalized -> (optional D4) -> averaged softmax."""

    def __init__(self, model, mean, std, tta):
        super().__init__()
        self.model = model
        self.tta = tta
        self.register_buffer("mean", torch.tensor(mean).reshape(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(std).reshape(1, 3, 1, 1) * 255)

    def forward(self, image):
        x = (image - self.mean) / self.std
        if self.tta == "d4":
            views = torch.cat(d4_views(x))
            return self.model(views).softmax(1).mean(0, keepdim=True)
        return self.model(x).softmax(1)


def load_predictor(task, manifest):
    entry = manifest["models"][task]
    (checkpoint,) = entry["predictor"]["checkpoints"]
    path = WORK / checkpoint
    _require(digest(path) == entry["checkpoint_sha256"][checkpoint],
             f"Checkpoint fingerprint changed: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    _require(saved.get("target") == task and set(saved["classes"]) == EXPECTED_CLASSES[task],
             f"Unexpected checkpoint task or classes: {path}")
    _require(saved["classes"] == sorted(saved["classes"]), "Class order must be sorted.")
    model = build_model(saved["architecture"], len(saved["classes"]), pretrained=False)
    model.load_state_dict(saved["state_dict"])
    spec = saved["preprocessing"]
    _require(not spec.get("foreground_crop"), "Foreground crop is not supported on device.")
    predictor = Predictor(model.eval(), spec["mean"], spec["std"], entry["predictor"]["tta"]).eval()
    return predictor, saved, entry, path


def export(task, manifest, precision):
    predictor, saved, entry, path = load_predictor(task, manifest)
    spec = saved["preprocessing"]
    size = spec["image_size"]
    example = torch.rand(1, 3, size, size) * 255
    with torch.inference_mode():
        traced = torch.jit.trace(predictor, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=example.shape, color_layout=ct.colorlayout.RGB)],
        classifier_config=ct.ClassifierConfig(saved["classes"]),
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.author = "TickPredict"
    mlmodel.version = manifest["release"]
    mlmodel.short_description = (
        f"TickPredict {manifest['release']} {task.replace('_', ' ')}: {entry['description']}. "
        "Input: EXIF-oriented RGB, bottom 4% removed, Lanczos letterbox to "
        f"{size}x{size} on RGB {tuple(spec['pad_color'])}. Output confidence is an "
        "uncalibrated softmax score.")
    mlmodel.input_description["image"] = f"Letterboxed {size}x{size} RGB tick photo"
    mlmodel.user_defined_metadata.update({
        "task": task,
        "release": manifest["release"],
        "classes": json.dumps(saved["classes"]),
        "preprocessing": json.dumps(spec, sort_keys=True),
        "tta": entry["predictor"]["tta"],
        "checkpoint_sha256": digest(path),
        "confidence_calibrated": "false",
        "precision": precision,
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / f"{TASKS[task]}.mlpackage"
    mlmodel.save(str(destination))
    return destination


class MultiHeadPredictor(torch.nn.Module):
    """0-255 RGB -> normalized -> (optional D4) -> one averaged softmax per head."""

    def __init__(self, model, mean, std, rows, tta):
        super().__init__()
        self.model = model
        self.rows = rows
        self.tta = tta
        self.register_buffer("mean", torch.tensor(mean).reshape(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(std).reshape(1, 3, 1, 1) * 255)

    def forward(self, image):
        x = (image - self.mean) / self.std
        if self.tta == "d4":
            x = torch.cat(d4_views(x))
        logits = self.model(x)
        return tuple(logits[:, start:end].softmax(1).mean(0, keepdim=True) for start, end in self.rows)


def export_multihead(checkpoint, destination, release, tta="d4", precision="float16"):
    """Export a v2 multi-head checkpoint as one model with an output per head."""
    from multihead_inference import load_multihead

    model, saved = load_multihead(checkpoint)
    _require(not saved["smoke_only"] or release.endswith("smoke"), "Refusing to release a smoke checkpoint.")
    spec = saved["preprocessing"]
    _require(not spec.get("foreground_crop"), "Foreground crop is not supported on device.")
    heads = list(saved["heads"])
    rows = [tuple(saved["logit_rows"][head]) for head in heads]
    predictor = MultiHeadPredictor(model, spec["mean"], spec["std"], rows, tta).eval()
    size = spec["image_size"]
    example = torch.rand(1, 3, size, size) * 255
    with torch.inference_mode():
        traced = torch.jit.trace(predictor, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=example.shape, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name=head) for head in heads],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.author = "TickPredict"
    mlmodel.version = release
    mlmodel.short_description = (
        f"TickPredict {release}: species, life stage and engorgement from one photo "
        f"({saved['architecture']}, {size} px, {'8-view average' if tta == 'd4' else 'single view'}). "
        "Each output is an uncalibrated softmax over that head's classes.")
    mlmodel.input_description["image"] = f"Letterboxed {size}x{size} RGB tick photo"
    for head in heads:
        mlmodel.output_description[head] = f"Probabilities over {json.dumps(saved['heads'][head])}"
    mlmodel.user_defined_metadata.update({
        "task": "multihead",
        "release": release,
        "heads": json.dumps(saved["heads"]),
        "preprocessing": json.dumps(spec, sort_keys=True),
        "tta": tta,
        "checkpoint_sha256": digest(Path(checkpoint)),
        "confidence_calibrated": "false",
        "precision": precision,
    })
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(destination))
    return Path(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("float16", "float32"), default="float16")
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--multihead", type=Path, help="Export this v2 multi-head checkpoint instead.")
    parser.add_argument("--release", help="Release name for --multihead, e.g. v2.0.")
    parser.add_argument("--output", type=Path, help="Destination .mlpackage for --multihead.")
    parser.add_argument("--tta", choices=("none", "d4"), default="d4")
    args = parser.parse_args()
    if args.multihead:
        _require(args.release and args.output, "--multihead needs --release and --output.")
        print(f"Exported: {export_multihead(args.multihead, args.output, args.release, args.tta, args.precision)}")
        return
    manifest = json.loads((RELEASE / "manifest.json").read_text())
    for task in args.tasks:
        print(f"Exported {task}: {export(task, manifest, args.precision)}")


if __name__ == "__main__":
    main()
