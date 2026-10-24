"""Export the app's two models to ONNX for the Android app.

Same graphs as the Core ML export (scripts/export_coreml.py): the input is the
already letterboxed 0-255 RGB image as float32 NCHW, and normalization, D4
averaging and softmax are built in.

- TickPredictV2.onnx: the v2.0 multi-head model, one output per head.
- EngorgementLevel.onnx: the v1.0 engorgement model, output "probabilities".

Weights are stored as float16 and cast back to float32 when the model loads, so
each file stays under GitHub's 100 MB limit while inference runs in float32.
Class lists, preprocessing and release go in the ONNX metadata.

    .venv/bin/python scripts/export_onnx.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

from common import WORK, digest
from export_coreml import RELEASE as V1, MultiHeadPredictor, _require, load_predictor
from multihead_inference import load_multihead

V2_CHECKPOINT = WORK / "models/v2.0/multihead/best.pt"
OUTPUT = WORK / "android/app/src/main/assets/models"
OPSET = 18


def _float16_weights(model, minimum=1024):
    """Store large float32 initializers as float16 plus a Cast node back to float32."""
    graph = model.graph
    casts, kept = [], []
    for tensor in graph.initializer:
        array = numpy_helper.to_array(tensor)
        if tensor.data_type != onnx.TensorProto.FLOAT or array.size < minimum:
            kept.append(tensor)
            continue
        half = numpy_helper.from_array(array.astype(np.float16), tensor.name + "_fp16")
        kept.append(half)
        casts.append(helper.make_node("Cast", [half.name], [tensor.name], to=onnx.TensorProto.FLOAT,
                                      name=tensor.name + "_cast"))
    del graph.initializer[:]
    graph.initializer.extend(kept)
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    return model


def _save(module, size, outputs, path, metadata):
    example = torch.rand(1, 3, size, size) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        program = torch.onnx.export(module, (example,), dynamo=True, opset_version=OPSET,
                                    input_names=["image"], output_names=outputs, external_data=False)
    model = _float16_weights(program.model_proto)
    for key, value in metadata.items():
        model.metadata_props.append(onnx.StringStringEntryProto(key=key, value=value))
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def export_multihead(release):
    model, saved = load_multihead(V2_CHECKPOINT)
    spec = saved["preprocessing"]
    _require(not spec.get("foreground_crop"), "Foreground crop is not supported on device.")
    heads = list(saved["heads"])
    rows = [tuple(saved["logit_rows"][head]) for head in heads]
    predictor = MultiHeadPredictor(model, spec["mean"], spec["std"], rows, "d4").eval()
    return _save(predictor, spec["image_size"], heads, OUTPUT / "TickPredictV2.onnx", {
        "task": "multihead",
        "release": release,
        "heads": json.dumps(saved["heads"]),
        "preprocessing": json.dumps(spec, sort_keys=True),
        "tta": "d4",
        "checkpoint_sha256": digest(V2_CHECKPOINT),
        "confidence_calibrated": "false",
    })


def export_engorgement():
    manifest = json.loads((V1 / "manifest.json").read_text())
    predictor, saved, entry, path = load_predictor("engorgement_level", manifest)
    spec = saved["preprocessing"]
    return _save(predictor, spec["image_size"], ["probabilities"], OUTPUT / "EngorgementLevel.onnx", {
        "task": "engorgement_level",
        "release": manifest["release"],
        "classes": json.dumps(saved["classes"]),
        "preprocessing": json.dumps(spec, sort_keys=True),
        "tta": entry["predictor"]["tta"],
        "checkpoint_sha256": digest(path),
        "confidence_calibrated": "false",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", default="v2.0")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    for path in (export_multihead(args.release), export_engorgement()):
        print(f"Exported {path.relative_to(WORK)} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
