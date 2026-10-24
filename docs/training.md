# TickPredict: v1 training and development guide

This is the original guide to the v1 life-stage and engorgement models, notebooks, and experiments. Paths and commands are relative to the repository root. For the current mobile apps, which also use v2 for species and life stage, see the [project overview](../README.md), [iOS guide](../ios/README.md), and [Android guide](../android/README.md).

Give the project a tick image and it returns two predictions, each with a confidence score:

- **Life stage:** `adult female` or `nymph`.
- **Engorgement:** `unengorged`, `semi-engorged`, or `fully engorged`.

Confidence is the predicted class's **uncalibrated softmax score**, averaged across views when enabled, from 0 to 1. It is not a measured probability that the prediction is correct. Both models always choose a supported class; they cannot reject non-tick images or unsupported species. These are Penn research models, with no independent phone-photo evaluation.

## Locked version v1.0

The accepted development release is tagged **`tickpredict-v1.0`**. It retains **97.27% life-stage accuracy** and **86.36% engorgement accuracy** on the 110 development images. Iterative tuning is finished for this release; the protected test remains untouched.

The selected weights and their original metadata are copied to `models/v1.0/`. `models/v1.0/manifest.json` records every file checksum and its original location. The testing notebook uses these versioned copies. Git preserves code, metadata, and checksums; the large `best.pt` files stay local, so copy the entire `models/v1.0/` folder when restoring on another computer. Training remains disabled by default and creates separate experiment runs when explicitly enabled.

## Use the notebooks

The local `.venv` is set up on this computer. Double-click **Start_TickPredict.command** and open the local Jupyter link printed in Terminal. The project has two notebook entry points:

- **`testing.ipynb`**: set `IMAGE_PATH` and run the cells to get both selected model predictions and their class scores. `MODEL_SET = "current"` uses the locked predictors recorded in `configs/current_models.json`; `"baseline"` selects the original models.
- **`train.ipynb`**: preview the selected training recipes, inspect results, or explicitly train models. Set `TASKS` to `["life_stage"]`, `["engorgement_level"]`, or both. The restored dataset is detected automatically.

`TRAINING_MODE` defaults to `"selected"` and supports three modes:

| Mode | What it trains |
|---|---|
| `"selected"` | The recipe snapshot in `configs/current_training.json`, including each model used by a selected ensemble. |
| `"experiments"` | The comparison plan chosen by `PLAN_PATH`; `RECIPES = None` runs all recipes, or a list selects a subset. |
| `"baseline"` | The original ResNet18 algorithm using the per-task configuration files. |

`PLAN_PATH` and `RECIPES` apply only to `"experiments"` mode. Selected mode creates fresh runs; it does not replace the current predictors.

In `train.ipynb`, `RUN_PREFLIGHT`, `RUN_SMOKE`, `RUN_TRAINING`, and `EXPORT_PLOTS` all default to `False`; enable each action explicitly. Leave the Terminal window open while using Jupyter; Control-C stops the server.

### Use the saved models from Terminal

The terminal wrappers below retain their original baseline defaults. Supply `--checkpoint` to use a single newer checkpoint. From Terminal, in this project directory:

```sh
.venv/bin/python scripts/predict_lifeStage.py /path/to/tick.jpg
.venv/bin/python scripts/predict_engorgementLevel.py /path/to/tick.jpg
```

Each JSON result contains `prediction`, `confidence`, `confidence_calibrated: false`, `model_scores_uncalibrated`, `task`, and `checkpoint`. An optional `--checkpoint /path/to/best.pt` selects another checkpoint of the same task. No training data or additional pretrained download is needed for inference.

### Setup on another computer

Use Python 3.12 and recreate the environment locally; virtual environments should not be copied between computers:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The dependency versions in `requirements.txt` were tested on macOS Apple Silicon. The launch script resolves its own project directory and starts authenticated Jupyter on `127.0.0.1:8891`. If that port is already occupied, use the existing instance or stop it before starting another.

## Training machine (DGX Spark)

Training runs on `spark-4ec2` (NVIDIA GB10, aarch64, CUDA 13) in `~/tickpredict`, a clone of this repository. The Mac keeps the testing notebook, the Core ML and ONNX exports, the iOS app (`ios/`) and the Android app (`android/`). Code moves through GitHub (`git push` on one machine, `git pull` on the other). Datasets, experiment runs and weights are untracked and live on the Spark; the Mac keeps a local copy of `dataset/` and `experiments/` as a backup.

Environment setup on the Spark (coremltools is macOS-only and is left out):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
grep -v -E "^(torch|torchvision|coremltools)==" requirements.txt | .venv/bin/python -m pip install -r /dev/stdin
```

On 2026-10-19, the v1.0 checkpoints reproduced their development results on the Spark GPU (107/110 life stage, 95/110 engorgement with D4, with matching NLL). Evaluate there with `--device cuda`. Training selects CUDA automatically.

To bring a finished release to the Mac for testing and Core ML export, run `scripts/pull_release.sh VERSION`. It copies `models/VERSION/`, including the weights, and verifies every checksum in its manifest.

## Development improvements

The first comparison trained four architectures for each task on the original split. The strongest single checkpoints reached **107/110 (97.27%) life stage** and **95/110 (86.36%) engorgement**. The life-stage model is ConvNeXt Tiny at 320 pixels; the engorgement model is EfficientNet V2 S at 384 pixels. These are development results, reused for model selection, with the protected test still unopened. Engorgement has not reached the greater-than-95% target.

Across three training seeds, life stage scored **107, 105, and 105 correct out of 110**; engorgement scored **95, 93, and 94**. The selected life-stage predictor uses one view. The selected engorgement predictor averages softmax scores from eight fixed rotations/reflections (D4): all 110 predicted labels and the 95/110 accuracy remain unchanged, while negative log-likelihood improves from 0.5769 to 0.5391. Neither three-seed ensemble was selected. These comparisons and their limits are recorded in `notes/refinement_results.json`.

`configs/current_models.json` records the explicit notebook selection and checkpoint fingerprints. Selected weights and adjacent provenance are preserved in `models/v1.0/`; only the large weight files are excluded from Git. Original experiment outputs remain under `experiments/`. Missing release weights cause a clear error; they are never silently replaced. Use `MODEL_SET = "baseline"` if only the committed original weights are available. The full validated leaderboard is in `notes/experiment_results.json`.

`configs/current_training.json` is the training recipe snapshot behind that selection; it contains settings, not weights, and records the selected inference views separately. Retraining writes new runs. After reviewing their results, selecting different predictors requires explicitly updating `configs/current_models.json` and the corresponding training snapshot. Training never promotes a model automatically.

Additional trials tested a training-reviewed foreground crop, auxiliary life-stage supervision, a frozen DINOv2 representation, and repeat training seeds. Every trial is retained, including ones that failed to improve accuracy. `notes/refinement_plan.json` records the bounded comparisons. All 44 CPU unit tests pass; the verification results are recorded in `notes/refinement_results.json`.

### Example review

The visual audit covered all 15 first-round engorgement errors and 15 correctly classified development controls, with observations recorded before labels and predictions were revealed. Underside views, damage, glare, and ambiguous adjacent grades occur, but image quality alone does not explain the errors. Rare stage/engorgement combinations also matter: adult female semi-engorged has only seven training examples and four errors among five development cases. The observations and disagreements are preserved in `notes/visual_review.json`; no labels were changed or development cases removed.

A subsequent trial replaced class-weighted loss with sampling that gives rare stage/engorgement combinations more exposure. It reached 93/110 (84.55%) and was not selected. It uses training labels only for sampling; prediction still needs only an image. This is an exploratory response to development errors, with the original split and evaluation labels preserved.

## Original retained checkpoints and results

The pinned baselines were selected using **validation balanced accuracy**. The retained life-stage run scored 88.76%, compared with 88.00% for the archived older run. This identifies the strongest saved development run from the original comparison, not a proven best architecture. New experiments use ordinary validation accuracy as their primary selection metric and continue to report balanced accuracy and per-class results.

| Task | Saved run | Best epoch | Validation accuracy | Validation balanced accuracy |
|---|---|---:|---:|---:|
| Life stage | `20261013-125510-220265_life_stage` | 7 | 100/110 = 90.91% | 88.76% |
| Engorgement | `20261013-125824-749104_engorgement_level` | 3 | 74/110 = 67.27% | 74.42% |

Both runs used 241 training images and the same 110 validation images. Validation was reused for checkpoint selection; these are not independent test results. **Zero protected test images were evaluated.** Balanced accuracy averages the recall for each class, giving each class equal weight. Person-level separation cannot be established from the available identifiers, and engorgement labels have no independently confirmed annotation rubric.

Each `runs/<run>/` directory retains the unchanged `best.pt`, `provenance.json`, `completion.json`, `history.json`, and `validation.json`, plus its original figures and a regenerated `RESULTS.md`. The reports include the full confusion matrices and per-class recall.

After restoring the dataset, reevaluating both pinned checkpoints reproduced all original development predictions. Fresh baseline training on this computer reached 99/110 for life stage and 73/110 for engorgement, each one correct prediction below the original result. This establishes a close reproduction, not a new improvement. Those runs are saved separately in `experiments/reproductions/`; the original checkpoint files remain unchanged. The testing notebook can select the improved candidates separately.

## Active files

- `testing.ipynb`: one image through both selected models; explicit current/baseline choice.
- `configs/current_models.json`: selected predictor paths and weight fingerprints.
- `configs/current_training.json`: training recipe snapshot for the selected predictors.
- `train.ipynb`: default to the selected recipes, inspect the development leaderboard, or choose experiment/baseline mode. Data checks, smoke tests, training, and plot export remain explicitly enabled actions; their switches default to `False`.
- `scripts/inference.py`: shared inference with checkpoint task/label validation.
- `scripts/predict_lifeStage.py` and `scripts/predict_engorgementLevel.py`: prediction functions and command-line entry points.
- `scripts/train.py`, `scripts/prepare.py`, `scripts/common.py`: training, data checks, and portable paths.
- `scripts/experiment.py` and `scripts/modeling.py`: configurable training, model definitions, initialization, and checkpoint preprocessing.
- `scripts/run_experiments.py`: preview or execute the predefined experiment plan; reuse completed matching runs.
- `scripts/compare_runs.py`: validate saved results and compare accuracy, class/subgroup performance, and uncalibrated confidence metrics on the fixed development split.
- `scripts/evaluate_predictor.py`: explicitly specified equal-probability ensembles and fixed D4 rotation/reflection averaging.
- `scripts/image_preprocessing.py`: shared letterboxing and optional frozen foreground crop.
- `scripts/multitask_experiment.py`: fixed auxiliary life-stage training experiment.
- `scripts/dino_probe.py`: frozen DINOv2 representation probe, with pinned official source and weights.
- `scripts/plot_run.py` and `scripts/summarize_run.py`: figures and result reports.
- `configs/`: the original per-task training configurations and `experiments.json` for the model comparison plan.
- `data/penn_train_dev.csv` and `notes/preflight.json`: original manifest and split fingerprints.
- `notes/cleanup_verification.json`: checks performed after cleanup.
- `dataset/`: restored source CSV, photos, and documentation; excluded from Git.
- `experiments/runs/` and `experiments/reproductions/`: new experiment outputs and baseline reproductions; excluded from Git.

The notebooks call the shared code in `scripts/`, so training and prediction logic stay in one place.

Both pinned models use ResNet18, fine-tuned in the last residual block and task head. Their inference pipeline remains EXIF orientation, RGB, bottom 4% cropped off, padding to 224 × 224, and ImageNet normalization. New checkpoints record their own architecture and preprocessing settings, which shared inference reads when an explicit checkpoint is selected. The crop was designed for Penn microscopy overlays and is not a validated universal phone-photo procedure. The models do not identify species, estimate attachment time, or estimate Lyme risk.

## Local dataset and training

The original dataset is restored locally. These locations are detected automatically:

```text
dataset/Penn_Dataset/prepared_data.csv
dataset/Penn Data/433 tick photos/
data/weights/resnet18-f37072fd.pth
```

The source CSV and all 351 train/development photo hashes match the recorded inputs. The original split remains 241 training / 110 development / 60 protected test images, plus 22 excluded records. Preflight reads split metadata but decodes only train/development images. Training checks source and manifest fingerprints and photo hashes, and never loads a test set.

The baseline ImageNet initialization is restored under `data/weights/`. The larger models use pinned torchvision ImageNet weights cached locally; on another computer, their first training use downloads missing initialization weights. Saved-checkpoint inference needs neither the dataset nor an initialization download.

If the dataset is stored elsewhere, set these overrides before launching Jupyter or a training command:

```sh
export TICKPREDICT_SOURCE_CSV="/path/to/original/prepared_data.csv"
export TICKPREDICT_IMAGE_DIR="/path/to/original/433 tick photos"
export TICKPREDICT_WEIGHTS="/path/to/resnet18-f37072fd.pth"
```

If the detected dataset folders are absent, the source CSV falls back to `data/prepared_data.csv` and photos fall back to their manifest paths. The baseline weights default to `data/weights/resnet18-f37072fd.pth`. `TICKPREDICT_IMAGE_DIR` remaps original filenames without editing the fingerprinted manifest. File hashes must still match. Restart the notebook kernel after changing environment variables. The dataset, initialization weights, and experiment outputs are not included in Git, so a fresh clone needs its own local copies or weight downloads.

Run preflight before training:

```sh
.venv/bin/python scripts/prepare.py
```

Preflight preserves the original train/dev/test assignments and checks separation by tick ID, image hash, and similarity group. In `train.ipynb`, the default `TRAINING_MODE = "selected"` retrains the chosen tasks' recipes from `configs/current_training.json`. Enable `RUN_SMOKE` for a small execution check, then `RUN_TRAINING` for full runs; selected-mode outputs are fresh runs under `experiments/runs/`. Training needs the local dataset and initialization weights even when saved predictor weights are already available.

To reproduce the original algorithm, use `TRAINING_MODE = "baseline"`; its outputs go into `experiments/reproductions/`.

### Compare model recipes

To compare recipes in the notebook, set `TRAINING_MODE = "experiments"` and choose `PLAN_PATH`. The first predefined comparison in `configs/experiments.json` includes four recipes for each task:

| Recipe | Architecture | Input size |
|---|---|---:|
| `resnet18_320` | ResNet18 | 320 × 320 |
| `resnet50_320` | ResNet50 | 320 × 320 |
| `efficientnet_v2_s_384` | EfficientNet V2 S | 384 × 384 |
| `convnext_tiny_320` | ConvNeXt Tiny | 320 × 320 |

These recipes compare combinations of architecture, resolution, and regularization. They retain full-frame letterboxing and the Penn overlay crop, warm up the new classification head, then fine-tune with frozen batch-normalization running statistics. Moderate augmentation adds color variation and rotation without cropping away the specimen. Training uses the available GPU, including MPS on Apple Silicon, and runs configurations sequentially.

Preview all planned configurations without training:

```sh
.venv/bin/python scripts/run_experiments.py
```

Run a small execution check for one recipe:

```sh
.venv/bin/python scripts/run_experiments.py --tasks engorgement_level --recipes resnet18_320 --execute --smoke
```

Start the full predefined comparison, or add `--tasks` and `--recipes` to select a subset:

```sh
.venv/bin/python scripts/run_experiments.py --execute
```

`--execute` is required to start training. Runs are saved under `experiments/runs/`; rerunning an experiment plan reuses completed configurations with matching settings and code/data fingerprints. This reuse applies to experiment mode and the plan runner; selected mode creates fresh runs. Smoke runs test loading, gradients, and checkpoint writing; their scores are excluded from comparisons.

Compare completed runs:

```sh
.venv/bin/python scripts/compare_runs.py
```

The leaderboard uses the same 110 development images and ranks ordinary accuracy first, balanced accuracy second, and lower negative log-likelihood third. A promotion candidate must match or exceed the pinned model's accuracy, with no class losing more than one correct development prediction. Reports also include stage × engorgement subgroups, which matter because several combinations have very few examples. No run automatically changes the notebook selection in `configs/current_models.json`.

The target is **greater than 95% accuracy for both tasks**, on all evaluated images. On this development split that means at least 105/110 correct per task. Repeated development comparisons are model selection, not independent evidence of that target; the protected test set stays untouched until final model selection. Confidence remains uncalibrated, and higher capacity alone cannot resolve uncertain labels or establish performance on phone photos.

## Cleanup and history

Cleanup on October 19, 2026 retained both image tasks. The duplicate workspace, four smoke runs, older life-stage run, species/questionnaire/attachment-time/risk/mockup notebooks, copied Jupyter runtime, and old logs/settings were moved to **`_archive/2026-10-19/`**. Nothing was permanently deleted; this reduces active clutter, not the archive's disk usage.

`_archive/2026-10-19/manifest.json` records the moves and retained checkpoint hashes. `before_cleanup/` contains the original active code and notebooks. In particular, its `scripts/train.py` still matches the training-script fingerprint recorded in the retained models. The original source-data manifest and all retained model JSON files are unchanged.

The two separate training notebooks were subsequently consolidated into root-level `train.ipynb`. Their prior versions are preserved in `_archive/2026-10-19/notebook_consolidation/Training/`.

The cleanup corrected the older default life-stage checkpoint, swapped notebook task settings, automatic full training, selection of the wrong task's results, computer-specific paths, and the hardcoded result summarizer. Saved notebook outputs were cleared. Unrelated experimental notebook interfaces depended on model folders absent from this copy; they remain available in the archive for later review.
