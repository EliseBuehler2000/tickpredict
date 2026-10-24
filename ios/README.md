# TickPredict for iOS

A SwiftUI app (iOS 26+, iPhone) that runs TickPredict's models on the phone with Core ML: the v2.0 multi-head model for species and life stage, and the v1.0 model for engorgement (it scored 95/110 on development photos against 92/110 for v2's engorgement head).

- **Scan:** camera or photo library, then species (8 U.S. ticks) with what it can carry, life stage (nymph, adult female, adult male), engorgement, and a Lyme risk card, each with its model confidence.
- **History:** scans saved on the device with the date, the bite spot, where you were, and notes.
- **Guide:** removal, life stage, engorgement, when to call a doctor, and the models' limits.
- **Find care:** nearby urgent care, doctors, or hospitals from Apple Maps. The location is used for that one search and never stored.

Nothing leaves the phone: no server, account, or analytics.

## Build

The Core ML models are generated from the release weights. The two the app uses are committed, so a fresh clone builds without this step:

```sh
.venv/bin/python -m pip install -r requirements.txt   # includes coremltools
cd scripts && ../.venv/bin/python export_coreml.py     # writes models/v1.0/coreml/*.mlpackage
../.venv/bin/python export_coreml.py --multihead ../models/v2.0/multihead/best.pt \
    --release v2.0 --output ../models/v2.0/coreml/TickPredictV2.mlpackage
```

The app bundles `TickPredictV2.mlpackage` and v1.0's `EngorgementLevel.mlpackage`.

Then open `ios/TickPredict.xcodeproj` in Xcode 27, choose your team under Signing & Capabilities, and run. `ios/project.yml` is the source for the project file; after adding or removing files, regenerate it with `cd ios && xcodegen generate`.

## How a photo becomes a prediction

`TickPredict/Inference/` is shared with the parity tool and has no UI code.

- `ImagePreparation.swift` reproduces the Python pipeline. It applies EXIF orientation, reads RGB without colour management (as Pillow does), removes the bottom 4%, and letterboxes to the model size on RGB (240, 240, 240) with a line-by-line port of Pillow's Lanczos resampler.
- `TickClassifier.swift` reads each model's preprocessing recipe and classes from its metadata and runs both models. Both use the same 384 px recipe, so the photo is prepared once. ImageNet normalisation, softmax, and 8-view averaging are built into the exported models.
- `Model/Species.swift` holds what each species can carry (from CDC's "Where ticks live") and the Lyme risk rules. An engorged tick is never told it can't spread Lyme: on Penn's development photos, all blacklegged, the species head named only 6 of 16 fully engorged ones correctly.

Species is saved inside each scan as a JSON blob rather than a nested struct. SwiftData reads a nested optional struct added after v1.0 as present but empty for older scans, and the app crashed on launch; scans saved with v1.0 now open without a species.

## Check the app against Python

```sh
S=$(mktemp -d)
cd scripts && ../.venv/bin/python ios_parity_fixtures.py $S/fixtures && cd ..
for m in v2.0/coreml/TickPredictV2 v1.0/coreml/EngorgementLevel; do xcrun coremlcompiler compile models/$m.mlpackage $S/compiled; done
swiftc -O -o $S/parity-check ios/TickPredict/Inference/*.swift ios/Tools/ParityCheck/main.swift
$S/parity-check $S/fixtures $S/compiled
```

Result on 2026-10-19 (110 development images plus 21 copies with EXIF orientations 2–8):

| Check | Result |
|---|---|
| Species labels matching PyTorch | 110/110 (89 correct; these are microscope photos, which the species head never trained on) |
| Life stage labels matching PyTorch | 110/110 (106 correct) |
| Engorgement labels matching PyTorch | 110/110 (95 correct) |
| Largest probability difference | 0.020 species, 0.019 life stage, 0.033 engorgement |
| Slowest full scan on an M-series Mac | 189 ms |
| Rotation, crop, letterbox and resampling | Byte-identical to Python when given the same decoded pixels |

The remaining input differences, at most 5 of 255 in about 7% of values, come from Apple's JPEG decoder differing slightly from Pillow's. No test images are used.

## UI tests

`TickPredictUITests` drives the app in the Simulator through the main flows on a real development photo: scan, save (toolbar and bottom button), edit, delete, delete from History while a scan is open in another tab, discard confirmation, Details and Find care, and backing out mid-analysis. They also check that an engorged blacklegged tick the species head misnames still gets the Lyme warning, and that species survives a relaunch. They need the local `dataset/` and are skipped without it.

```sh
cd ios && xcodebuild test -project TickPredict.xcodeproj -scheme TickPredict \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro'
```

Debug builds compile Swift with `-O`. Image preparation is pixel-by-pixel Swift, and unoptimized it took about 30 seconds per scan on an iPhone. Optimized, and spread across all cores, a 12 MP photo takes about 65 ms on an M-series Mac.

`FindCareDeviceTests` also run on a real iPhone. `testFindCareFromResult` needs a photo copied into the app's Documents first:

```sh
xcrun devicectl device copy to --device <UDID> --domain-type appDataContainer \
  --domain-identifier com.ericbuehler.TickPredict --source photo.jpg --destination Documents/sample.jpg
```

## Debug launch arguments

For reviewing screens in the Simulator, in Debug builds only:
- `-demoImage /path.jpg`: open a new scan of that photo. A relative path is inside the app's Documents folder.
- `-seedScans /a.jpg,/b.jpg`: fill History.
- `-tab history|guide`: start on that tab.
- `-findCare YES`: open Find care.
- `-camera YES`: open the camera.
- `-resetData YES`: delete all saved scans first (used by the UI tests).
