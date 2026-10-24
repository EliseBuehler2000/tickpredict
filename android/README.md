# TickPredict for Android

A Jetpack Compose port of the iOS app (`ios/`), screen for screen. It runs the same two models on the phone with ONNX Runtime: the v2.0 multi-head model for species and life stage, and the v1.0 model for engorgement.

- **Scan:** camera (CameraX) or the photo picker, then the same analysis steps, species card, life stage, engorgement, Lyme risk, and details as on iOS.
- **History:** scans saved on the device with the date, the bite spot, where you were, and notes. Swipe left to delete.
- **Guide:** the same text as iOS.
- **Find care:** nearby urgent care, doctors, or hospitals. Android has no Apple Maps, so places come from OpenStreetMap: [Photon](https://photon.komoot.io) finds them, [Nominatim](https://nominatim.org) adds phone numbers and handles typed towns or ZIP codes, and osmdroid draws the map. None of these need an API key. Directions open Google Maps. The location is used for that one search and never stored.

Nothing else leaves the phone: no server, account, or analytics.

## Differences from iOS

- Liquid Glass has no Android equivalent; buttons and sheets use Material 3 in the same palette, and headings use Nunito (SIL Open Font License, `licenses/`) in place of SF Rounded.
- "iPhone" reads "phone" in the app's text.
- Urgent care has no category of its own in most OpenStreetMap data, so it's found by name ("urgent", "walk-in", "express care"…). Coverage depends on the map data and can be thinner than Apple Maps.

## Build

Requires the Android SDK (platform 35) and JDK 17 or later; Android Studio's bundled JDK works.

```sh
cd android
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Or open `android/` in Android Studio and run. Minimum Android 10 (API 29).

The models in `app/src/main/assets/models/` are exported from the release weights:

```sh
.venv/bin/python -m pip install onnx onnxruntime onnxscript
cd scripts && ../.venv/bin/python export_onnx.py
```

Normalization, the 8-view averaging and softmax are inside each model, as in the Core ML export. Weights are stored as float16 (57 MB and 42 MB, under GitHub's 100 MB limit) and run in float32.

## How a photo becomes a prediction

`inference/` is a line-for-line port of the Swift pipeline:

1. `ImageDecoding` decodes the photo without colour conversion (as Pillow does) and applies the EXIF orientation.
2. `ImagePreparation` removes the bottom 4% and letterboxes to 384 × 384 with a port of Pillow's Lanczos resampler, fixed-point rounding included.
3. `TickClassifier` runs both ONNX models on the prepared pixels.

## Parity check

`app/src/androidTest/.../ParityTest.kt` runs the whole pipeline on the 110 development photos (plus copies with each EXIF orientation) and compares with Python. Development images only.

```sh
.venv/bin/python scripts/ios_parity_fixtures.py /tmp/fixtures
# Collect fixtures.json, the *_multihead.rgb files, and every photo the fixtures name into one folder, then:
adb push FOLDER /data/local/tmp/parity
adb shell run-as com.ericbuehler.tickpredict cp -r /data/local/tmp/parity files/
cd android && ./gradlew :app:assembleDebugAndroidTest
adb install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell am instrument -w -e class com.ericbuehler.tickpredict.ParityTest \
    com.ericbuehler.tickpredict.test/androidx.test.runner.AndroidJUnitRunner
adb shell run-as com.ericbuehler.tickpredict cat files/parity-result.txt
```

(`connectedAndroidTest` reinstalls the app and would delete the fixtures, so the test is run directly.)

Result on a Pixel 8 emulator (Android 15), 2026-10-22:

| Check | Result |
|---|---|
| Prepared inputs identical to Python | 131/131 (worst pixel difference 0) |
| Species agrees with PyTorch | 110/110, max probability difference 0.0008 |
| Life stage agrees with PyTorch | 110/110, max difference 0.0002 |
| Engorgement agrees with PyTorch | 110/110, max difference 0.0020 |
| Slowest scan (emulator CPU) | 2.0 s |

`scripts/verify_onnx.py` runs the same comparison on the Mac with ONNX Runtime for Python.

## Debug launch extras

For reviewing screens on an emulator (debug builds only). Photos given by relative path are read from the app's external files folder (`/sdcard/Android/data/com.ericbuehler.tickpredict/files/`).

```sh
adb shell am start -n com.ericbuehler.tickpredict/.MainActivity \
    --es demoImage sample.jpg          # open a new scan of that photo
    --es tab history                   # or guide
    --ez findCare true                 # push Find care
    --ez camera true                   # open the camera
    --ez resetData true                # delete saved scans first
    --es seedScans a.jpg,b.jpg         # save scans of these photos if history is empty
```
