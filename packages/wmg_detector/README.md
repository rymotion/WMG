# wmg_detector

An on-device AI-music detector for the Flutter app on **iOS and macOS**. The app passes in PCM audio and gets back P(AI) plus per-band evidence. Two models can be bundled: a **10 s** model (`WmgDetector`, required) and a **5 s** model (`WmgDetector5s`, optional). Everything runs locally:

```
PCM (int16) ─► wmg-features (Rust, via C ABI) ─► level-normalised log-mel 10 s windows
            ─► Core ML BandNet (WmgDetector.mlmodelc) ─► ai_logit + band scores per window
            ─► P(AI) = sigmoid(mean window logit)
```

`wmg-features` is the same Rust crate that produces training features, so the app's input matches what the model was trained on. At startup the plugin checks that the model's `feature_spec_hash` matches the bundled library and refuses to score if they differ.

## Dart API

```dart
const detector = WmgDetector();
final info = await detector.info();          // featureSpecHash, modelSpecHash, modelRun, compatible
final r = await detector.scorePcm16(pcm, 32000);            // 10 s model; mono s16le, any rate
final r5 = await detector.scorePcm16(pcm, 32000, model: '5s'); // 5 s model (halves of the 10 s windows)
r.pAi; r.windows.first.bands;                // [bass, mid, treble] in 0..1
```

Inputs shorter than 3 s, or quieter than the −45 dBFS window gate after normalisation, return an empty result.

## Building the native pieces

The XCFramework and the compiled model are generated artefacts, so they're gitignored. Install them from a training run:

```bash
packages/wmg_detector/tool/install.sh training/runs/<10 s run> [training/runs/<5 s run>]
```

`install.sh` does three things:
1. Builds `WmgFeatures.xcframework` with `features/apple/build_xcframework.sh`. The slices are macOS arm64+x86_64, iOS arm64, and simulator arm64+x86_64.
2. Exports the model to Core ML if needed, using `python -m wmg_train.export_coreml`. The export includes a parity check against PyTorch.
3. Copies both into `darwin/`.

After switching models, regenerate the parity fixture and run the on-device test:

```bash
cd training && .venv-arm64/bin/python -m wmg_train.parity_fixture runs/<10 s run>/model.pt --model5 runs/<5 s run>/model.pt
cd ../frontend && flutter test integration_test/on_device_parity_test.dart -d macos
```

Run the same test with an iOS simulator id as well.

## Tests in `frontend/integration_test/`

- **`on_device_parity_test.dart`:** scores a deterministic synthetic signal through the real native path and checks it against PyTorch (P(AI) within 0.01).
- **`live_mic_test.dart`:** launches the app, presses the mic, records through the device microphone and prints the on-device estimate. Play music near the mic while it runs, and pass `--dart-define=LIVE_IDENTIFY=false` to skip ACRCloud.

## How the app uses both models

While recording, the app scores every **5 s**. The last 5 s goes to the 5 s model, and the last 10 s (the latest 5 s plus the 5 s before it) goes to the 10 s model.

- **Headline:** the average of the two models' P(AI). For the first 10 s it shows the 5 s model alone, marked as an early estimate.
- **Delta:** the 5 s minus 10 s difference is shown next to it. A gap of more than 0.1 is flagged "Models disagree, uncertain".
- **Why the delta matters:** on held-out bench data, results where the two agreed were 98.6% accurate on clean files and 85.9% through a speaker. Where they disagreed, accuracy fell to 60–80% (see `reports/bench/`).
