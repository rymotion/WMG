# wmg_detector

An on-device AI-music detector for the Flutter app on **iOS and macOS**. The app passes in PCM audio and gets back P(AI) plus per-band evidence. Everything runs locally:

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
final r = await detector.scorePcm16(pcm, 32000);  // mono s16le bytes at any sample rate
r.pAi; r.windows.first.bands;                // [bass, mid, treble] in 0..1
```

Inputs shorter than 3 s, or quieter than the −45 dBFS window gate after normalisation, return an empty result.

## Building the native pieces

The XCFramework and the compiled model are generated artefacts, so they're gitignored. Install them from a training run:

```bash
packages/wmg_detector/tool/install.sh training/runs/<run>
```

`install.sh` does three things:
1. Builds `WmgFeatures.xcframework` with `features/apple/build_xcframework.sh`. The slices are macOS arm64+x86_64, iOS arm64, and simulator arm64+x86_64.
2. Exports the model to Core ML if needed, using `python -m wmg_train.export_coreml`. The export includes a parity check against PyTorch.
3. Copies both into `darwin/`.

After switching models, regenerate the parity fixture and run the on-device test:

```bash
cd training && .venv-arm64/bin/python -m wmg_train.parity_fixture runs/<run>/model.pt
cd ../frontend && flutter test integration_test/on_device_parity_test.dart -d macos
```

Run the same test with an iOS simulator id as well.

## Tests in `frontend/integration_test/`

- **`on_device_parity_test.dart`:** scores a deterministic synthetic signal through the real native path and checks it against PyTorch (P(AI) within 0.01).
- **`live_mic_test.dart`:** launches the app, presses the mic, records through the device microphone and prints the on-device estimate. Play music near the mic while it runs, and pass `--dart-define=LIVE_IDENTIFY=false` to skip ACRCloud.
