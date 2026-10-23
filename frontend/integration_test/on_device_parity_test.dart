// Runs the real native detector (Rust features + Core ML) inside the app on a synthetic
// signal and checks it against PyTorch reference scores from wmg_train.parity_fixture.
//   flutter test integration_test/on_device_parity_test.dart -d macos   (or an iOS simulator id)
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:wmg_detector/wmg_detector.dart';

import 'parity_expected.dart';

const _sr = 32000;
const _seconds = 25;

/// Must match synthetic_pcm() in training/wmg_train/parity_fixture.py exactly.
Uint8List syntheticPcm() {
  final n = _sr * _seconds;
  final out = ByteData(n * 2);
  var state = 12345;
  for (var i = 0; i < n; i++) {
    state = (state * 1103515245 + 12345) % 2147483648;
    final noise = state / 2147483648 * 2 - 1;
    final t = i / _sr;
    final gate = (i ~/ 16000) % 2;
    final x = 0.25 * math.sin(2 * math.pi * 220 * t) +
        0.15 * math.sin(2 * math.pi * 1760 * t) * gate +
        0.10 * math.sin(2 * math.pi * 6000 * t) * (1 - gate) +
        0.03 * noise;
    out.setInt16(i * 2, (x.clamp(-1.0, 1.0) * 32767).round(), Endian.little);
  }
  return out.buffer.asUint8List();
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('on-device scores match PyTorch for the bundled model', (tester) async {
    const detector = WmgDetector();
    final info = await detector.info();
    expect(info.featureSpecHash, parityFeatureSpecHash);
    expect(info.compatible, isTrue);
    expect(info.modelRun, parityModelRun, reason: 'regenerate parity_expected.dart for the bundled model');

    final pcm = syntheticPcm();
    final sw = Stopwatch()..start();
    final r = await detector.scorePcm16(pcm, _sr);
    sw.stop();
    // ignore: avoid_print
    print('on-device: ${r.windows.length} windows in ${sw.elapsedMilliseconds} ms; '
        'logits ${r.windows.map((w) => w.logit.toStringAsFixed(3)).toList()} vs $parityLogits');

    expect(r.windows.map((w) => w.startS).toList(), parityStarts);
    // On-device fine-tuning recomputes the score as head . embedding + bias; with the bundled
    // weights that must reproduce the model's own logit.
    final w = info.headWeights!;
    for (final win in r.windows) {
      final e = win.embedding!;
      expect(e.length, w.length);
      var z = info.headBias;
      for (var k = 0; k < w.length; k++) {
        z += w[k] * e[k];
      }
      expect(z, closeTo(win.logit, 0.02), reason: 'embedding/head path must match the model');
    }
    for (var i = 0; i < parityLogits.length; i++) {
      // fp16 Core ML vs fp32 PyTorch: compare probabilities, which is what the app shows.
      expect(r.windows[i].pAi, closeTo(sigmoid(parityLogits[i]), 0.01), reason: 'window $i');
    }
  });
}
