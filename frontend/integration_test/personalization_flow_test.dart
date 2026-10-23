// On-device personalisation with the real detector and storage: label recordings, retrain the
// final layer in an isolate, persist it, and build the .wmgmodel bundle.
//   flutter test integration_test/personalization_flow_test.dart -d macos   (or a simulator id)
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path_provider/path_provider.dart';
import 'package:wmg_detector/wmg_detector.dart';
import 'package:wmg_frontend/data/api/spectro_api_client.dart';
import 'package:wmg_frontend/data/personalization/personal_model.dart';
import 'package:wmg_frontend/data/personalization/personalization_store.dart';
import 'package:wmg_frontend/features/personalization/cubit/personalization_cubit.dart';

/// 30 s of synthetic audio; `bright` recordings are tones + hiss, the others low tones.
Uint8List _recording(int seed, {required bool bright}) {
  const sr = 32000, n = sr * 30;
  final rng = math.Random(seed);
  final out = ByteData(n * 2);
  for (var i = 0; i < n; i++) {
    final t = i / sr;
    final x = bright
        ? 0.15 * math.sin(2 * math.pi * 3000 * t) + 0.08 * (rng.nextDouble() * 2 - 1)
        : 0.3 * math.sin(2 * math.pi * (110 + seed) * t) + 0.02 * (rng.nextDouble() * 2 - 1);
    out.setInt16(i * 2, (x.clamp(-1.0, 1.0) * 32767).round(), Endian.little);
  }
  return out.buffer.asUint8List();
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('label, retrain on device, persist and export', (tester) async {
    final root = Directory('${(await getTemporaryDirectory()).path}/personalization_test');
    if (await root.exists()) await root.delete(recursive: true);
    const detector = WmgDetector();
    final cubit = PersonalizationCubit(
      store: PersonalizationStore(root: root),
      detector: detector,
      api: SpectroApiClient(baseUrl: Uri.parse('http://127.0.0.1:9')),
    );
    await cubit.load();
    expect(cubit.state.base?.headWeights, isNotNull);

    // Two recordings per class, labelled as a user would after each recording.
    for (var i = 0; i < 4; i++) {
      final bright = i.isEven;
      final r = await detector.scorePcm16(_recording(i, bright: bright), 32000);
      await cubit.labelRecording('rec-$i', bright ? 1 : 0, r.windows);
    }
    final c = cubit.state.counts;
    expect(c.aiRecordings, 2);
    expect(c.humanRecordings, 2);

    await cubit.retrain();
    final head = cubit.state.head;
    expect(head, isNotNull, reason: cubit.state.message);
    // ignore: avoid_print
    print('PERSONALIZE ${cubit.state.message}');
    expect(head!.report.accuracyAfter, greaterThanOrEqualTo(head.report.accuracyBefore));
    expect(cubit.state.activeHead, isNotNull);

    // Persisted: a fresh cubit on the same store sees the head and samples.
    final again = PersonalizationCubit(store: PersonalizationStore(root: root), detector: detector,
        api: SpectroApiClient(baseUrl: Uri.parse('http://127.0.0.1:9')));
    await again.load();
    expect(again.state.head?.id, head.id);
    expect(again.state.counts.aiWindows, c.aiWindows);

    // The export bundle is what the backend / Python merge expect.
    final samples = await PersonalizationStore(root: root).loadSamples();
    final bundle = jsonDecode(encodeBundle(exportBundle(head, samples, device: 'test'))) as Map<String, dynamic>;
    expect(bundle['format'], 'wmg-personal-head/1');
    expect((bundle['head']['weights'] as List).length, head.weights.length);
    expect((bundle['samples'] as List).length, c.aiWindows + c.humanWindows);
    await root.delete(recursive: true);
  });
}
