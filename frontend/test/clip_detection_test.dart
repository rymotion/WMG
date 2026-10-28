import 'package:flutter_test/flutter_test.dart';
import 'package:wmg_detector/wmg_detector.dart';
import 'package:wmg_frontend/features/recognizer/models/on_device_detection.dart';

void main() {
  WindowScore w(double startS, double logit) => WindowScore(startS: startS, logit: logit, bands: const [0, 0, 0]);

  test('slice keeps only the windows that start inside a clip', () {
    final d = OnDeviceDetection(
      status: OnDeviceStatus.ready,
      windows: [w(0, 2), w(10, 2), w(20, -2), w(30, -2)],
      windows5s: [for (var s = 0.0; s < 40; s += 5) w(s, s < 30 ? 1 : -1)],
    );

    final clip0 = d.slice(0, 30); // first 30 s clip
    expect(clip0.windows.map((x) => x.startS), [0, 10, 20]);
    expect(clip0.windows5s.map((x) => x.startS), [0, 5, 10, 15, 20, 25]);
    expect(clip0.p10, closeTo(sigmoid(2 / 3), 1e-9));
    expect(clip0.p5, closeTo(sigmoid(1), 1e-9));

    final clip1 = d.slice(30, 60);
    expect(clip1.windows.single.startS, 30);
    expect(clip1.p5, closeTo(sigmoid(-1), 1e-9));

    expect(d.slice(60, 90).hasScore, isFalse);
  });
}
