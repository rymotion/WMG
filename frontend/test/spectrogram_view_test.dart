import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:wmg_detector/wmg_detector.dart';
import 'package:wmg_frontend/features/recognizer/models/spectrogram.dart';
import 'package:wmg_frontend/features/recognizer/widgets/spectrogram_view.dart';

void main() {
  // 60 s at the server's 32 ms hop; the view shows 320 frames = 10.24 s.
  final spec = const Spectrogram().append(0, List.generate(1875, (_) => List.filled(64, -40.0)));
  final windows = [
    for (var s = 0.0; s <= 50; s += 5) WindowScore(startS: s, logit: s / 10 - 2, bands: const [0, 0, 0]),
  ];
  final scores = <ScoreSeries>[(windows: windows, windowS: 10, color: Colors.blue, label: '10 s')];

  Widget host(Widget child) => MaterialApp(home: Scaffold(body: child));

  testWidgets('follows the live edge, scrubs on drag, and seeks for the controller', (tester) async {
    final controller = SpectrogramController();
    await tester.pumpWidget(host(SpectrogramView(spectrogram: spec, controller: controller, scores: scores)));
    expect(find.text('P(AI)'), findsOneWidget);
    expect(find.text('10 s'), findsOneWidget);
    expect(find.text('0:49 – 1:00'), findsOneWidget); // 60 s - 10.24 s
    expect(find.byTooltip('Back to live'), findsNothing);

    // Dragging right goes back in time.
    await tester.drag(find.byType(SpectrogramView), const Offset(400, 0));
    await tester.pump();
    expect(find.text('0:44 – 0:55'), findsOneWidget);
    expect(find.byTooltip('Back to live'), findsOneWidget);

    // Seek puts the target just inside the left edge.
    controller.seekTo(40);
    await tester.pumpAndSettle();
    expect(find.text('0:38 – 0:49'), findsOneWidget);

    await tester.tap(find.byTooltip('Back to live'));
    await tester.pump();
    expect(find.text('0:49 – 1:00'), findsOneWidget);
  });

  testWidgets('pinch zooms, keeping the live edge pinned; double-tap resets', (tester) async {
    await tester.pumpWidget(host(SpectrogramView(spectrogram: spec, scores: scores)));
    final centre = tester.getCenter(find.byType(SpectrogramView));

    // Fingers 200 px apart spread to 400 px: 2× zoom, 10.24 s -> 5.12 s.
    final a = await tester.startGesture(centre - const Offset(100, 0), pointer: 1);
    final b = await tester.startGesture(centre + const Offset(100, 0), pointer: 2);
    await a.moveBy(const Offset(-100, 0));
    await b.moveBy(const Offset(100, 0));
    await a.up();
    await b.up();
    await tester.pump();
    expect(find.text('0:54 – 1:00'), findsOneWidget);
    expect(find.byTooltip('Back to live'), findsNothing); // still following live
    await tester.pump(const Duration(milliseconds: 400)); // let the pinch's taps expire

    await tester.tap(find.byType(SpectrogramView));
    await tester.pump(const Duration(milliseconds: 50));
    await tester.tap(find.byType(SpectrogramView));
    await tester.pumpAndSettle();
    expect(find.text('0:49 – 1:00'), findsOneWidget);
  });

  testWidgets('pinch while scrubbed keeps the pinched time under the fingers', (tester) async {
    final controller = SpectrogramController();
    await tester.pumpWidget(host(SpectrogramView(spectrogram: spec, controller: controller, scores: scores)));
    controller.seekTo(20); // view 18.976 – 29.216 s
    await tester.pumpAndSettle();
    final centre = tester.getCenter(find.byType(SpectrogramView)); // ~24.1 s

    // Pinch in to half the zoom: 20.48 s wide, centred on the same time.
    final a = await tester.startGesture(centre - const Offset(200, 0), pointer: 1);
    final b = await tester.startGesture(centre + const Offset(200, 0), pointer: 2);
    await a.moveBy(const Offset(100, 0));
    await b.moveBy(const Offset(-100, 0));
    await a.up();
    await b.up();
    await tester.pump();
    expect(find.text('0:13 – 0:34'), findsOneWidget);
    await tester.pumpAndSettle(const Duration(milliseconds: 400)); // double-tap timeout
  });

  testWidgets('offline: draws the P(AI) chart without a spectrogram', (tester) async {
    await tester.pumpWidget(host(SpectrogramView(spectrogram: const Spectrogram(), scores: scores)));
    expect(find.text('Spectrogram needs the server'), findsOneWidget);
    expect(find.text('P(AI)'), findsOneWidget);
  });

  testWidgets('nothing to show yet', (tester) async {
    await tester.pumpWidget(host(const SpectrogramView(spectrogram: Spectrogram())));
    expect(find.text('Waiting for spectrum…'), findsOneWidget);
    expect(find.text('P(AI)'), findsNothing);
  });
}
