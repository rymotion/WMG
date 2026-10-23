// End-to-end on device: launches the real app, presses the mic, records live audio through the
// device microphone, and reads the on-device AI estimate. Play music near the mic while it runs.
//   flutter test integration_test/live_mic_test.dart -d <device> --dart-define=LIVE_IDENTIFY=false
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:wmg_frontend/features/recognizer/bloc/recognizer_bloc.dart';
import 'package:wmg_frontend/features/recognizer/view/recognizer_page.dart';
import 'package:wmg_frontend/features/recognizer/widgets/mic_button.dart';
import 'package:wmg_frontend/main.dart' as app;

const _windowsWanted = int.fromEnvironment('LIVE_WINDOWS', defaultValue: 3);

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('live microphone audio is scored on device', (tester) async {
    app.main();
    Future<void> waitFor(bool Function() done, Duration timeout) async {
      final end = DateTime.now().add(timeout);
      while (!done() && DateTime.now().isBefore(end)) {
        await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 250)));
        await tester.pump();
      }
    }

    // A fresh build can take a while to show the first frame; wait for the button.
    await waitFor(() => find.byType(MicButton).evaluate().isNotEmpty, const Duration(seconds: 30));
    RecognizerBloc bloc() => BlocProvider.of<RecognizerBloc>(tester.element(find.byType(RecognizerView)));

    for (var attempt = 0; attempt < 3 && !bloc().state.isStreaming; attempt++) {
      await tester.tap(find.byType(MicButton));
      await waitFor(() => bloc().state.isStreaming, const Duration(seconds: 10));
    }
    expect(bloc().state.isStreaming, isTrue, reason: 'recording did not start: ${bloc().state.error}');
    final deadline = DateTime.now().add(const Duration(seconds: 30 + _windowsWanted * 15));
    var announced = false;
    while (DateTime.now().isBefore(deadline) && bloc().state.onDevice.windows.length < _windowsWanted) {
      await tester.runAsync(() => Future<void>.delayed(const Duration(seconds: 1)));
      await tester.pump();
      if (!announced && bloc().state.isStreaming) {
        announced = true;
        // ignore: avoid_print
        print('RECORDING_STARTED');
      }
    }
    final d = bloc().state.onDevice;
    // ignore: avoid_print
    print('LIVE_RESULT status=${d.status.name} windows=${d.windows.length} '
        'p_ai=${d.hasScore ? d.pAi.toStringAsFixed(3) : '-'} '
        'per_window=${d.windows.map((w) => w.pAi.toStringAsFixed(2)).toList()} '
        'bands=${d.bands.map((b) => b.toStringAsFixed(2)).toList()} '
        'input_dbfs=${d.inputRmsDbfs?.toStringAsFixed(1)} model=${d.modelRun} error=${d.error}');

    await tester.tap(find.byType(MicButton)); // stop
    await tester.runAsync(() => Future<void>.delayed(const Duration(seconds: 3)));
    await tester.pump();
    expect(d.windows, isNotEmpty, reason: 'no audio window was scored — was anything playing?');
  });
}
