import 'dart:async';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:wmg_detector/wmg_detector.dart';
import 'package:wmg_frontend/data/recorder/audio_recorder_service.dart';
import 'package:wmg_frontend/data/repository/identification_repository.dart';
import 'package:wmg_frontend/features/recognizer/bloc/recognizer_bloc.dart';
import 'package:wmg_frontend/features/recognizer/models/on_device_detection.dart';

import 'support.dart';

class _MockRecorder extends Mock implements AudioRecorderService {}

class _MockRepository extends Mock implements IdentificationRepository {}

/// Scores each chunk as one window whose logit encodes the chunk length (in seconds).
class _FakeDetector implements WmgDetector {
  final calls = <(int bytes, int sampleRate)>[];

  @override
  Future<DetectorInfo> info() async =>
      const DetectorInfo(featureSpecHash: 'abc', modelSpecHash: 'abc', modelRun: 'test-run');

  @override
  Future<DetectorResult> scorePcm16(Uint8List pcm, int sampleRate) async {
    calls.add((pcm.length, sampleRate));
    final seconds = pcm.length / (sampleRate * 2);
    return DetectorResult(
      windows: [WindowScore(startS: 0, logit: seconds >= 10 ? 2.0 : -2.0, bands: const [0.1, 0.5, 0.9])],
      inputRmsDbfs: -38,
    );
  }
}

void main() {
  const sr = 1000; // small rate keeps buffers tiny: 10 s = 20 000 bytes
  late _MockRecorder recorder;
  late _MockRepository repository;
  late StreamController<Uint8List> mic;
  late _FakeDetector detector;

  setUp(() {
    recorder = _MockRecorder();
    repository = _MockRepository();
    mic = StreamController<Uint8List>();
    detector = _FakeDetector();
    when(() => recorder.hasPermission()).thenAnswer((_) async => true);
    when(() => repository.openLiveSession()).thenAnswer((_) async => FakeLiveSession());
    when(() => recorder.startPcmStream(sampleRate: any(named: 'sampleRate'))).thenAnswer((_) async => mic.stream);
    when(() => recorder.stop()).thenAnswer((_) async => mic.close());
    when(() => recorder.cancel()).thenAnswer((_) async {});
  });

  Future<void> pump() => Future<void>.delayed(const Duration(milliseconds: 20));

  test('scores each 10 s of mic audio on device, plus the trailing partial chunk on stop', () async {
    final bloc = RecognizerBloc(recorder: recorder, repository: repository, sampleRate: sr, detector: detector);
    bloc.add(const RecognizerMicPressed());
    await pump();
    expect(bloc.state.onDevice.status, OnDeviceStatus.listening);
    expect(bloc.state.onDevice.modelRun, 'test-run');

    // 25 s of audio in 2.5 s chunks -> two full 10 s chunks scored while streaming.
    for (var i = 0; i < 10; i++) {
      mic.add(Uint8List(sr * 2 * 5 ~/ 2));
    }
    await pump();
    expect(detector.calls, [(20000, sr), (20000, sr)]);
    expect(bloc.state.onDevice.status, OnDeviceStatus.ready);
    expect(bloc.state.onDevice.windows.map((w) => w.startS), [0, 10]);

    // Stop flushes the remaining 5 s (>= 3 s minimum).
    bloc.add(const RecognizerMicPressed());
    await pump();
    expect(detector.calls.last, (10000, sr));
    final d = bloc.state.onDevice;
    expect(d.windows.map((w) => w.startS), [0, 10, 20]);
    expect(d.logit, closeTo((2 + 2 - 2) / 3, 1e-9)); // mean of window logits
    expect(d.bands, [closeTo(0.1, 1e-9), closeTo(0.5, 1e-9), closeTo(0.9, 1e-9)]);
    await bloc.close();
  });

  test('a model trained on different features is reported, not used', () async {
    final bad = _MismatchDetector();
    final bloc = RecognizerBloc(recorder: recorder, repository: repository, sampleRate: sr, detector: bad);
    bloc.add(const RecognizerMicPressed());
    await pump();
    expect(bloc.state.onDevice.status, OnDeviceStatus.failed);
    expect(bloc.state.onDevice.error, contains('expects features'));
    await bloc.close();
  });
}

class _MismatchDetector extends _FakeDetector {
  @override
  Future<DetectorInfo> info() async =>
      const DetectorInfo(featureSpecHash: 'abc', modelSpecHash: 'zzz', modelRun: 'old-run');
}
