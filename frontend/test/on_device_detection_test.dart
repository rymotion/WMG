import 'dart:async';
import 'dart:math' as math;
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

double _sig(double x) => 1 / (1 + math.exp(-x));

/// One window per call: the 10 s model says logit 2, the 5 s model logit 1, so they disagree
/// by more than 0.1 in probability.
class _FakeDetector implements WmgDetector {
  _FakeDetector({this.has5s = true});

  final bool has5s;
  final calls = <(String model, int bytes)>[];

  @override
  Future<DetectorInfo> info({String model = '10s'}) async {
    if (model == '5s' && !has5s) throw Exception('WmgDetector5s.mlmodelc not bundled');
    return DetectorInfo(featureSpecHash: 'abc', modelSpecHash: 'abc', modelRun: 'run-$model');
  }

  @override
  Future<DetectorResult> scorePcm16(Uint8List pcm, int sampleRate, {String model = '10s'}) async {
    calls.add((model, pcm.length));
    return DetectorResult(
      windows: [
        WindowScore(
          startS: 0,
          logit: model == '10s' ? 2.0 : 1.0,
          bands: const [0.1, 0.5, 0.9],
          embedding: Float32List.fromList([1, 2, 3]),
        ),
      ],
      inputRmsDbfs: -38,
    );
  }
}

void main() {
  const sr = 1000; // small rate keeps buffers tiny: 5 s = 10 000 bytes, 10 s = 20 000 bytes
  late _MockRecorder recorder;
  late _MockRepository repository;
  late StreamController<Uint8List> mic;

  setUp(() {
    recorder = _MockRecorder();
    repository = _MockRepository();
    mic = StreamController<Uint8List>();
    when(() => recorder.hasPermission()).thenAnswer((_) async => true);
    when(() => repository.openLiveSession(identify: any(named: 'identify'))).thenAnswer((_) async => FakeLiveSession());
    when(() => recorder.startPcmStream(sampleRate: any(named: 'sampleRate'))).thenAnswer((_) async => mic.stream);
    when(() => recorder.stop()).thenAnswer((_) async => mic.close());
    when(() => recorder.cancel()).thenAnswer((_) async {});
  });

  Future<void> pump() => Future<void>.delayed(const Duration(milliseconds: 20));

  void play(double seconds) {
    for (var i = 0; i < seconds / 2.5; i++) {
      mic.add(Uint8List(sr * 2 * 5 ~/ 2)); // 2.5 s chunks
    }
  }

  test('every 5 s: last 5 s to the 5 s model, last 10 s to the 10 s model; average and delta', () async {
    final detector = _FakeDetector();
    final bloc = RecognizerBloc(recorder: recorder, repository: repository, sampleRate: sr, detector: detector);
    bloc.add(const RecognizerMicPressed());
    await pump();
    expect(bloc.state.onDevice.status, OnDeviceStatus.listening);
    expect(bloc.state.onDevice.modelRun, 'run-10s');
    expect(bloc.state.onDevice.modelRun5s, 'run-5s');

    play(5);
    await pump();
    // After 5 s only the 5 s model has answered: an early, provisional estimate.
    expect(detector.calls, [('5s', 10000)]);
    expect(bloc.state.onDevice.provisional, isTrue);
    expect(bloc.state.onDevice.pAi, closeTo(_sig(1), 1e-9));
    expect(bloc.state.onDevice.delta, isNull);

    play(20); // 25 s total
    await pump();
    final d = bloc.state.onDevice;
    expect(detector.calls.where((c) => c.$1 == '5s').map((c) => c.$2), everyElement(10000));
    expect(detector.calls.where((c) => c.$1 == '10s').map((c) => c.$2), everyElement(20000));
    expect(d.windows5s.map((w) => w.startS), [0, 5, 10, 15, 20]);
    expect(d.windows.map((w) => w.startS), [0, 5, 10, 15]); // 10 s windows ending at 10, 15, 20, 25 s
    expect(d.provisional, isFalse);
    expect(d.p10, closeTo(_sig(2), 1e-9));
    expect(d.p5, closeTo(_sig(1), 1e-9));
    expect(d.pAi, closeTo((_sig(2) + _sig(1)) / 2, 1e-9));
    expect(d.delta, closeTo(_sig(1) - _sig(2), 1e-9));
    expect(d.agree, isFalse); // |delta| ~ 0.15 > 0.1
    // Embeddings survive into state, so Training mode can label the recording.
    expect(d.windows.first.embedding, isNotNull);
    await bloc.close();
  });

  test('stop scores the trailing audio when it is at least 3 s', () async {
    final detector = _FakeDetector();
    final bloc = RecognizerBloc(recorder: recorder, repository: repository, sampleRate: sr, detector: detector);
    bloc.add(const RecognizerMicPressed());
    await pump();
    play(7.5); // one 5 s hop, 2.5 s left over
    mic.add(Uint8List(sr * 2)); // +1 s: 3.5 s left over
    await pump();
    expect(detector.calls, [('5s', 10000)]);
    bloc.add(const RecognizerMicPressed()); // stop
    await pump();
    // Rest (3.5 s) to the 5 s model; previous hop + rest (8.5 s) to the 10 s model.
    expect(detector.calls.skip(1), [('5s', 7000), ('10s', 17000)]);
    expect(bloc.state.onDevice.windows5s.map((w) => w.startS), [0, 5]);
    expect(bloc.state.onDevice.windows.map((w) => w.startS), [0]);
    await bloc.close();
  });

  test('without a bundled 5 s model the 10 s model runs alone', () async {
    final detector = _FakeDetector(has5s: false);
    final bloc = RecognizerBloc(recorder: recorder, repository: repository, sampleRate: sr, detector: detector);
    bloc.add(const RecognizerMicPressed());
    await pump();
    play(15);
    await pump();
    expect(detector.calls.map((c) => c.$1), ['10s', '10s']);
    final d = bloc.state.onDevice;
    expect(d.windows5s, isEmpty);
    expect(d.pAi, closeTo(_sig(2), 1e-9));
    expect(d.delta, isNull);
    expect(d.agree, isNull);
    expect(d.status, isNot(OnDeviceStatus.failed));
    await bloc.close();
  });

  test('a model trained on different features is reported, not used', () async {
    final bloc = RecognizerBloc(recorder: recorder, repository: repository, sampleRate: sr, detector: _MismatchDetector());
    bloc.add(const RecognizerMicPressed());
    await pump();
    expect(bloc.state.onDevice.status, OnDeviceStatus.failed);
    expect(bloc.state.onDevice.error, contains('expects features'));
    await bloc.close();
  });
}

class _MismatchDetector extends _FakeDetector {
  @override
  Future<DetectorInfo> info({String model = '10s'}) async =>
      const DetectorInfo(featureSpecHash: 'abc', modelSpecHash: 'zzz', modelRun: 'old-run');
}
