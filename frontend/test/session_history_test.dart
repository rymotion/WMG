import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:wmg_detector/wmg_detector.dart';
import 'package:wmg_frontend/data/models/identification.dart';
import 'package:wmg_frontend/data/models/track_match.dart';
import 'package:wmg_frontend/data/recorder/audio_recorder_service.dart';
import 'package:wmg_frontend/data/repository/identification_repository.dart';
import 'package:wmg_frontend/features/recognizer/bloc/recognizer_bloc.dart';
import 'package:wmg_frontend/features/recognizer/models/clip_result.dart';
import 'package:wmg_frontend/features/recognizer/models/on_device_detection.dart';
import 'package:wmg_frontend/features/recognizer/models/spectrogram.dart';
import 'package:wmg_frontend/features/recognizer/widgets/spectrogram_tiles.dart';
import 'package:wmg_frontend/features/results/data/session_history_store.dart';
import 'package:wmg_frontend/features/results/data/wav_writer.dart';
import 'package:wmg_frontend/features/results/models/saved_session.dart';

class _MockRecorder extends Mock implements AudioRecorderService {}

class _MockRepository extends Mock implements IdentificationRepository {}

class _FakeDetector implements WmgDetector {
  @override
  Future<DetectorInfo> info({String model = '10s'}) async =>
      DetectorInfo(featureSpecHash: 'abc', modelSpecHash: 'abc', modelRun: 'run-$model');

  @override
  Future<DetectorResult> scorePcm16(Uint8List pcm, int sampleRate, {String model = '10s'}) async =>
      DetectorResult(
        windows: [WindowScore(startS: 0, logit: model == '10s' ? 2.0 : 1.0, bands: const [0.1, 0.5, 0.9])],
        inputRmsDbfs: -38,
      );
}

WindowScore _w(double startS, double logit) => WindowScore(startS: startS, logit: logit, bands: const [0.1, 0.2, 0.3]);

void main() {
  late Directory tmp;

  setUp(() async => tmp = await Directory.systemTemp.createTemp('wmg_history_test'));
  tearDown(() => tmp.delete(recursive: true));

  test('WavWriter streams PCM and patches the RIFF and data sizes on close', () async {
    final f = File('${tmp.path}/a.wav');
    final w = await WavWriter.open(f, sampleRate: 32000);
    w
      ..add(Uint8List.fromList(List.filled(1000, 7)))
      ..add(Uint8List.fromList(List.filled(600, 9)));
    await w.close();
    await w.close(); // idempotent

    final bytes = await f.readAsBytes();
    final h = ByteData.sublistView(bytes);
    expect(bytes.length, 44 + 1600);
    expect(String.fromCharCodes(bytes.sublist(0, 4)), 'RIFF');
    expect(h.getUint32(4, Endian.little), 36 + 1600);
    expect(h.getUint32(24, Endian.little), 32000);
    expect(h.getUint32(40, Endian.little), 1600);
    expect(bytes[44], 7);
    expect(bytes.last, 9);
    expect(w.durationS, closeTo(1600 / 64000, 1e-9));
  });

  test('spectrogram encode/decode keeps levels and gaps', () {
    final spec = const Spectrogram(nMels: 4).append(0, [
      [-80, -40, 0, -100],
    ]).append(0.064, [
      [-20, -20, -20, -20],
    ]); // frame 1 skipped -> gap
    final back = Spectrogram.decode(spec.encode(), nMels: 4);
    expect(back.length, 3);
    expect(back.frameAt(1), isNull);
    expect(back.frameAt(0)!.toList(), [closeTo(-80, 0.2), closeTo(-40, 0.2), closeTo(0, 0.2), closeTo(-80, 0.2)]);
    expect(back.frameAt(2)!.first, closeTo(-20, 0.2));
  });

  test('tile pixels: low bins at the bottom, gaps transparent', () {
    final spec = const Spectrogram(nMels: 2).append(0, [
      [0, -80], // loud low bin, floor high bin
    ]);
    final px = spectrogramTilePixels(spec, 0, 2).buffer.asUint32List();
    // Row 0 is the top (bin 1), row 1 the bottom (bin 0); column 1 is past the end.
    expect(px[1 * 2 + 0], isNot(0)); // bin 0, loud
    expect(px[1 * 2 + 0] >>> 24, 255); // opaque
    expect(px[0 * 2 + 1], 0); // no frame: transparent
    expect(px[1 * 2 + 0], isNot(px[0 * 2 + 0])); // loud and floor differ in colour
  });

  test('SavedSession survives a JSON round trip', () {
    final s = SavedSession(
      id: '20261026-012233-123',
      startedAt: DateTime(2026, 10, 26, 1, 22, 33),
      durationS: 61.5,
      serverSessionId: 'srv-1',
      clips: const [
        ClipResult(
          index: 1,
          id: 'clip-1',
          startS: 30,
          durationS: 30,
          identification: Identification(
            status: IdentificationStatus.possibleMatch,
            matches: [
              TrackMatch(source: 'music', title: 'LOVE', artists: ['Jhené Aiko'], score: 61),
            ],
          ),
        ),
        ClipResult(index: 0, id: 'clip-0', startS: 0, durationS: 30),
      ],
      onDevice: OnDeviceDetection(
        status: OnDeviceStatus.ready,
        windows: [_w(0, 2), _w(5, -1)],
        windows5s: [_w(0, 1)],
        modelRun: 'run-10s',
      ),
      hasSpectrogram: true,
      hasAudio: true,
    );
    final back = SavedSession.fromJson(s.toJson());
    expect(back, s);
    expect(back.clips.first.identification!.status, IdentificationStatus.possibleMatch);
    expect(back.clips.first.identification!.matches.single.displayArtists, 'Jhené Aiko');
    expect(back.onDevice.windows.map((w) => w.logit), [2, -1]);
    expect(back.onDevice.pAi, closeTo(s.onDevice.pAi, 1e-12));
    expect(back.matchedTitles, ['LOVE']);
  });

  test('store saves, lists newest first, loads the spectrogram, deletes', () async {
    final store = SessionHistoryStore(root: tmp);
    final spec = const Spectrogram(nMels: 2).append(0, [
      [-10, -20],
    ]);
    for (final (id, at) in [('old', DateTime(2026, 1, 1)), ('new', DateTime(2026, 2, 1))]) {
      await store.save(SavedSession(id: id, startedAt: at, durationS: 5, nMels: 2, hasSpectrogram: true), spec);
    }
    await Directory('${tmp.path}/unfinished').create(); // no session.json: skipped

    final listed = await store.list();
    expect(listed.map((s) => s.id), ['new', 'old']);
    expect((await store.loadSpectrogram(listed.first)).length, 1);
    expect(await store.audioFile(listed.first), isNull);

    await store.delete('new');
    expect((await store.list()).map((s) => s.id), ['old']);
  });

  group('bloc', () {
    const sr = 1000; // 5 s = 10 000 bytes
    late _MockRecorder recorder;
    late _MockRepository repository;
    late StreamController<Uint8List> mic;
    late SessionHistoryStore store;

    setUp(() {
      recorder = _MockRecorder();
      repository = _MockRepository();
      mic = StreamController<Uint8List>();
      store = SessionHistoryStore(root: Directory('${tmp.path}/history'));
      when(() => recorder.hasPermission()).thenAnswer((_) async => true);
      when(() => recorder.startPcmStream(sampleRate: any(named: 'sampleRate'))).thenAnswer((_) async => mic.stream);
      when(() => recorder.stop()).thenAnswer((_) async => mic.close());
    });

    RecognizerBloc build() => RecognizerBloc(
      recorder: recorder,
      repository: repository,
      sampleRate: sr,
      detector: _FakeDetector(),
      history: store,
    );

    Future<void> pump([int ms = 20]) => Future<void>.delayed(Duration(milliseconds: ms));

    test('a finished session is saved with its scores and audio', () async {
      final bloc = build();
      bloc.add(const RecognizerMicPressed(offline: true));
      await pump();
      for (var i = 0; i < 6; i++) {
        mic.add(Uint8List(sr * 2 * 5 ~/ 2)); // 15 s
      }
      await pump();
      bloc.add(const RecognizerMicPressed());
      await pump(700); // completion + the 500 ms save debounce + late scores

      final saved = await store.list();
      expect(saved, hasLength(1));
      final s = saved.single;
      expect(s.offline, isTrue);
      expect(s.durationS, closeTo(15, 0.01));
      expect(s.onDevice.windows, isNotEmpty);
      expect(s.onDevice.windows5s, hasLength(3)); // includes the hop scored after stop
      expect(s.hasAudio, isTrue);
      final audio = await store.audioFile(s);
      expect(await audio!.length(), 44 + sr * 2 * 15);
      await bloc.close();
    });

    test('a session reset mid-recording is not kept', () async {
      final bloc = build();
      bloc.add(const RecognizerMicPressed(offline: true));
      await pump();
      mic.add(Uint8List(sr * 2 * 4));
      await pump();
      bloc.add(const RecognizerResetRequested());
      await pump(700);
      expect(await store.list(), isEmpty);
      expect(await Directory('${tmp.path}/history').list().toList(), isEmpty);
      await bloc.close();
    });
  });
}
