import 'dart:async';
import 'dart:typed_data';

import 'package:bloc_test/bloc_test.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:wmg_frontend/data/models/identification.dart';
import 'package:wmg_frontend/data/recorder/audio_recorder_service.dart';
import 'package:wmg_frontend/data/repository/identification_repository.dart';
import 'package:wmg_frontend/features/recognizer/bloc/recognizer_bloc.dart';

import 'support.dart';

class _MockRecorder extends Mock implements AudioRecorderService {}

class _MockRepository extends Mock implements IdentificationRepository {}

void main() {
  late _MockRecorder recorder;
  late _MockRepository repository;
  late StreamController<Uint8List> mic;
  late FakeLiveSession session;

  final chunkA = Uint8List.fromList(List.filled(640, 1));
  final chunkB = Uint8List.fromList(List.filled(640, 2));

  setUp(() {
    recorder = _MockRecorder();
    repository = _MockRepository();
    mic = StreamController<Uint8List>();
    session = FakeLiveSession();
    when(() => recorder.hasPermission()).thenAnswer((_) async => true);
    when(() => repository.openLiveSession()).thenAnswer((_) async => session);
    when(() => recorder.startPcmStream(sampleRate: any(named: 'sampleRate'))).thenAnswer((_) async => mic.stream);
    when(() => recorder.stop()).thenAnswer((_) async => mic.close());
    when(() => recorder.cancel()).thenAnswer((_) async {});
  });

  RecognizerBloc build() => RecognizerBloc(recorder: recorder, repository: repository);

  Future<void> pump() => Future<void>.delayed(const Duration(milliseconds: 20));

  blocTest<RecognizerBloc, RecognizerState>(
    'full session: stream PCM, live messages, stop, server flushes and closes',
    build: build,
    act: (bloc) async {
      bloc.add(const RecognizerMicPressed());
      await pump();
      session
        ..serverSends(sessionMsg())
        ..serverSends(spectrumMsg(t: 0))
        ..serverSends(clipMsg(0))
        ..serverSends(identificationMsg(0));
      mic
        ..add(chunkA)
        ..add(chunkB);
      await pump();
      bloc.add(const RecognizerMicPressed()); // stop
      await pump();
      expect(bloc.state.status, RecognizerStatus.finishing);
      expect(session.stopped, isTrue);
      // Server saves the last partial clip, sends its identification, then closes.
      session
        ..serverSends(clipMsg(1))
        ..serverSends(identificationMsg(1, status: 'no_match'))
        ..serverCloses(1000);
    },
    wait: const Duration(milliseconds: 50),
    verify: (bloc) {
      final s = bloc.state;
      expect(s.status, RecognizerStatus.completed);
      expect(s.error, isNull);
      expect(session.sent, [chunkA, chunkB]);
      expect(s.bytesStreamed, 1280);
      expect(s.session?.clipSeconds, 30);
      expect(s.spectrogram.frames, hasLength(3));
      expect(s.spectrogram.dbFloor, -80);
      expect(s.clips.map((c) => c.index), [1, 0]); // newest first
      expect(s.clips.last.identification?.matches.single.title, 'Song 0');
      expect(s.clips.first.identification?.status, IdentificationStatus.noMatch);
      expect(session.closedByClient, isFalse);
    },
  );

  blocTest<RecognizerBloc, RecognizerState>(
    'identification before its clip message still lands on the right clip',
    build: build,
    act: (bloc) async {
      bloc.add(const RecognizerMicPressed());
      await pump();
      session
        ..serverSends(identificationMsg(0))
        ..serverSends(clipMsg(0));
    },
    wait: const Duration(milliseconds: 30),
    verify: (bloc) {
      final clip = bloc.state.clips.single;
      expect(clip.startS, 0);
      expect(clip.identification?.status, IdentificationStatus.matched);
    },
  );

  blocTest<RecognizerBloc, RecognizerState>(
    'server error + 1008 close fails with the server message and stops the mic',
    build: build,
    act: (bloc) async {
      bloc.add(const RecognizerMicPressed());
      await pump();
      session
        ..serverSends('{"type":"error","message":"session limit reached"}')
        ..serverCloses(1008);
    },
    wait: const Duration(milliseconds: 50),
    verify: (bloc) {
      expect(bloc.state.status, RecognizerStatus.failure);
      expect(bloc.state.error, 'session limit reached');
      verify(() => recorder.stop()).called(1);
    },
  );

  blocTest<RecognizerBloc, RecognizerState>(
    'unexpected drop while streaming keeps saved clips and reports it',
    build: build,
    act: (bloc) async {
      bloc.add(const RecognizerMicPressed());
      await pump();
      session
        ..serverSends(clipMsg(0))
        ..serverCloses(1006);
    },
    wait: const Duration(milliseconds: 50),
    verify: (bloc) {
      expect(bloc.state.status, RecognizerStatus.failure);
      expect(bloc.state.error, contains('code 1006'));
      expect(bloc.state.clips, hasLength(1));
    },
  );

  blocTest<RecognizerBloc, RecognizerState>(
    'server never closes after stop → client closes after the finish timeout',
    build: () =>
        RecognizerBloc(recorder: recorder, repository: repository, finishTimeout: const Duration(milliseconds: 30)),
    act: (bloc) async {
      bloc.add(const RecognizerMicPressed());
      await pump();
      bloc.add(const RecognizerMicPressed());
    },
    wait: const Duration(milliseconds: 100),
    verify: (bloc) {
      expect(session.closedByClient, isTrue);
      expect(bloc.state.status, RecognizerStatus.completed);
    },
  );

  blocTest<RecognizerBloc, RecognizerState>(
    'reset while streaming closes the socket and returns to idle',
    build: build,
    act: (bloc) async {
      bloc.add(const RecognizerMicPressed());
      await pump();
      bloc.add(const RecognizerResetRequested());
    },
    wait: const Duration(milliseconds: 50),
    verify: (bloc) {
      expect(session.closedByClient, isTrue);
      expect(bloc.state, const RecognizerState());
    },
  );

  blocTest<RecognizerBloc, RecognizerState>(
    'permission is checked before the socket is opened',
    setUp: () => when(() => recorder.hasPermission()).thenAnswer((_) async => false),
    build: build,
    act: (bloc) => bloc.add(const RecognizerMicPressed()),
    expect: () => [
      isA<RecognizerState>().having((s) => s.status, 'status', RecognizerStatus.connecting),
      isA<RecognizerState>()
          .having((s) => s.status, 'status', RecognizerStatus.failure)
          .having((s) => s.error, 'error', contains('permission')),
    ],
    verify: (_) => verifyNever(() => repository.openLiveSession()),
  );

  blocTest<RecognizerBloc, RecognizerState>(
    'unreachable server fails before the mic opens',
    setUp: () => when(() => repository.openLiveSession()).thenThrow(Exception('refused')),
    build: build,
    act: (bloc) => bloc.add(const RecognizerMicPressed()),
    verify: (bloc) {
      expect(bloc.state.error, contains('Could not reach'));
      verifyNever(() => recorder.startPcmStream(sampleRate: any(named: 'sampleRate')));
    },
  );

  test('pcmLevel maps silence to 0 and full scale to 1', () {
    expect(AudioRecorderService.pcmLevel(Uint8List(320)), 0);
    final loud = ByteData(320);
    for (var i = 0; i < 160; i++) {
      loud.setInt16(i * 2, i.isEven ? 32767 : -32768, Endian.little);
    }
    expect(AudioRecorderService.pcmLevel(loud.buffer.asUint8List()), closeTo(1, 0.01));
  });
}
