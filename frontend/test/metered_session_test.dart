import 'dart:async';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:wmg_frontend/data/recorder/audio_recorder_service.dart';
import 'package:wmg_frontend/data/repository/identification_repository.dart';
import 'package:wmg_frontend/features/recognizer/bloc/recognizer_bloc.dart';

import 'support.dart';

class _MockRecorder extends Mock implements AudioRecorderService {}

class _MockRepository extends Mock implements IdentificationRepository {}

/// One tap = one ID: the ticket rides along, and the session ends at the first match or
/// when the plan's listen window runs out.
void main() {
  late _MockRecorder recorder;
  late _MockRepository repository;
  late StreamController<Uint8List> mic;
  late FakeLiveSession session;

  setUp(() {
    recorder = _MockRecorder();
    repository = _MockRepository();
    mic = StreamController<Uint8List>();
    session = FakeLiveSession();
    when(() => recorder.hasPermission()).thenAnswer((_) async => true);
    when(
      () => repository.openLiveSession(identify: any(named: 'identify'), ticket: any(named: 'ticket')),
    ).thenAnswer((_) async => session);
    when(() => recorder.startPcmStream(sampleRate: any(named: 'sampleRate'))).thenAnswer((_) async => mic.stream);
    when(() => recorder.stop()).thenAnswer((_) async => mic.close());
  });

  Future<void> pump([int ms = 20]) => Future<void>.delayed(Duration(milliseconds: ms));

  test('passes the ticket and stops at the first match', () async {
    final bloc = RecognizerBloc(recorder: recorder, repository: repository);
    bloc.add(const RecognizerMicPressed(ticket: 't-1', listenFor: Duration(seconds: 30), stopOnMatch: true));
    await pump();
    verify(() => repository.openLiveSession(identify: true, ticket: 't-1')).called(1);

    session
      ..serverSends(sessionMsg())
      ..serverSends(clipMsg(0))
      ..serverSends(identificationMsg(0, status: 'no_match'));
    await pump();
    expect(session.stopped, isFalse, reason: 'no match yet: keep listening');

    session
      ..serverSends(clipMsg(1))
      ..serverSends(identificationMsg(1));
    await pump();
    expect(session.stopped, isTrue);
    expect(bloc.state.status, RecognizerStatus.finishing);
    await bloc.close();
  });

  test('stops on its own when the listen window runs out', () async {
    final bloc = RecognizerBloc(recorder: recorder, repository: repository);
    bloc.add(const RecognizerMicPressed(ticket: 't-2', listenFor: Duration(milliseconds: 250)));
    await pump();
    expect(bloc.state.status, RecognizerStatus.streaming);
    await pump(400);
    expect(session.stopped, isTrue);
    await bloc.close();
  });

  test('an unmetered tap keeps streaming past a match', () async {
    final bloc = RecognizerBloc(recorder: recorder, repository: repository);
    bloc.add(const RecognizerMicPressed());
    await pump();
    session
      ..serverSends(sessionMsg())
      ..serverSends(clipMsg(0))
      ..serverSends(identificationMsg(0));
    await pump(300);
    expect(session.stopped, isFalse);
    expect(bloc.state.status, RecognizerStatus.streaming);
    await bloc.close();
  });
}
