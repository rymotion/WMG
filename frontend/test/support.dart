import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:wmg_frontend/data/api/live_audio_client.dart';
import 'package:wmg_frontend/data/models/live_message.dart';

const sampleListJson =
    '[{"id":"ef864cac-5be7-4e03-b93f-e579e6ae2059","filename":"Glass.aiff","extension":"aiff",'
    '"size_bytes":475278,"duration_s":1.6500833333333331,"sample_rate":48000,"channels":2,'
    '"codec":"pcm_s24be","created_at":"2026-10-18T09:17:19.637436543Z","identification":'
    '{"status":"no_match","code":1001,"message":"No result","window_offset_s":0.0,"matches":[],'
    '"updated_at":"2026-10-18T09:17:20.327544752Z"}}]';

/// Messages shaped exactly like the examples in the live streaming guide.
String sessionMsg({double clipSeconds = 30}) => jsonEncode({
  'type': 'session',
  'session_id': '6f0c0000-0000-0000-0000-000000000000',
  'clip_seconds': clipSeconds,
  'sample_rate': 16000,
  'spectrum': {
    'n_mels': 64,
    'hop_s': 0.032,
    'db_floor': -80,
    'freqs': [0, 31.2],
  },
});

String spectrumMsg({double t = 0, int frames = 3}) =>
    jsonEncode({'type': 'spectrum', 't': t, 'frames': List.generate(frames, (_) => List.filled(64, -40.0))});

String clipMsg(int index) =>
    jsonEncode({'type': 'clip', 'index': index, 'id': 'clip-$index', 'start_s': index * 30.0, 'duration_s': 30.0});

String identificationMsg(int index, {String status = 'matched'}) => jsonEncode({
  'type': 'identification',
  'index': index,
  'id': 'clip-$index',
  'status': status,
  'matches': [
    if (status == 'matched')
      {
        'source': 'music',
        'title': 'Song $index',
        'artists': ['Artist'],
        'album': 'Album',
        'score': 100,
      },
  ],
});

/// In-memory stand-in for the WebSocket session.
class FakeLiveSession implements LiveAudioSession {
  final sent = <Uint8List>[];
  final _messages = StreamController<LiveServerMessage>();
  final _done = Completer<void>();
  var stopped = false;
  var closedByClient = false;
  var _closed = false;
  int? _closeCode;

  void serverSends(String json) => _messages.add(LiveServerMessage.parse(json)!);

  /// Simulates the server closing the socket.
  void serverCloses([int? code = 1000]) {
    if (_closed) return;
    _closed = true;
    _closeCode = code;
    _messages.close();
    if (!_done.isCompleted) _done.complete();
  }

  @override
  int get bytesSent => sent.fold(0, (n, c) => n + c.lengthInBytes);

  @override
  Stream<LiveServerMessage> get messages => _messages.stream;

  @override
  Future<void> get done => _done.future;

  @override
  bool get isClosed => _closed;

  @override
  int? get closeCode => _closeCode;

  @override
  String? get closeReason => null;

  @override
  void addPcm(Uint8List chunk) => sent.add(chunk);

  @override
  void stop() => stopped = true;

  @override
  Future<void> close() async {
    closedByClient = true;
    serverCloses(1005);
  }
}
