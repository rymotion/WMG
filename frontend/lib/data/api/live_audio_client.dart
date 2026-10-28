import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:web_socket_channel/web_socket_channel.dart';

import '../models/live_message.dart';

/// Opens `/live` sessions. One socket is one session.
class LiveAudioClient {
  const LiveAudioClient({required this.uri, required this.sampleRate});

  final Uri uri;
  final int sampleRate;

  /// Connects and waits for the handshake, so an unreachable server fails here
  /// rather than after the mic is open. `identify: false` makes the server skip
  /// ACRCloud for this session; it can't turn identification back on if [uri]
  /// already disables it.
  Future<LiveAudioSession> connect({bool identify = true}) async {
    final target = identify ? uri : uri.replace(queryParameters: {...uri.queryParameters, 'identify': 'false'});
    final channel = WebSocketChannel.connect(target);
    await channel.ready;
    // 100 ms of mono s16le, the low end of the guide's 100–200 ms frame window.
    return WebSocketLiveAudioSession(channel, frameBytes: sampleRate ~/ 10 * 2);
  }
}

/// PCM goes up as binary frames, JSON messages come back.
abstract interface class LiveAudioSession {
  /// Total PCM bytes handed to the socket so far.
  int get bytesSent;

  /// Server messages. Buffered until listened to; single subscription.
  Stream<LiveServerMessage> get messages;

  /// Completes when the socket closes, for whatever reason.
  Future<void> get done;

  bool get isClosed;
  int? get closeCode;
  String? get closeReason;

  /// Queues PCM; it is sent in ~100 ms frames.
  void addPcm(Uint8List chunk);

  /// Flushes buffered audio and sends `{"type":"stop"}`. The server saves the
  /// last partial clip (if ≥ 3 s), sends pending identifications, then closes.
  void stop();

  /// Closes without the final messages.
  Future<void> close();
}

class WebSocketLiveAudioSession implements LiveAudioSession {
  WebSocketLiveAudioSession(this._channel, {required this.frameBytes}) {
    _channel.stream.listen(
      (frame) {
        if (frame is! String) return;
        try {
          final message = LiveServerMessage.parse(frame);
          if (message != null) _messages.add(message);
        } on FormatException catch (e, st) {
          _messages.addError(e, st);
        }
      },
      onError: _messages.addError,
      onDone: () {
        _closed = true;
        _messages.close();
        if (!_done.isCompleted) _done.complete();
      },
    );
  }

  final WebSocketChannel _channel;

  /// Size of each binary frame sent (the last one on stop may be shorter).
  final int frameBytes;

  final _messages = StreamController<LiveServerMessage>();
  final _done = Completer<void>();
  final _pending = BytesBuilder(copy: false);
  var _bytesSent = 0;
  var _closed = false;
  var _stopped = false;

  @override
  int get bytesSent => _bytesSent;

  @override
  Stream<LiveServerMessage> get messages => _messages.stream;

  @override
  Future<void> get done => _done.future;

  @override
  bool get isClosed => _closed;

  @override
  int? get closeCode => _channel.closeCode;

  @override
  String? get closeReason => _channel.closeReason;

  @override
  void addPcm(Uint8List chunk) {
    if (_closed || _stopped || chunk.isEmpty) return;
    _pending.add(chunk);
    while (_pending.length >= frameBytes) {
      final bytes = _pending.takeBytes();
      _send(Uint8List.sublistView(bytes, 0, frameBytes));
      if (bytes.length > frameBytes) _pending.add(Uint8List.sublistView(bytes, frameBytes));
    }
  }

  void _send(Uint8List frame) {
    _channel.sink.add(frame);
    _bytesSent += frame.lengthInBytes;
  }

  @override
  void stop() {
    if (_closed || _stopped) return;
    _stopped = true;
    if (_pending.isNotEmpty) _send(_pending.takeBytes());
    _channel.sink.add(jsonEncode({'type': 'stop'}));
  }

  @override
  Future<void> close() async {
    _closed = true;
    await _channel.sink.close();
  }
}
