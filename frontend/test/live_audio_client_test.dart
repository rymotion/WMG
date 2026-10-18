import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:wmg_frontend/core/app_config.dart';
import 'package:wmg_frontend/data/api/live_audio_client.dart';
import 'package:wmg_frontend/data/models/live_message.dart';

import 'support.dart';

/// Talks to a real in-process WebSocket server implementing the guide's `/live` contract.
void main() {
  late HttpServer server;
  late List<int> frameSizes;
  late List<int> received;
  late List<String> textFrames;
  late Uri requestUri;

  setUp(() async {
    frameSizes = [];
    received = [];
    textFrames = [];
    server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    server.listen((request) async {
      requestUri = request.uri;
      final ws = await WebSocketTransformer.upgrade(request);
      ws.add(sessionMsg());
      ws.listen((frame) async {
        if (frame is List<int>) {
          frameSizes.add(frame.length);
          received.addAll(frame);
        } else if (frame is String) {
          textFrames.add(frame);
          if (jsonDecode(frame)['type'] == 'stop') {
            ws.add(clipMsg(0));
            ws.add(identificationMsg(0));
            await ws.close(1000);
          }
        }
      });
    });
  });

  tearDown(() => server.close(force: true));

  AppConfig config({bool identify = true}) =>
      AppConfig(apiBaseUrl: Uri.parse('http://127.0.0.1:${server.port}'), liveIdentify: identify);

  test('connects to /live, sends ~100 ms PCM frames, stop, and reads replies until close', () async {
    final c = config();
    final session = await LiveAudioClient(uri: c.liveAudioUri, sampleRate: c.liveSampleRate).connect();

    // 5 × 1500 bytes = 7500 → two full 3200-byte frames + 1100 flushed on stop.
    for (var i = 0; i < 5; i++) {
      session.addPcm(Uint8List.fromList(List.filled(1500, i)));
    }
    session.stop();
    final messages = await session.messages.toList();
    await session.done;

    expect(requestUri.path, '/live');
    expect(requestUri.queryParameters, {'sample_rate': '16000'});
    expect(frameSizes, [3200, 3200, 1100]);
    expect(received, [for (var i = 0; i < 5; i++) ...List.filled(1500, i)]);
    expect(session.bytesSent, 7500);
    expect(textFrames, ['{"type":"stop"}']);
    expect(messages.map((m) => m.runtimeType), [LiveSessionStarted, LiveClip, LiveClipIdentified]);
    expect(session.closeCode, 1000);
  });

  test('identify=false is passed through', () {
    expect(config(identify: false).liveAudioUri.queryParameters, {'sample_rate': '16000', 'identify': 'false'});
  });

  test('https base URL maps to wss', () {
    final uri = AppConfig(apiBaseUrl: Uri.parse('https://api.example.com')).liveAudioUri;
    expect(uri.toString(), 'wss://api.example.com/live?sample_rate=16000');
  });

  test('connect fails fast when nothing is listening', () async {
    final port = server.port;
    await server.close(force: true);
    final client = LiveAudioClient(uri: Uri.parse('ws://127.0.0.1:$port/live'), sampleRate: 16000);
    expect(client.connect(), throwsA(anything));
  });
}
