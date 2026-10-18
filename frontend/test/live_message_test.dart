import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:wmg_frontend/data/models/audio_upload.dart';
import 'package:wmg_frontend/data/models/identification.dart';
import 'package:wmg_frontend/data/models/live_message.dart';

import 'support.dart';

void main() {
  test('parses the REST list response (AudioMeta)', () {
    final upload = AudioUpload.listFromJson(jsonDecode(sampleListJson) as List<dynamic>).single;
    expect(upload.filename, 'Glass.aiff');
    expect(upload.identification.status, IdentificationStatus.noMatch);
    expect(upload.identification.code, 1001);
  });

  test('session', () {
    final msg = LiveServerMessage.parse(sessionMsg())! as LiveSessionStarted;
    expect(msg.sessionId, startsWith('6f0c'));
    expect(msg.clipSeconds, 30);
    expect(msg.sampleRate, 16000);
    expect(msg.nMels, 64);
    expect(msg.hopS, 0.032);
    expect(msg.dbFloor, -80);
    expect(msg.freqs, [0, 31.2]);
  });

  test('spectrum', () {
    final msg = LiveServerMessage.parse(spectrumMsg(t: 12, frames: 2))! as LiveSpectrum;
    expect(msg.t, 12);
    expect(msg.frames, hasLength(2));
    expect(msg.frames.first, hasLength(64));
  });

  test('clip', () {
    final msg = LiveServerMessage.parse(clipMsg(1))! as LiveClip;
    expect([msg.index, msg.id, msg.startS, msg.durationS], [1, 'clip-1', 30.0, 30.0]);
  });

  test('identification (flat, no updated_at)', () {
    final msg = LiveServerMessage.parse(identificationMsg(0))! as LiveClipIdentified;
    expect(msg.index, 0);
    expect(msg.identification.status, IdentificationStatus.matched);
    expect(msg.identification.matches.single.title, 'Song 0');
    expect(msg.identification.updatedAt, isNull);
  });

  test('every documented status maps', () {
    for (final (raw, status) in [
      ('matched', IdentificationStatus.matched),
      ('possible_match', IdentificationStatus.possibleMatch),
      ('no_match', IdentificationStatus.noMatch),
      ('error', IdentificationStatus.error),
      ('disabled', IdentificationStatus.disabled),
      ('skipped', IdentificationStatus.skipped),
    ]) {
      final msg = LiveServerMessage.parse(identificationMsg(0, status: raw))! as LiveClipIdentified;
      expect(msg.identification.status, status, reason: raw);
    }
  });

  test('error', () {
    final msg = LiveServerMessage.parse('{"type":"error","message":"sample_rate must be between 8000 and 96000"}');
    expect((msg! as LiveError).message, contains('sample_rate'));
  });

  test('unknown types are ignored', () {
    expect(LiveServerMessage.parse('{"type":"something_new"}'), isNull);
  });
}
