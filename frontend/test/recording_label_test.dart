import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:wmg_frontend/data/labels/recording_label.dart';
import 'package:wmg_frontend/data/personalization/head_trainer.dart';
import 'package:wmg_frontend/data/personalization/personal_model.dart';
import 'package:wmg_frontend/features/recognizer/models/clip_result.dart';
import 'package:wmg_frontend/features/recognizer/models/spectrogram.dart';
import 'package:wmg_frontend/features/results/data/session_history_store.dart';
import 'package:wmg_frontend/features/results/models/saved_session.dart';

SavedSession _session(String id, {RecordingLabel? label}) =>
    SavedSession(id: id, startedAt: DateTime.utc(2026, 10, 26, 1), durationS: 30, label: label);

void main() {
  test('categories carry the AI-share ranges and binary targets', () {
    expect([for (final c in AiCategory.values) (c.wire, c.aiShareMin, c.aiShareMax, c.binaryLabel)], [
      ('human', 0.0, 0.1, 0),
      ('ai_assisted', 0.1, 0.5, null),
      ('ai', 0.5, 1.0, 1),
    ]);
    expect(AiCategory.fromWire('ai_assisted'), AiCategory.aiAssisted);
  });

  test('saved sessions round-trip their label; older files have none', () {
    final s = _session('a', label: RecordingLabel(category: AiCategory.aiAssisted, labelledAt: DateTime.utc(2026)));
    final json = jsonDecode(jsonEncode(s.toJson())) as Map<String, dynamic>;
    expect(json['label'], containsPair('ai_share_min', 0.1));
    expect(SavedSession.fromJson(json).label?.category, AiCategory.aiAssisted);
    expect(SavedSession.fromJson(_session('b').toJson()).label, isNull);
  });

  group('store', () {
    late Directory tmp;
    late SessionHistoryStore store;
    setUp(() async {
      tmp = await Directory.systemTemp.createTemp('labels');
      store = SessionHistoryStore(root: tmp);
    });
    tearDown(() => tmp.delete(recursive: true));

    test('setLabel changes only the label and keeps the audio', () async {
      await store.save(_session('s1'), const Spectrogram());
      final audio = File('${tmp.path}/s1/audio.wav')..writeAsBytesSync([1, 2, 3]);
      final s = await store.setLabel('s1', RecordingLabel.now(AiCategory.aiAssisted));
      expect(s.label?.category, AiCategory.aiAssisted);
      expect((await store.list()).single.label?.category, AiCategory.aiAssisted);
      expect(audio.readAsBytesSync(), [1, 2, 3]);
      await store.setLabel('s1', null);
      expect((await store.list()).single.label, isNull);
    });

    test('a late re-save without a label keeps the label already set', () async {
      await store.save(_session('s2'), const Spectrogram());
      await store.setLabel('s2', RecordingLabel.now(AiCategory.ai));
      await store.save(_session('s2'), const Spectrogram()); // recognizer re-saving late results
      expect((await store.list()).single.label?.category, AiCategory.ai);
    });
  });

  test('training samples: old files default from the binary label; AI-assisted is kept but not trained on', () {
    final old = LabelledSample.fromJson({
      'session_id': 'x', 'label': 1, 'embedding': [0.0, 1.0], 'model_run': 'm', 'recorded_at': '2026-10-01T00:00:00Z',
    });
    expect(old.category, AiCategory.ai);
    expect(old.binary, isTrue);

    LabelledSample s(String session, AiCategory c, double x) => LabelledSample(
      sessionId: session, label: c.binaryLabel ?? -1, category: c,
      embedding: Float32List.fromList([x, 1]), modelRun: 'm', recordedAt: DateTime.utc(2026));
    final samples = [
      for (var i = 0; i < 8; i++) s('h${i % 2}', AiCategory.human, -1),
      for (var i = 0; i < 8; i++) s('a${i % 2}', AiCategory.ai, 1),
      for (var i = 0; i < 30; i++) s('p', AiCategory.aiAssisted, 0.2),
    ];
    final json = samples.last.toJson();
    expect(json, containsPair('category', 'ai_assisted'));
    expect(json, containsPair('label', -1));

    final head = const HeadTrainer().train(
      samples: samples, w0: Float64List.fromList([0, 0]), b0: 0, baseModelRun: 'm', featureSpecHash: 'h');
    // Only the 16 Human/AI windows count; 30 AI-assisted ones would otherwise trip the balance check.
    expect(head.report.aiWindows + head.report.humanWindows, 16);
    expect(exportBundle(head, samples, device: 't')['samples'], hasLength(46)); // all kept in the export
  });

  group('clip labels', () {
    late Directory tmp;
    late SessionHistoryStore store;
    setUp(() async {
      tmp = await Directory.systemTemp.createTemp('clip-labels');
      store = SessionHistoryStore(root: tmp);
    });
    tearDown(() => tmp.delete(recursive: true));

    SavedSession withClips(String id, {RecordingLabel? label}) => SavedSession(
      id: id,
      startedAt: DateTime.utc(2026, 10, 27),
      durationS: 90,
      label: label,
      clips: const [ClipResult(index: 1, id: 'c1'), ClipResult(index: 0, id: 'c0')],
    );

    test('a clip label overrides the recording label and round-trips', () async {
      await store.save(withClips('s', label: RecordingLabel.now(AiCategory.human)), const Spectrogram());
      final s = await store.setClipLabel('s', 1, RecordingLabel.now(AiCategory.ai));
      final back = (await store.list()).single;
      expect(back, s);
      expect([for (final c in back.clips) back.labelFor(c)?.category], [AiCategory.ai, AiCategory.human]);
      final cleared = await store.setClipLabel('s', 1, null);
      expect(cleared.labelFor(cleared.clips.first)?.category, AiCategory.human);
    });

    test('a late re-save from the recognizer keeps clip labels set meanwhile', () async {
      await store.save(withClips('s'), const Spectrogram());
      await store.setClipLabel('s', 0, RecordingLabel.now(AiCategory.aiAssisted));
      await store.save(withClips('s'), const Spectrogram()); // late results, no labels in it
      final back = (await store.list()).single;
      expect(back.clips.last.label?.category, AiCategory.aiAssisted);
      expect(back.clips.first.label, isNull);
    });
  });
}
