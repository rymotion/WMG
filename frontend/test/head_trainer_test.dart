import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:wmg_frontend/data/personalization/head_trainer.dart';
import 'package:wmg_frontend/data/personalization/personal_model.dart';

const _d = 16;

/// Two separable clusters; the base head (w0) leans the wrong way, as when the device's
/// speaker/mic makes AI windows look human.
List<LabelledSample> _samples(math.Random rng, {required int ai, required int human, int sessions = 4}) {
  LabelledSample s(int label, int i) {
    final e = Float32List(_d);
    for (var k = 0; k < _d; k++) {
      e[k] = (rng.nextDouble() - 0.5) * 0.5 + (k == 0 ? (label == 1 ? 1.0 : -1.0) : 0.0);
    }
    return LabelledSample(
      sessionId: '${label == 1 ? 'ai' : 'hu'}-${i % sessions}',
      label: label,
      embedding: e,
      modelRun: 'base',
      recordedAt: DateTime(2026),
    );
  }

  return [for (var i = 0; i < ai; i++) s(1, i), for (var i = 0; i < human; i++) s(0, i)];
}

void main() {
  final w0 = Float64List(_d)..[0] = -0.5; // base says "human" for AI-like windows
  const b0 = -1.0;

  test('fine-tuning fixes a base layer that misreads this device', () {
    final samples = _samples(math.Random(1), ai: 24, human: 24);
    expect(HeadTrainer.accuracy(samples, w0, b0), lessThan(0.6));
    final head = const HeadTrainer().train(
      samples: samples, w0: w0, b0: b0, baseModelRun: 'base', featureSpecHash: 'abc');
    expect(head.report.accuracyAfter, greaterThan(0.95));
    expect(head.report.heldOutAccuracyAfter, greaterThan(0.9)); // leave-one-recording-out
    expect(head.report.heldOutAccuracyBefore, lessThan(0.6));
    expect(head.logit(samples.first.embedding), greaterThan(0)); // an AI window now scores AI
  });

  test('refuses too little data or one-sided data', () {
    final rng = math.Random(2);
    expect(() => const HeadTrainer().check(_samples(rng, ai: 3, human: 20)), throwsA(isA<NotEnoughData>()));
    expect(() => const HeadTrainer().check(_samples(rng, ai: 10, human: 60)), throwsA(isA<NotEnoughData>()));
    const HeadTrainer().check(_samples(rng, ai: 10, human: 12)); // balanced enough: no throw
  });

  test('samples from a different base model are ignored', () {
    final samples = _samples(math.Random(3), ai: 10, human: 10);
    final other = samples.map((s) => LabelledSample(
      sessionId: s.sessionId, label: s.label, embedding: s.embedding, modelRun: 'other', recordedAt: s.recordedAt));
    expect(
      () => const HeadTrainer().train(samples: other.toList(), w0: w0, b0: b0, baseModelRun: 'base', featureSpecHash: 'x'),
      throwsA(isA<NotEnoughData>()),
    );
  });

  test('export bundle round-trips the head', () {
    final samples = _samples(math.Random(4), ai: 8, human: 8);
    final head = const HeadTrainer().train(samples: samples, w0: w0, b0: b0, baseModelRun: 'base', featureSpecHash: 'abc');
    final bundle = exportBundle(head, samples, device: 'test');
    expect(bundle['format'], 'wmg-personal-head/1');
    expect((bundle['samples'] as List).length, 16);
    final back = PersonalHead.fromJson(bundle);
    expect(back.logit(samples.first.embedding), closeTo(head.logit(samples.first.embedding), 1e-9));
  });
}
