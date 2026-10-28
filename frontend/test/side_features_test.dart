import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:wmg_frontend/data/personalization/head_trainer.dart';
import 'package:wmg_frontend/data/personalization/personal_model.dart';

LabelledSample _sample(String session, int label, List<double> emb, [List<double>? side]) => LabelledSample(
  sessionId: session,
  label: label,
  embedding: Float32List.fromList(emb),
  side: side == null ? null : Float32List.fromList(side),
  modelRun: 'run',
  recordedAt: DateTime.utc(2026),
);

void main() {
  group('side features', () {
    test('head input is embedding followed by side, zeros when absent', () {
      expect(headInput(Float32List.fromList([1, 2]), Float32List.fromList([3, 4, 5])), [1, 2, 3, 4, 5]);
      expect(headInput(Float32List.fromList([1, 2]), null), [1, 2, 0, 0, 0]);
      expect(_sample('a', 0, [1]).side, [0, 0, 0]);
    });

    test('an old head with only embedding weights ignores side features', () {
      final head = PersonalHead(
        id: 'h',
        baseModelRun: 'run',
        featureSpecHash: 'x',
        weights: Float64List.fromList([1, 1]),
        bias: 0,
        createdAt: DateTime.utc(2026),
        report: const TrainReport(aiWindows: 1, humanWindows: 1, sessions: 1, accuracyBefore: 0, accuracyAfter: 0, l2: 0, epochs: 0),
      );
      expect(head.logit(Float32List.fromList([1, 2]), Float32List.fromList([9, 9, 9])), 3);
    });

    test('samples round-trip through JSON with their side features', () {
      final s = _sample('a', 1, [0.5], [1, 2, 3]);
      final back = LabelledSample.fromJson(s.toJson());
      expect(back.side, [1, 2, 3]);
      expect(LabelledSample.fromJson(s.toJson()..remove('side')).side, [0, 0, 0]);
    });

    test('the trainer learns a side feature the embedding cannot explain', () {
      // Embedding carries no information; the first side feature says it all.
      final samples = [
        for (var i = 0; i < 8; i++) _sample('s$i', 0, [0.1], [-1, 0, 0]),
        for (var i = 0; i < 8; i++) _sample('t$i', 1, [0.1], [1, 0, 0]),
      ];
      final head = const HeadTrainer().train(
        samples: samples,
        w0: Float64List.fromList([0]),
        b0: 0,
        baseModelRun: 'run',
        featureSpecHash: 'x',
      );
      expect(head.weights.length, 1 + kSideFeatures);
      expect(head.weights[1], greaterThan(0.5));
      expect(head.report.accuracyAfter, 1.0);
      expect(head.logit(Float32List.fromList([0.1]), Float32List.fromList([1, 0, 0])), greaterThan(0));
      expect(head.logit(Float32List.fromList([0.1]), Float32List.fromList([-1, 0, 0])), lessThan(0));
    });
  });
}
