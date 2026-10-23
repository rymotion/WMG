import 'dart:math' as math;
import 'dart:typed_data';

import 'personal_model.dart';

class NotEnoughData implements Exception {
  const NotEnoughData(this.message);

  final String message;

  @override
  String toString() => message;
}

/// Fine-tunes the model's final layer (logistic regression on the embedding) on labelled
/// windows recorded on this device.
///
/// Starts from the trained weights and is pulled back toward them (L2 on w - w0), so a few
/// minutes of audio adjust the decision boundary to this device's speaker/mic instead of
/// replacing what the model learned. Classes are weighted equally.
class HeadTrainer {
  const HeadTrainer({this.l2 = 0.05, this.epochs = 400, this.learningRate = 0.01, this.minWindowsPerClass = 6});

  final double l2;
  final int epochs;
  final double learningRate;
  final int minWindowsPerClass;

  /// Refuses data that would teach "this device = one class" (same rule as server training).
  void check(List<LabelledSample> samples) {
    final ai = samples.where((s) => s.label == 1).length;
    final human = samples.length - ai;
    if (ai < minWindowsPerClass || human < minWindowsPerClass) {
      throw NotEnoughData(
        'Need at least $minWindowsPerClass labelled windows of each class '
        '(have $ai AI, $human human). Record and label more of the missing class.',
      );
    }
    final share = ai / samples.length;
    if (share < 0.2 || share > 0.8) {
      throw NotEnoughData(
        'Labelled windows are ${(share * 100).round()}% AI. Keep both classes between 20% and 80% '
        'so the layer doesn\'t learn that this device means one class.',
      );
    }
  }

  (Float64List, double) fit(List<LabelledSample> samples, Float64List w0, double b0) {
    final n = samples.length, d = w0.length;
    final ai = samples.where((s) => s.label == 1).length;
    final wPos = n / (2 * ai), wNeg = n / (2 * (n - ai));
    final w = Float64List.fromList(w0);
    var b = b0;
    // Adam
    final mw = Float64List(d), vw = Float64List(d);
    var mb = 0.0, vb = 0.0;
    const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    final gw = Float64List(d);
    for (var t = 1; t <= epochs; t++) {
      gw.fillRange(0, d, 0);
      var gb = 0.0;
      for (final s in samples) {
        final x = s.embedding;
        var z = b;
        for (var i = 0; i < d; i++) {
          z += w[i] * x[i];
        }
        final err = (sigmoid(z) - s.label) * (s.label == 1 ? wPos : wNeg) / n;
        for (var i = 0; i < d; i++) {
          gw[i] += err * x[i];
        }
        gb += err;
      }
      for (var i = 0; i < d; i++) {
        final g = gw[i] + l2 * (w[i] - w0[i]);
        mw[i] = beta1 * mw[i] + (1 - beta1) * g;
        vw[i] = beta2 * vw[i] + (1 - beta2) * g * g;
        w[i] -= learningRate * (mw[i] / (1 - math.pow(beta1, t))) / (math.sqrt(vw[i] / (1 - math.pow(beta2, t))) + eps);
      }
      final g = gb + l2 * (b - b0);
      mb = beta1 * mb + (1 - beta1) * g;
      vb = beta2 * vb + (1 - beta2) * g * g;
      b -= learningRate * (mb / (1 - math.pow(beta1, t))) / (math.sqrt(vb / (1 - math.pow(beta2, t))) + eps);
    }
    return (w, b);
  }

  static double accuracy(List<LabelledSample> samples, Float64List w, double b) {
    if (samples.isEmpty) return double.nan;
    var ok = 0;
    for (final s in samples) {
      var z = b;
      for (var i = 0; i < w.length; i++) {
        z += w[i] * s.embedding[i];
      }
      if ((z > 0 ? 1 : 0) == s.label) ok++;
    }
    return ok / samples.length;
  }

  /// Leave-one-recording-out accuracy (base vs fine-tuned), or null with < 2 recordings per class.
  (double, double)? heldOut(List<LabelledSample> samples, Float64List w0, double b0) {
    final sessions = samples.map((s) => s.sessionId).toSet();
    final perClass = [0, 1].map((c) => samples.where((s) => s.label == c).map((s) => s.sessionId).toSet().length);
    if (perClass.any((n) => n < 2)) return null;
    var okBase = 0, okTuned = 0, total = 0;
    for (final held in sessions) {
      final train = samples.where((s) => s.sessionId != held).toList();
      final test = samples.where((s) => s.sessionId == held).toList();
      final ai = train.where((s) => s.label == 1).length;
      if (ai == 0 || ai == train.length) continue;
      final (w, b) = fit(train, w0, b0);
      okBase += (accuracy(test, w0, b0) * test.length).round();
      okTuned += (accuracy(test, w, b) * test.length).round();
      total += test.length;
    }
    return total == 0 ? null : (okBase / total, okTuned / total);
  }

  PersonalHead train({
    required List<LabelledSample> samples,
    required Float64List w0,
    required double b0,
    required String baseModelRun,
    required String featureSpecHash,
  }) {
    final usable = samples.where((s) => s.modelRun == baseModelRun && s.embedding.length == w0.length).toList();
    check(usable);
    final (w, b) = fit(usable, w0, b0);
    final held = heldOut(usable, w0, b0);
    final ai = usable.where((s) => s.label == 1).length;
    final now = DateTime.now().toUtc();
    return PersonalHead(
      id: '${now.toIso8601String().replaceAll(RegExp(r'[^0-9]'), '').substring(0, 14)}-${baseModelRun.hashCode.toUnsigned(20).toRadixString(16)}',
      baseModelRun: baseModelRun,
      featureSpecHash: featureSpecHash,
      weights: w,
      bias: b,
      createdAt: now,
      report: TrainReport(
        aiWindows: ai,
        humanWindows: usable.length - ai,
        sessions: usable.map((s) => s.sessionId).toSet().length,
        accuracyBefore: accuracy(usable, w0, b0),
        accuracyAfter: accuracy(usable, w, b),
        heldOutAccuracyBefore: held?.$1,
        heldOutAccuracyAfter: held?.$2,
        l2: l2,
        epochs: epochs,
      ),
    );
  }
}
