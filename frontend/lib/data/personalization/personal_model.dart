import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import '../labels/recording_label.dart';

/// One labelled 10 s window: the model's final-layer input, never the audio itself.
class LabelledSample {
  LabelledSample({
    required this.sessionId,
    required this.label,
    required this.embedding,
    required this.modelRun,
    required this.recordedAt,
    this.note = '',
    AiCategory? category,
  }) : category = category ?? AiCategory.fromBinary(label);

  factory LabelledSample.fromJson(Map<String, dynamic> j) => LabelledSample(
    sessionId: j['session_id'] as String,
    label: j['label'] as int,
    // Samples stored before the AI-assisted label only have the binary label.
    category: j['category'] == null ? null : AiCategory.fromWire(j['category'] as String),
    embedding: Float32List.fromList([for (final v in j['embedding'] as List) (v as num).toDouble()]),
    modelRun: j['model_run'] as String,
    recordedAt: DateTime.parse(j['recorded_at'] as String),
    note: j['note'] as String? ?? '',
  );

  final String sessionId;

  /// Binary target: 1 = AI-generated, 0 = human, -1 = not used for binary training (AI-assisted).
  final int label;

  /// Human / AI-assisted / AI-generated, with its AI-share range.
  final AiCategory category;

  /// Usable by the current Human-vs-AI layer.
  bool get binary => category.binaryLabel != null;
  final Float32List embedding;

  /// Embeddings come from one specific base model; they can't be mixed across models.
  final String modelRun;
  final DateTime recordedAt;
  final String note;

  Map<String, dynamic> toJson() => {
    'session_id': sessionId,
    'label': label,
    'category': category.wire,
    'ai_share_min': category.aiShareMin,
    'ai_share_max': category.aiShareMax,
    'embedding': embedding,
    'model_run': modelRun,
    'recorded_at': recordedAt.toUtc().toIso8601String(),
    'note': note,
  };
}

/// Outcome of an on-device fine-tune, kept with the head and exported with it.
class TrainReport {
  const TrainReport({
    required this.aiWindows,
    required this.humanWindows,
    required this.sessions,
    required this.accuracyBefore,
    required this.accuracyAfter,
    this.heldOutAccuracyBefore,
    this.heldOutAccuracyAfter,
    required this.l2,
    required this.epochs,
  });

  factory TrainReport.fromJson(Map<String, dynamic> j) => TrainReport(
    aiWindows: j['ai_windows'] as int,
    humanWindows: j['human_windows'] as int,
    sessions: j['sessions'] as int,
    accuracyBefore: (j['accuracy_before'] as num).toDouble(),
    accuracyAfter: (j['accuracy_after'] as num).toDouble(),
    heldOutAccuracyBefore: (j['held_out_accuracy_before'] as num?)?.toDouble(),
    heldOutAccuracyAfter: (j['held_out_accuracy_after'] as num?)?.toDouble(),
    l2: (j['l2'] as num).toDouble(),
    epochs: j['epochs'] as int,
  );

  final int aiWindows;
  final int humanWindows;
  final int sessions;

  /// Window accuracy at 0.5 on all labelled windows, with the base vs fine-tuned layer.
  final double accuracyBefore;
  final double accuracyAfter;

  /// Leave-one-recording-out: each recording scored by a layer fitted without it. This is the
  /// honest estimate; null when there are too few recordings per class.
  final double? heldOutAccuracyBefore;
  final double? heldOutAccuracyAfter;
  final double l2;
  final int epochs;

  Map<String, dynamic> toJson() => {
    'ai_windows': aiWindows,
    'human_windows': humanWindows,
    'sessions': sessions,
    'accuracy_before': accuracyBefore,
    'accuracy_after': accuracyAfter,
    'held_out_accuracy_before': heldOutAccuracyBefore,
    'held_out_accuracy_after': heldOutAccuracyAfter,
    'l2': l2,
    'epochs': epochs,
  };
}

/// A fine-tuned final layer for one base model: logit = weights . embedding + bias.
class PersonalHead {
  PersonalHead({
    required this.id,
    required this.baseModelRun,
    required this.featureSpecHash,
    required this.weights,
    required this.bias,
    required this.createdAt,
    required this.report,
  });

  factory PersonalHead.fromJson(Map<String, dynamic> j) => PersonalHead(
    id: j['id'] as String,
    baseModelRun: j['base_model_run'] as String,
    featureSpecHash: j['feature_spec_hash'] as String,
    weights: Float64List.fromList([for (final v in j['head']['weights'] as List) (v as num).toDouble()]),
    bias: (j['head']['bias'] as num).toDouble(),
    createdAt: DateTime.parse(j['created_at'] as String),
    report: TrainReport.fromJson(j['train'] as Map<String, dynamic>),
  );

  final String id;
  final String baseModelRun;
  final String featureSpecHash;
  final Float64List weights;
  final double bias;
  final DateTime createdAt;
  final TrainReport report;

  double logit(Float32List embedding) {
    var s = bias;
    for (var i = 0; i < weights.length; i++) {
      s += weights[i] * embedding[i];
    }
    return s;
  }

  Map<String, dynamic> toJson() => {
    'id': id,
    'base_model_run': baseModelRun,
    'feature_spec_hash': featureSpecHash,
    'created_at': createdAt.toUtc().toIso8601String(),
    'head': {'weights': weights, 'bias': bias},
    'train': report.toJson(),
  };
}

/// The `.wmgmodel` file sent back to the computer (`python -m wmg_train.personalize merge`).
Map<String, dynamic> exportBundle(PersonalHead head, List<LabelledSample> samples, {required String device}) => {
  'format': 'wmg-personal-head/1',
  'device': device,
  'exported_at': DateTime.now().toUtc().toIso8601String(),
  ...head.toJson(),
  'samples': [for (final s in samples.where((s) => s.modelRun == head.baseModelRun)) s.toJson()],
};

String encodeBundle(Map<String, dynamic> bundle) => jsonEncode(bundle);

double sigmoid(double x) => 1 / (1 + math.exp(-x));
