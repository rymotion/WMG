import 'package:equatable/equatable.dart';
import 'package:flutter/material.dart';

/// What a recording is, by how much of the music was made with AI.
///
/// The share ranges are the targets a future AI-utilisation model trains toward; until then,
/// [aiAssisted] recordings are kept with their audio but left out of Human-vs-AI training.
enum AiCategory {
  human('human', 'Human-made', 0.0, 0.1),
  aiAssisted('ai_assisted', 'AI-assisted', 0.1, 0.5),
  ai('ai', 'AI-generated', 0.5, 1.0);

  const AiCategory(this.wire, this.displayName, this.aiShareMin, this.aiShareMax);

  /// Value written to session.json, segments.csv and .wmgmodel bundles.
  final String wire;
  final String displayName;

  /// Estimated share of AI involvement, 0..1.
  final double aiShareMin;
  final double aiShareMax;

  /// Target for the current Human-vs-AI model: 0 / 1, or null when the category isn't used
  /// for binary training (AI-assisted, for now).
  int? get binaryLabel => switch (this) {
    AiCategory.human => 0,
    AiCategory.ai => 1,
    AiCategory.aiAssisted => null,
  };

  Color get color => switch (this) {
    AiCategory.human => const Color(0xFF2E9E5B),
    AiCategory.aiAssisted => const Color(0xFFE0A100),
    AiCategory.ai => const Color(0xFFD64545),
  };

  IconData get icon => switch (this) {
    AiCategory.human => Icons.person_rounded,
    AiCategory.aiAssisted => Icons.auto_fix_high_rounded,
    AiCategory.ai => Icons.smart_toy_rounded,
  };

  static AiCategory fromWire(String s) => AiCategory.values.firstWhere(
    (c) => c.wire == s,
    orElse: () => throw FormatException('unknown AI category "$s"'),
  );

  /// Older data stored only a binary label (0 human, 1 AI).
  static AiCategory fromBinary(int label) => label == 1 ? AiCategory.ai : AiCategory.human;
}

/// A whole-recording label set by the user.
final class RecordingLabel extends Equatable {
  const RecordingLabel({required this.category, required this.labelledAt});

  factory RecordingLabel.now(AiCategory category) =>
      RecordingLabel(category: category, labelledAt: DateTime.now().toUtc());

  factory RecordingLabel.fromJson(Map<String, dynamic> j) => RecordingLabel(
    category: AiCategory.fromWire(j['category'] as String),
    labelledAt: DateTime.parse(j['labelled_at'] as String),
  );

  final AiCategory category;
  final DateTime labelledAt;

  Map<String, dynamic> toJson() => {
    'category': category.wire,
    'ai_share_min': category.aiShareMin,
    'ai_share_max': category.aiShareMax,
    'labelled_at': labelledAt.toUtc().toIso8601String(),
  };

  @override
  List<Object?> get props => [category, labelledAt];
}
