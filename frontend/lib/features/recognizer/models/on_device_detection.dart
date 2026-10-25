import 'package:equatable/equatable.dart';
import 'package:wmg_detector/wmg_detector.dart';

import '../../../data/personalization/personal_model.dart' show PersonalHead;

enum OnDeviceStatus {
  /// No detector on this platform (Android / web for now).
  unavailable,

  /// Recording; waiting for the first 5 s of audio.
  listening,

  /// At least one window scored.
  ready,
  failed,
}

/// Running on-device AI estimate for the current recording, from two detectors.
///
/// Every 5 s the last 5 s of audio go to the 5 s model and the last 10 s to the 10 s model.
/// Each model's score is sigmoid of its mean window logit (as in training); the headline is
/// the average of the two probabilities and [delta] (5 s minus 10 s) shows how far they
/// disagree. On held-out data, answers where they agreed within 0.1 were right far more often
/// (clean files 98.6%, speaker -> mic 85.9%) than ones where they didn't (60-80%).
final class OnDeviceDetection extends Equatable {
  const OnDeviceDetection({
    this.status = OnDeviceStatus.unavailable,
    this.windows = const [],
    this.windows5s = const [],
    this.inputRmsDbfs,
    this.modelRun,
    this.modelRun5s,
    this.error,
  });

  /// Above this |P(AI) 5 s - P(AI) 10 s| the two models are treated as disagreeing.
  static const agreementThreshold = 0.1;

  final OnDeviceStatus status;

  /// 10 s model windows, start times relative to the recording.
  final List<WindowScore> windows;

  /// 5 s model windows (empty when the 5 s model isn't bundled).
  final List<WindowScore> windows5s;

  /// Level of the latest scored audio before normalisation.
  final double? inputRmsDbfs;
  final String? modelRun;
  final String? modelRun5s;
  final String? error;

  bool get hasScore => windows.isNotEmpty || windows5s.isNotEmpty;

  static double _meanLogit(List<WindowScore> w) =>
      w.isEmpty ? double.nan : w.map((x) => x.logit).reduce((a, b) => a + b) / w.length;

  /// 10 s model's mean window logit.
  double get logit => _meanLogit(windows);

  double? get p10 => windows.isEmpty ? null : sigmoid(_meanLogit(windows));

  double? get p5 => windows5s.isEmpty ? null : sigmoid(_meanLogit(windows5s));

  /// Headline estimate: the average of both models, or whichever has answered so far.
  double get pAi {
    final a = p10, b = p5;
    if (a != null && b != null) return (a + b) / 2;
    return a ?? b ?? double.nan;
  }

  /// P(AI) from the 5 s model minus the 10 s model; null until both have answered.
  double? get delta => (p10 == null || p5 == null) ? null : p5! - p10!;

  /// Both models have answered and are within [agreementThreshold] of each other.
  bool? get agree => delta == null ? null : delta!.abs() <= agreementThreshold;

  /// Only the 5 s model has answered (the 10 s model needs 10 s of audio).
  bool get provisional => windows.isEmpty && windows5s.isNotEmpty;

  /// Mean per-band evidence (bass, mid, treble) across both models' windows.
  List<double> get bands {
    final all = [...windows, ...windows5s];
    return all.isEmpty
        ? const [0, 0, 0]
        : [for (var b = 0; b < 3; b++) all.map((w) => w.bands[b]).reduce((x, y) => x + y) / all.length];
  }

  /// The 10 s windows re-scored with an on-device fine-tuned final layer (personalisation is
  /// fitted on the 10 s model's embeddings; the 5 s model is unchanged).
  OnDeviceDetection personalised(PersonalHead head) =>
      copyWith(windows: [for (final w in windows) w.embedding == null ? w : w.withLogit(head.logit(w.embedding!))]);

  /// Only the windows that start within `[startS, endS)`, e.g. one saved clip.
  /// A 10 s window starting near the end runs a little past it.
  OnDeviceDetection slice(double startS, double endS) {
    bool inside(WindowScore w) => w.startS >= startS && w.startS < endS;
    return copyWith(windows: windows.where(inside).toList(), windows5s: windows5s.where(inside).toList());
  }

  OnDeviceDetection copyWith({
    OnDeviceStatus? status,
    List<WindowScore>? windows,
    List<WindowScore>? windows5s,
    double? inputRmsDbfs,
    String? modelRun,
    String? modelRun5s,
    String? Function()? error,
  }) => OnDeviceDetection(
    status: status ?? this.status,
    windows: windows ?? this.windows,
    windows5s: windows5s ?? this.windows5s,
    inputRmsDbfs: inputRmsDbfs ?? this.inputRmsDbfs,
    modelRun: modelRun ?? this.modelRun,
    modelRun5s: modelRun5s ?? this.modelRun5s,
    error: error != null ? error() : this.error,
  );

  @override
  List<Object?> get props => [
    status,
    windows.length,
    windows5s.length,
    logit,
    _meanLogit(windows5s),
    inputRmsDbfs,
    modelRun,
    modelRun5s,
    error,
  ];
}
