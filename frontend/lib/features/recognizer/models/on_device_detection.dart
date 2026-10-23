import 'package:equatable/equatable.dart';
import 'package:wmg_detector/wmg_detector.dart';

import '../../../data/personalization/personal_model.dart' show PersonalHead;

enum OnDeviceStatus {
  /// No detector on this platform (Android / web for now).
  unavailable,

  /// Recording; waiting for the first 10 s of audio.
  listening,

  /// At least one window scored.
  ready,
  failed,
}

/// Running on-device AI estimate for the current recording.
///
/// Every 10 s of microphone audio is scored locally (Rust features + Core ML); the session
/// score is sigmoid of the mean window logit, the same aggregation used in training.
final class OnDeviceDetection extends Equatable {
  const OnDeviceDetection({
    this.status = OnDeviceStatus.unavailable,
    this.windows = const [],
    this.inputRmsDbfs,
    this.modelRun,
    this.error,
  });

  final OnDeviceStatus status;

  /// Scored windows, start times relative to the recording.
  final List<WindowScore> windows;

  /// Level of the latest scored audio before normalisation.
  final double? inputRmsDbfs;
  final String? modelRun;
  final String? error;

  bool get hasScore => windows.isNotEmpty;

  double get logit => windows.isEmpty ? double.nan : windows.map((w) => w.logit).reduce((a, b) => a + b) / windows.length;

  double get pAi => sigmoid(logit);

  /// Mean per-band evidence (bass, mid, treble).
  List<double> get bands => windows.isEmpty
      ? const [0, 0, 0]
      : [for (var b = 0; b < 3; b++) windows.map((w) => w.bands[b]).reduce((x, y) => x + y) / windows.length];

  /// The same windows re-scored with an on-device fine-tuned final layer.
  OnDeviceDetection personalised(PersonalHead head) => copyWith(
    windows: [for (final w in windows) w.embedding == null ? w : w.withLogit(head.logit(w.embedding!))],
  );

  OnDeviceDetection copyWith({
    OnDeviceStatus? status,
    List<WindowScore>? windows,
    double? inputRmsDbfs,
    String? modelRun,
    String? Function()? error,
  }) => OnDeviceDetection(
    status: status ?? this.status,
    windows: windows ?? this.windows,
    inputRmsDbfs: inputRmsDbfs ?? this.inputRmsDbfs,
    modelRun: modelRun ?? this.modelRun,
    error: error != null ? error() : this.error,
  );

  @override
  List<Object?> get props => [status, windows.length, logit, inputRmsDbfs, modelRun, error];
}
