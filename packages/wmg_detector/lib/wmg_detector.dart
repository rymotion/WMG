/// On-device AI-generated-music detector.
///
/// PCM goes to native code, where the `wmg-features` Rust library (the same code used in
/// training) computes level-normalised log-mel windows and a Core ML model scores them.
/// Nothing leaves the device.
library;

import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Score for one 10 s window.
@immutable
class WindowScore {
  const WindowScore({required this.startS, required this.logit, required this.bands, this.embedding});

  /// Start of the window within the scored audio, in seconds.
  final double startS;

  /// Raw model output; P(AI) = sigmoid(logit).
  final double logit;

  /// Per-band AI evidence in [0, 1]: bass (20-250 Hz), mid (250 Hz-4 kHz), treble (4-16 kHz).
  final List<double> bands;

  /// Input to the model's final layer (base logit = head . embedding + bias); used to
  /// fine-tune that layer on device. Null where the native side doesn't provide it.
  final Float32List? embedding;

  double get pAi => sigmoid(logit);

  WindowScore withLogit(double l) => WindowScore(startS: startS, logit: l, bands: bands, embedding: embedding);
}

/// Result of scoring one piece of audio.
@immutable
class DetectorResult {
  const DetectorResult({required this.windows, required this.inputRmsDbfs});

  final List<WindowScore> windows;

  /// Level of the audio before normalisation (phone captures are usually -45 to -30 dBFS).
  final double inputRmsDbfs;

  bool get isEmpty => windows.isEmpty;

  /// Mean window logit, as in training (a clip's score is sigmoid of this).
  double get logit => windows.isEmpty ? double.nan : windows.map((w) => w.logit).reduce((a, b) => a + b) / windows.length;

  double get pAi => sigmoid(logit);
}

/// Model and feature versions reported by the native side.
@immutable
class DetectorInfo {
  const DetectorInfo({
    required this.featureSpecHash,
    required this.modelSpecHash,
    required this.modelRun,
    this.windowSeconds = 10,
    this.headWeights,
    this.headBias = 0,
  });

  final String featureSpecHash;
  final String modelSpecHash;
  final String modelRun;

  /// Audio per model window: 10 s or 5 s.
  final double windowSeconds;

  /// The bundled model's final-layer weights: the starting point for on-device fine-tuning.
  final Float64List? headWeights;
  final double headBias;

  /// The model was trained on features from this exact library version.
  bool get compatible => featureSpecHash == modelSpecHash;
}

double sigmoid(double x) => 1 / (1 + math.exp(-x));

class WmgDetector {
  const WmgDetector();

  static const _channel = MethodChannel('wmg_detector');

  /// Native sample rate the features run at; other rates are resampled natively.
  static const featureSampleRate = 32000;

  /// Bundled models: `'10s'` (always) and `'5s'` (when installed).
  static const models = ['10s', '5s'];

  Future<DetectorInfo> info({String model = '10s'}) async {
    final m = (await _channel.invokeMapMethod<String, Object?>('info', {'model': model}))!;
    return DetectorInfo(
      featureSpecHash: m['featureSpecHash']! as String,
      modelSpecHash: m['modelSpecHash']! as String,
      modelRun: m['modelRun']! as String,
      windowSeconds: (m['windowSeconds'] as num?)?.toDouble() ?? 10,
      headWeights: m['headWeights'] as Float64List?,
      headBias: (m['headBias'] as num?)?.toDouble() ?? 0,
    );
  }

  /// Scores mono signed 16-bit little-endian PCM at [sampleRate] (the app's recorder format).
  /// Audio shorter than 3 s, or quieter than the -45 dBFS window gate after normalisation,
  /// yields an empty result.
  /// [model] picks the 10 s or 5 s detector; both share the native feature path.
  Future<DetectorResult> scorePcm16(Uint8List pcm, int sampleRate, {String model = '10s'}) async {
    final m = (await _channel.invokeMapMethod<String, Object?>('score', {
      'pcm': pcm,
      'sampleRate': sampleRate,
      'model': model,
    }))!;
    final windows = [
      for (final w in (m['windows']! as List).cast<Map<Object?, Object?>>())
        WindowScore(
          startS: (w['startS']! as num).toDouble(),
          logit: (w['logit']! as num).toDouble(),
          bands: [for (final b in (w['bands']! as List)) (b as num).toDouble()],
          embedding: w['embedding'] as Float32List?,
        ),
    ];
    return DetectorResult(windows: windows, inputRmsDbfs: (m['inputRmsDbfs']! as num).toDouble());
  }
}
