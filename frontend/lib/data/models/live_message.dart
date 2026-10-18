import 'dart:convert';

import 'package:equatable/equatable.dart';

import 'identification.dart';

/// A JSON text message from the `/live` socket. Every message has a `type`.
sealed class LiveServerMessage extends Equatable {
  const LiveServerMessage();

  /// Returns null for types this client doesn't know, so new server messages
  /// don't break older clients.
  static LiveServerMessage? parse(String raw) {
    final json = jsonDecode(raw);
    if (json is! Map<String, dynamic>) throw FormatException('Live message is not an object', raw);
    return switch (json['type']) {
      'session' => LiveSessionStarted.fromJson(json),
      'spectrum' => LiveSpectrum.fromJson(json),
      'clip' => LiveClip.fromJson(json),
      'identification' => LiveClipIdentified.fromJson(json),
      'error' => LiveError(json['message']?.toString() ?? 'Live stream error'),
      _ => null,
    };
  }

  @override
  bool? get stringify => true;
}

/// `session`: sent once, right after connecting.
final class LiveSessionStarted extends LiveServerMessage {
  const LiveSessionStarted({
    required this.sessionId,
    required this.clipSeconds,
    required this.sampleRate,
    required this.nMels,
    required this.hopS,
    required this.dbFloor,
    this.freqs = const [],
  });

  factory LiveSessionStarted.fromJson(Map<String, dynamic> json) {
    final spectrum = json['spectrum'] as Map<String, dynamic>? ?? const {};
    return LiveSessionStarted(
      sessionId: json['session_id'] as String,
      clipSeconds: (json['clip_seconds'] as num?)?.toDouble() ?? 30,
      sampleRate: (json['sample_rate'] as num).toInt(),
      nMels: (spectrum['n_mels'] as num?)?.toInt() ?? 64,
      hopS: (spectrum['hop_s'] as num?)?.toDouble() ?? 0.032,
      dbFloor: (spectrum['db_floor'] as num?)?.toDouble() ?? -80,
      freqs: (spectrum['freqs'] as List<dynamic>? ?? const []).map((f) => (f as num).toDouble()).toList(),
    );
  }

  final String sessionId;
  final double clipSeconds;
  final int sampleRate;
  final int nMels;
  final double hopS;
  final double dbFloor;
  final List<double> freqs;

  @override
  List<Object?> get props => [sessionId, clipSeconds, sampleRate, nMels, hopS, dbFloor, freqs];
}

/// `spectrum`: ~1 s of mel frames, `frames[frame][bin]` in absolute dB, bin 0 lowest.
final class LiveSpectrum extends LiveServerMessage {
  const LiveSpectrum({required this.t, required this.frames});

  factory LiveSpectrum.fromJson(Map<String, dynamic> json) => LiveSpectrum(
    t: (json['t'] as num).toDouble(),
    frames: (json['frames'] as List<dynamic>)
        .map((f) => (f as List<dynamic>).map((v) => (v as num).toDouble()).toList(growable: false))
        .toList(growable: false),
  );

  final double t;
  final List<List<double>> frames;

  @override
  List<Object?> get props => [t, frames.length];
}

/// `clip`: a 30 s (or final partial) clip was saved. `id` works with `/audio/{id}/...`.
final class LiveClip extends LiveServerMessage {
  const LiveClip({required this.index, required this.id, required this.startS, required this.durationS});

  factory LiveClip.fromJson(Map<String, dynamic> json) => LiveClip(
    index: (json['index'] as num).toInt(),
    id: json['id'] as String,
    startS: (json['start_s'] as num).toDouble(),
    durationS: (json['duration_s'] as num).toDouble(),
  );

  final int index;
  final String id;
  final double startS;
  final double durationS;

  @override
  List<Object?> get props => [index, id, startS, durationS];
}

/// `identification`: ACRCloud finished with clip [index].
final class LiveClipIdentified extends LiveServerMessage {
  const LiveClipIdentified({required this.index, required this.id, required this.identification});

  factory LiveClipIdentified.fromJson(Map<String, dynamic> json) => LiveClipIdentified(
    index: (json['index'] as num).toInt(),
    id: json['id'] as String,
    identification: Identification.fromJson(json),
  );

  final int index;
  final String id;
  final Identification identification;

  @override
  List<Object?> get props => [index, id, identification];
}

/// `error`: fatal; the server closes with code 1008 right after.
final class LiveError extends LiveServerMessage {
  const LiveError(this.message);

  final String message;

  @override
  List<Object?> get props => [message];
}
