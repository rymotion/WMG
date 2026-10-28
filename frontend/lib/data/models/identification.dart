import 'package:equatable/equatable.dart';

import 'track_match.dart';

enum IdentificationStatus {
  pending,
  matched,
  possibleMatch,
  noMatch,
  error,
  disabled,
  skipped,
  unknown;

  static IdentificationStatus parse(String? raw) => switch (raw) {
    'pending' => pending,
    'matched' => matched,
    'possible_match' => possibleMatch,
    'no_match' => noMatch,
    'error' => error,
    'disabled' => disabled,
    'skipped' => skipped,
    _ => unknown,
  };

  /// The server's spelling, the inverse of [parse].
  String get wire => switch (this) {
    possibleMatch => 'possible_match',
    noMatch => 'no_match',
    _ => name,
  };

  /// Anything but `pending` is final; stop polling.
  bool get isTerminal => this != pending;
}

class Identification extends Equatable {
  const Identification({
    required this.status,
    this.updatedAt,
    this.code,
    this.message,
    this.windowOffsetS,
    this.matches = const [],
  });

  factory Identification.fromJson(Map<String, dynamic> json) => Identification(
    status: IdentificationStatus.parse(json['status'] as String?),
    code: (json['code'] as num?)?.toInt(),
    message: json['message'] as String?,
    windowOffsetS: (json['window_offset_s'] as num?)?.toDouble(),
    matches: (json['matches'] as List<dynamic>? ?? const [])
        .map((m) => TrackMatch.fromJson(m as Map<String, dynamic>))
        .toList(),
    // Absent on live `identification` messages.
    updatedAt: switch (json['updated_at']) {
      final String raw => DateTime.parse(raw),
      _ => null,
    },
  );

  /// Same keys as [Identification.fromJson], for saving sessions on device.
  Map<String, dynamic> toJson() => {
    'status': status.wire,
    if (code != null) 'code': code,
    if (message != null) 'message': message,
    if (windowOffsetS != null) 'window_offset_s': windowOffsetS,
    'matches': [for (final m in matches) m.toJson()],
    if (updatedAt != null) 'updated_at': updatedAt!.toIso8601String(),
  };

  final IdentificationStatus status;
  final int? code;
  final String? message;
  final double? windowOffsetS;
  final List<TrackMatch> matches;
  final DateTime? updatedAt;

  @override
  List<Object?> get props => [status, code, message, windowOffsetS, matches, updatedAt];
}
