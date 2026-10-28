import 'package:equatable/equatable.dart';

import '../../../data/labels/recording_label.dart';
import '../../../data/models/identification.dart';

/// One saved clip of the live session and, once ACRCloud is done, its result.
class ClipResult extends Equatable {
  const ClipResult({
    required this.index,
    required this.id,
    this.startS,
    this.durationS,
    this.identification,
    this.label,
  });

  factory ClipResult.fromJson(Map<String, dynamic> json) => ClipResult(
    index: (json['index'] as num).toInt(),
    id: json['id'] as String,
    startS: (json['start_s'] as num?)?.toDouble(),
    durationS: (json['duration_s'] as num?)?.toDouble(),
    identification: switch (json['identification']) {
      final Map<String, dynamic> j => Identification.fromJson(j),
      _ => null,
    },
    label: json['label'] == null ? null : RecordingLabel.fromJson(json['label'] as Map<String, dynamic>),
  );

  Map<String, dynamic> toJson() => {
    'index': index,
    'id': id,
    if (startS != null) 'start_s': startS,
    if (durationS != null) 'duration_s': durationS,
    if (identification != null) 'identification': identification!.toJson(),
    if (label != null) 'label': label!.toJson(),
  };

  final int index;
  final String id;
  final double? startS;
  final double? durationS;

  /// Null while identification is still running.
  final Identification? identification;

  /// What the user says this clip is, when it differs from (or there is no) label for the
  /// whole recording. Null: the clip takes the recording's label.
  final RecordingLabel? label;

  ClipResult copyWith({
    double? startS,
    double? durationS,
    Identification? identification,
    RecordingLabel? Function()? label,
  }) => ClipResult(
    index: index,
    id: id,
    startS: startS ?? this.startS,
    durationS: durationS ?? this.durationS,
    identification: identification ?? this.identification,
    label: label != null ? label() : this.label,
  );

  @override
  List<Object?> get props => [index, id, startS, durationS, identification, label];
}
