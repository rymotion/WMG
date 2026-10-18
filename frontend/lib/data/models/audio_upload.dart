import 'package:equatable/equatable.dart';

import 'identification.dart';

/// Metadata for an uploaded clip, as returned by `POST /audio`, `GET /audio/{id}`
/// and (as a list) `GET /audio`.
class AudioUpload extends Equatable {
  const AudioUpload({
    required this.id,
    required this.filename,
    required this.extension,
    required this.sizeBytes,
    required this.durationS,
    required this.sampleRate,
    required this.channels,
    required this.codec,
    required this.createdAt,
    required this.identification,
  });

  factory AudioUpload.fromJson(Map<String, dynamic> json) => AudioUpload(
    id: json['id'] as String,
    filename: json['filename'] as String,
    extension: json['extension'] as String,
    sizeBytes: (json['size_bytes'] as num).toInt(),
    durationS: (json['duration_s'] as num).toDouble(),
    sampleRate: (json['sample_rate'] as num).toInt(),
    channels: (json['channels'] as num).toInt(),
    codec: json['codec'] as String,
    createdAt: DateTime.parse(json['created_at'] as String),
    identification: Identification.fromJson(json['identification'] as Map<String, dynamic>),
  );

  static List<AudioUpload> listFromJson(List<dynamic> json) =>
      json.map((e) => AudioUpload.fromJson(e as Map<String, dynamic>)).toList();

  final String id;
  final String filename;
  final String extension;
  final int sizeBytes;
  final double durationS;
  final int sampleRate;
  final int channels;
  final String codec;
  final DateTime createdAt;
  final Identification identification;

  AudioUpload copyWith({Identification? identification}) => AudioUpload(
    id: id,
    filename: filename,
    extension: extension,
    sizeBytes: sizeBytes,
    durationS: durationS,
    sampleRate: sampleRate,
    channels: channels,
    codec: codec,
    createdAt: createdAt,
    identification: identification ?? this.identification,
  );

  @override
  List<Object?> get props => [
    id,
    filename,
    extension,
    sizeBytes,
    durationS,
    sampleRate,
    channels,
    codec,
    createdAt,
    identification,
  ];
}
