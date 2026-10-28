import 'package:equatable/equatable.dart';

class TrackMatch extends Equatable {
  const TrackMatch({
    required this.source,
    this.title,
    this.artists = const [],
    this.album,
    this.label,
    this.releaseDate,
    this.score,
    this.durationMs,
    this.playOffsetMs,
    this.acrid,
  });

  factory TrackMatch.fromJson(Map<String, dynamic> json) => TrackMatch(
    source: json['source'] as String? ?? 'unknown',
    title: json['title'] as String?,
    artists: (json['artists'] as List<dynamic>? ?? const []).cast<String>(),
    album: json['album'] as String?,
    label: json['label'] as String?,
    releaseDate: json['release_date'] as String?,
    score: (json['score'] as num?)?.toDouble(),
    durationMs: (json['duration_ms'] as num?)?.toInt(),
    playOffsetMs: (json['play_offset_ms'] as num?)?.toInt(),
    acrid: json['acrid'] as String?,
  );

  /// Same keys as [TrackMatch.fromJson], for saving sessions on device.
  Map<String, dynamic> toJson() => {
    'source': source,
    if (title != null) 'title': title,
    'artists': artists,
    if (album != null) 'album': album,
    if (label != null) 'label': label,
    if (releaseDate != null) 'release_date': releaseDate,
    if (score != null) 'score': score,
    if (durationMs != null) 'duration_ms': durationMs,
    if (playOffsetMs != null) 'play_offset_ms': playOffsetMs,
    if (acrid != null) 'acrid': acrid,
  };

  final String source;
  final String? title;
  final List<String> artists;
  final String? album;
  final String? label;
  final String? releaseDate;
  final double? score;
  final int? durationMs;
  final int? playOffsetMs;
  final String? acrid;

  String get displayTitle => title ?? 'Unknown title';
  String get displayArtists => artists.isEmpty ? 'Unknown artist' : artists.join(', ');

  @override
  List<Object?> get props => [
    source,
    title,
    artists,
    album,
    label,
    releaseDate,
    score,
    durationMs,
    playOffsetMs,
    acrid,
  ];
}
