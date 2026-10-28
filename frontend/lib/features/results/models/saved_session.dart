import 'package:equatable/equatable.dart';
import 'package:wmg_detector/wmg_detector.dart';

import '../../../data/labels/recording_label.dart';
import '../../../data/models/identification.dart';
import '../../recognizer/models/clip_result.dart';
import '../../recognizer/models/on_device_detection.dart';

/// One finished live session as kept on device: its clips and identifications and the
/// on-device window scores. The spectrogram and mic audio live in files next to it
/// (see `SessionHistoryStore`).
final class SavedSession extends Equatable {
  const SavedSession({
    required this.id,
    required this.startedAt,
    required this.durationS,
    this.serverSessionId,
    this.offline = false,
    this.clips = const [],
    this.onDevice = const OnDeviceDetection(),
    this.nMels = 64,
    this.dbFloor = -80,
    this.hopS = 0.032,
    this.sampleRate = 32000,
    this.hasSpectrogram = false,
    this.hasAudio = false,
    this.label,
  });

  factory SavedSession.fromJson(Map<String, dynamic> json) {
    final spec = json['spectrogram'] as Map<String, dynamic>? ?? const {};
    final d = json['on_device'] as Map<String, dynamic>? ?? const {};
    final windows = _windowsFromJson(d['windows']);
    final windows5s = _windowsFromJson(d['windows_5s']);
    return SavedSession(
      id: json['id'] as String,
      startedAt: DateTime.parse(json['started_at'] as String),
      durationS: (json['duration_s'] as num).toDouble(),
      serverSessionId: json['server_session_id'] as String?,
      offline: json['offline'] as bool? ?? false,
      clips: [
        for (final c in json['clips'] as List<dynamic>? ?? const [])
          ClipResult.fromJson(c as Map<String, dynamic>),
      ],
      onDevice: OnDeviceDetection(
        status: windows.isEmpty && windows5s.isEmpty
            ? OnDeviceStatus.unavailable
            : OnDeviceStatus.ready,
        windows: windows,
        windows5s: windows5s,
        modelRun: d['model_run'] as String?,
        modelRun5s: d['model_run_5s'] as String?,
      ),
      nMels: (spec['n_mels'] as num?)?.toInt() ?? 64,
      dbFloor: (spec['db_floor'] as num?)?.toDouble() ?? -80,
      hopS: (spec['hop_s'] as num?)?.toDouble() ?? 0.032,
      sampleRate: (json['sample_rate'] as num?)?.toInt() ?? 32000,
      hasSpectrogram: spec['saved'] as bool? ?? false,
      hasAudio: json['has_audio'] as bool? ?? false,
      label: json['label'] == null ? null : RecordingLabel.fromJson(json['label'] as Map<String, dynamic>),
    );
  }

  /// Local id, also the folder name: the start time, e.g. `20261026-012233-123`.
  final String id;
  final DateTime startedAt;
  final double durationS;

  /// The server's `session_id`; null for offline sessions.
  final String? serverSessionId;
  final bool offline;

  /// Newest first, as in the live view.
  final List<ClipResult> clips;

  /// Base-model window scores (embeddings aren't kept, so no personalised re-scoring).
  final OnDeviceDetection onDevice;
  final int nMels;
  final double dbFloor;
  final double hopS;
  final int sampleRate;
  final bool hasSpectrogram;
  final bool hasAudio;

  /// What the user says the recording is (Human / AI-assisted / AI-generated); null if unlabelled.
  final RecordingLabel? label;

  SavedSession withLabel(RecordingLabel? label) => _copy(label: label, clips: clips);

  /// Sets (or, with null, clears) one clip's own label.
  SavedSession withClipLabel(int clipIndex, RecordingLabel? clipLabel) => withClipLabels({clipIndex}, clipLabel);

  /// Sets (or, with null, clears) the own label of every clip in [clipIndexes].
  SavedSession withClipLabels(Set<int> clipIndexes, RecordingLabel? clipLabel) => _copy(
    label: label,
    clips: [for (final c in clips) clipIndexes.contains(c.index) ? c.copyWith(label: () => clipLabel) : c],
  );

  /// The clip's own label, else the recording's.
  RecordingLabel? labelFor(ClipResult clip) => clip.label ?? label;

  SavedSession _copy({required RecordingLabel? label, required List<ClipResult> clips}) => SavedSession(
    id: id,
    startedAt: startedAt,
    durationS: durationS,
    serverSessionId: serverSessionId,
    offline: offline,
    clips: clips,
    onDevice: onDevice,
    nMels: nMels,
    dbFloor: dbFloor,
    hopS: hopS,
    sampleRate: sampleRate,
    hasSpectrogram: hasSpectrogram,
    hasAudio: hasAudio,
    label: label,
  );

  /// Titles of the clips' best matches, oldest clip first, without repeats.
  List<String> get matchedTitles => {
    for (final c in clips.reversed)
      if (c.identification case Identification(
        status: IdentificationStatus.matched ||
            IdentificationStatus.possibleMatch,
        matches: [final best, ...],
      ))
        best.displayTitle,
  }.toList();

  Map<String, dynamic> toJson() => {
    'id': id,
    'started_at': startedAt.toIso8601String(),
    'duration_s': durationS,
    if (serverSessionId != null) 'server_session_id': serverSessionId,
    'offline': offline,
    'clips': [for (final c in clips) c.toJson()],
    'on_device': {
      'windows': [for (final w in onDevice.windows) _windowToJson(w)],
      'windows_5s': [for (final w in onDevice.windows5s) _windowToJson(w)],
      if (onDevice.modelRun != null) 'model_run': onDevice.modelRun,
      if (onDevice.modelRun5s != null) 'model_run_5s': onDevice.modelRun5s,
    },
    'spectrogram': {
      'saved': hasSpectrogram,
      'n_mels': nMels,
      'db_floor': dbFloor,
      'hop_s': hopS,
    },
    'sample_rate': sampleRate,
    'has_audio': hasAudio,
    if (label != null) 'label': label!.toJson(),
  };

  static Map<String, dynamic> _windowToJson(WindowScore w) => {
    'start_s': w.startS,
    'logit': w.logit,
    'bands': w.bands,
  };

  static List<WindowScore> _windowsFromJson(Object? raw) => [
    for (final w
        in (raw as List<dynamic>? ?? const []).cast<Map<String, dynamic>>())
      WindowScore(
        startS: (w['start_s'] as num).toDouble(),
        logit: (w['logit'] as num).toDouble(),
        bands: [
          for (final b in w['bands'] as List<dynamic>) (b as num).toDouble(),
        ],
      ),
  ];

  @override
  List<Object?> get props => [
    id,
    startedAt,
    durationS,
    serverSessionId,
    offline,
    clips,
    onDevice,
    hasAudio,
    label,
  ];
}
