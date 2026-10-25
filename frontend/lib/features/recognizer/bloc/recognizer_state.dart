part of 'recognizer_bloc.dart';

enum RecognizerStatus {
  idle,
  connecting,

  /// Mic open, PCM flowing to the server.
  streaming,

  /// `stop` sent; waiting for the last clip, pending identifications and the close.
  finishing,
  completed,
  failure,
}

final class RecognizerState extends Equatable {
  const RecognizerState({
    this.status = RecognizerStatus.idle,
    this.elapsed = Duration.zero,
    this.level = 0,
    this.bytesStreamed = 0,
    this.session,
    this.clips = const [],
    this.spectrogram = const Spectrogram(),
    this.onDevice = const OnDeviceDetection(),
    this.offline = false,
    this.clipMetaData = const [],
    this.error,
  });

  final RecognizerStatus status;

  /// Stream time so far.
  final Duration elapsed;

  /// Normalised mic input level, 0..1.
  final double level;

  /// PCM bytes sent over the socket for this session.
  final int bytesStreamed;

  /// From the server's `session` message.
  final LiveSessionStarted? session;

  /// Saved clips, newest first, with their identifications as they arrive.
  final List<ClipResult> clips;

  /// Saved clips metadata, newest first.
  final List<ClipMetaData> clipMetaData;

  /// Live mel spectrogram from `spectrum` messages.
  final Spectrogram spectrogram;

  /// Local AI estimate (Rust features + Core ML), independent of the server.
  final OnDeviceDetection onDevice;

  /// This session runs on the on-device model only: no socket, so no
  /// spectrogram, clips or identifications.
  final bool offline;
  final String? error;

  bool get isStreaming => status == RecognizerStatus.streaming;
  bool get isBusy =>
      status == RecognizerStatus.connecting ||
      status == RecognizerStatus.finishing;

  /// Seconds per server clip (30 unless the server says otherwise).
  double get clipSeconds => session?.clipSeconds ?? 30;

  /// Progress through the clip currently being recorded, 0..1.
  double get clipProgress {
    final s = elapsed.inMilliseconds / 1000;
    return (s % clipSeconds) / clipSeconds;
  }

  RecognizerState copyWith({
    RecognizerStatus? status,
    Duration? elapsed,
    double? level,
    int? bytesStreamed,
    LiveSessionStarted? session,
    List<ClipResult>? clips,
    List<ClipMetaData>? clipMetaData,
    Spectrogram? spectrogram,
    OnDeviceDetection? onDevice,
    bool? offline,
    String? Function()? error,
  }) => RecognizerState(
    status: status ?? this.status,
    elapsed: elapsed ?? this.elapsed,
    level: level ?? this.level,
    bytesStreamed: bytesStreamed ?? this.bytesStreamed,
    session: session ?? this.session,
    clips: clips ?? this.clips,
    spectrogram: spectrogram ?? this.spectrogram,
    onDevice: onDevice ?? this.onDevice,
    offline: offline ?? this.offline,
    clipMetaData: clipMetaData ?? this.clipMetaData,
    error: error != null ? error() : this.error,
  );

  @override
  List<Object?> get props => [
    status,
    elapsed,
    level,
    bytesStreamed,
    session,
    clips,
    clipMetaData,
    spectrogram,
    onDevice,
    offline,
    error,
  ];
}
