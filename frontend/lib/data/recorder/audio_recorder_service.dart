import 'dart:io' show Platform;
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:record/record.dart';

class MicrophonePermissionDenied implements Exception {
  @override
  String toString() => 'Microphone permission was denied';
}

/// Wraps the `record` plugin so the BLoC never touches platform APIs directly.
class AudioRecorderService {
  AudioRecorderService({AudioRecorder? recorder}) : _recorder = recorder ?? AudioRecorder();

  final AudioRecorder _recorder;

  /// Asks for (or confirms) microphone access. Call before opening the socket.
  Future<bool> hasPermission() => _recorder.hasPermission();

  /// Starts capturing mono, signed 16-bit little-endian PCM — exactly what
  /// `/live` expects — and returns the chunk stream. The stream closes once
  /// [stop] or [cancel] is called.
  ///
  /// Voice processing (echo cancellation, noise suppression, auto gain) is off:
  /// it is tuned for speech and distorts music, which hurts identification.
  Future<Stream<Uint8List>> startPcmStream({int sampleRate = 16000}) async {
    if (!await _recorder.hasPermission()) throw MicrophonePermissionDenied();
    try {
      return await _recorder.startStream(_config(sampleRate, AndroidAudioSource.unprocessed));
    } catch (_) {
      // UNPROCESSED isn't supported on every Android device; VOICE_RECOGNITION
      // is the closest widely available source without AGC.
      if (!Platform.isAndroid) rethrow;
      return _recorder.startStream(_config(sampleRate, AndroidAudioSource.voiceRecognition));
    }
  }

  static RecordConfig _config(int sampleRate, AndroidAudioSource source) => RecordConfig(
    encoder: AudioEncoder.pcm16bits,
    sampleRate: sampleRate,
    numChannels: 1,
    autoGain: false,
    echoCancel: false,
    noiseSuppress: false,
    androidConfig: AndroidRecordConfig(audioSource: source),
  );

  /// Normalised level (0..1) of one PCM16 chunk on a -60..0 dBFS scale.
  static double pcmLevel(Uint8List chunk) {
    final samples = chunk.lengthInBytes ~/ 2;
    if (samples == 0) return 0;
    final data = ByteData.sublistView(chunk);
    var sumSquares = 0.0;
    for (var i = 0; i < samples; i++) {
      final v = data.getInt16(i * 2, Endian.little) / 32768.0;
      sumSquares += v * v;
    }
    final rms = math.sqrt(sumSquares / samples);
    if (rms <= 0) return 0;
    final db = 20 * math.log(rms) / math.ln10;
    return ((db + 60) / 60).clamp(0.0, 1.0);
  }

  Future<void> stop() async => _recorder.stop();

  Future<void> cancel() => _recorder.cancel();

  Future<void> dispose() => _recorder.dispose();
}
