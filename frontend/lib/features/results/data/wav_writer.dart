import 'dart:io';
import 'dart:typed_data';

/// Streams 16-bit little-endian PCM into a `.wav` file. The header is written with zero
/// sizes up front and patched on [close], so audio can be appended while recording.
class WavWriter {
  WavWriter._(
    this.file,
    this._sink, {
    required this.sampleRate,
    required this.channels,
  });

  static Future<WavWriter> open(
    File file, {
    required int sampleRate,
    int channels = 1,
  }) async {
    await file.parent.create(recursive: true);
    final sink = file.openWrite();
    sink.add(header(sampleRate: sampleRate, channels: channels, dataBytes: 0));
    return WavWriter._(file, sink, sampleRate: sampleRate, channels: channels);
  }

  final File file;
  final IOSink _sink;
  final int sampleRate;
  final int channels;
  int _dataBytes = 0;
  Future<void>? _closing;

  bool get isClosed => _closing != null;

  double get durationS => _dataBytes / (sampleRate * 2 * channels);

  void add(Uint8List pcm) {
    if (isClosed) return;
    _sink.add(pcm);
    _dataBytes += pcm.length;
  }

  /// Flushes and fixes the RIFF/data sizes. Safe to call more than once.
  Future<void> close() => _closing ??= _close();

  Future<void> _close() async {
    await _sink.flush();
    await _sink.close();
    final raf = await file.open(mode: FileMode.append);
    try {
      await raf.setPosition(0);
      await raf.writeFrom(
        header(
          sampleRate: sampleRate,
          channels: channels,
          dataBytes: _dataBytes,
        ),
      );
    } finally {
      await raf.close();
    }
  }

  /// The 44-byte canonical PCM header.
  static Uint8List header({
    required int sampleRate,
    required int channels,
    required int dataBytes,
  }) {
    final b = ByteData(44);
    void ascii(int at, String s) {
      for (var i = 0; i < s.length; i++) {
        b.setUint8(at + i, s.codeUnitAt(i));
      }
    }

    ascii(0, 'RIFF');
    b.setUint32(4, 36 + dataBytes, Endian.little);
    ascii(8, 'WAVE');
    ascii(12, 'fmt ');
    b.setUint32(16, 16, Endian.little); // fmt chunk size
    b.setUint16(20, 1, Endian.little); // PCM
    b.setUint16(22, channels, Endian.little);
    b.setUint32(24, sampleRate, Endian.little);
    b.setUint32(28, sampleRate * channels * 2, Endian.little); // byte rate
    b.setUint16(32, channels * 2, Endian.little); // block align
    b.setUint16(34, 16, Endian.little); // bits per sample
    ascii(36, 'data');
    b.setUint32(40, dataBytes, Endian.little);
    return b.buffer.asUint8List();
  }
}
