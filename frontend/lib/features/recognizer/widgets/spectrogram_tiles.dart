import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/painting.dart' show HSLColor;

import '../models/spectrogram.dart';

/// RGBA for each [Spectrogram.level] byte: 0 (a gap) is transparent, 1..255 run
/// floor → 0 dB through the same HSL ramp the spectrogram has always used.
final Uint32List _lut = () {
  final lut = Uint32List(256);
  for (var level = 1; level < 256; level++) {
    final v = (level - 1) / 254;
    final c = HSLColor.fromAHSL(
      1,
      (280 - v * 240) % 360,
      0.9,
      v * 0.6,
    ).toColor();
    // Little-endian uint32 → bytes R, G, B, A in memory (rgba8888).
    lut[level] =
        (c.a * 255).round() << 24 |
        (c.b * 255).round() << 16 |
        (c.g * 255).round() << 8 |
        (c.r * 255).round();
  }
  return lut;
}();

/// RGBA pixels for frames `[first, first + width)` of [spec]: one pixel column per frame,
/// one row per mel bin, low bins at the bottom. Pure, so it could run off the UI isolate.
Uint8List spectrogramTilePixels(Spectrogram spec, int first, int width) {
  final h = spec.nMels;
  final px = Uint32List(width * h);
  for (var x = 0; x < width; x++) {
    final frame = spec.frameAt(first + x);
    if (frame == null) continue;
    for (var b = 0; b < h && b < frame.length; b++) {
      px[(h - 1 - b) * width + x] =
          _lut[Spectrogram.level(frame[b], spec.dbFloor)];
    }
  }
  return px.buffer.asUint8List();
}

/// The spectrogram as fixed-width image tiles, so painting is a few image blits instead
/// of a rect per cell. Tiles are made on demand for what's on screen and remade only
/// when new frames land in them (live, that's just the last one). Least recently used
/// tiles are dropped past [maxTiles].
class SpectrogramTiles {
  SpectrogramTiles({required this.onChanged, this.maxTiles = 48});

  /// Frames per tile (~8 s at the 32 ms hop).
  static const tileFrames = 256;

  /// Called when a tile image is ready, to repaint.
  final void Function() onChanged;
  final int maxTiles;

  Spectrogram _spec = const Spectrogram();
  final _images = <int, ui.Image>{};

  /// Spectrogram length each tile's image was made from.
  final _madeAt = <int, int>{};
  final _pending = <int>{};
  final _lastUsed = <int, int>{};
  int _clock = 0;
  bool _disposed = false;

  /// Takes a new snapshot; drops every tile if it isn't a continuation of the last one.
  void update(Spectrogram spec) {
    final restart =
        spec.length < _spec.length ||
        spec.nMels != _spec.nMels ||
        spec.dbFloor != _spec.dbFloor ||
        (spec.length > 0 &&
            _spec.length > 0 &&
            !identical(spec.frameAt(0), _spec.frameAt(0)));
    _spec = spec;
    if (restart) _clear();
  }

  /// Tile `k`'s image if made yet (possibly a little behind the live edge).
  ui.Image? operator [](int k) => _images[k];

  /// Makes sure tiles `[first, last]` are made and current.
  void request(int first, int last) {
    final tiles = (_spec.length + tileFrames - 1) ~/ tileFrames;
    for (var k = first < 0 ? 0 : first; k <= last && k < tiles; k++) {
      _lastUsed[k] = ++_clock;
      if (_pending.contains(k)) continue;
      final filled = (_spec.length - k * tileFrames).clamp(0, tileFrames);
      final madeFrom = _madeAt[k];
      final upToDate =
          madeFrom != null &&
          (madeFrom - k * tileFrames).clamp(0, tileFrames) >= filled;
      if (upToDate) continue;
      _make(k);
    }
  }

  void _make(int k) {
    _pending.add(k);
    final spec = _spec;
    final pixels = spectrogramTilePixels(spec, k * tileFrames, tileFrames);
    ui.decodeImageFromPixels(
      pixels,
      tileFrames,
      spec.nMels,
      ui.PixelFormat.rgba8888,
      (image) {
        _pending.remove(k);
        if (_disposed ||
            !identical(spec.frameAt(0), _spec.frameAt(0)) ||
            spec.nMels != _spec.nMels) {
          image.dispose(); // made for a session that's gone
          return;
        }
        _images.remove(k)?.dispose();
        _images[k] = image;
        _madeAt[k] = spec.length;
        _evict();
        onChanged();
      },
    );
  }

  void _evict() {
    while (_images.length > maxTiles) {
      final oldest = _images.keys.reduce(
        (a, b) => (_lastUsed[a] ?? 0) <= (_lastUsed[b] ?? 0) ? a : b,
      );
      _images.remove(oldest)?.dispose();
      _madeAt.remove(oldest);
    }
  }

  void _clear() {
    for (final image in _images.values) {
      image.dispose();
    }
    _images.clear();
    _madeAt.clear();
    _lastUsed.clear();
  }

  void dispose() {
    _disposed = true;
    _clear();
  }
}
