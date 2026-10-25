import 'dart:typed_data';

/// Every live mel frame of the session, indexed by stream time so the view can
/// scrub back to any clip. Immutable from the outside: [append] returns a new
/// instance, and equality is by [revision] so BLoC state comparisons stay
/// cheap even with thousands of frames.
class Spectrogram {
  const Spectrogram({this.nMels = 64, this.dbFloor = -80, this.hopS = 0.032, this.visibleFrames = 320})
    : _frames = const [],
      length = 0,
      revision = 0;

  const Spectrogram._(
    this._frames, {
    required this.length,
    required this.nMels,
    required this.dbFloor,
    required this.hopS,
    required this.visibleFrames,
    required this.revision,
  });

  /// Append-only buffer shared between snapshots; only `[0, length)` belongs
  /// to this one. Null slots are frames the server skipped.
  final List<Float32List?> _frames;

  /// Frame slots in this snapshot; slot `i` starts at `i * hopS` seconds.
  final int length;
  final int nMels;
  final double dbFloor;

  /// Seconds per frame.
  final double hopS;

  /// Frames shown across the view's width (~10 s at the server's 32 ms hop).
  final int visibleFrames;
  final int revision;

  bool get isEmpty => length == 0;

  double get durationS => length * hopS;

  /// `[mel bin]` in absolute dB, bin 0 lowest; null for a gap.
  Float32List? frameAt(int i) => i >= 0 && i < length ? _frames[i] : null;

  /// Adds a `spectrum` message whose first frame starts at [t] seconds.
  Spectrogram append(double t, List<List<double>> incoming) {
    final start = (t / hopS).round();
    // Reuse the shared buffer when this is its latest snapshot; otherwise copy.
    final buf = length > 0 && _frames.length == length ? _frames : _frames.take(length).toList();
    // Skipped messages leave a gap; overlapping frames are already drawn.
    while (buf.length < start) {
      buf.add(null);
    }
    for (var i = buf.length - start; i < incoming.length; i++) {
      buf.add(Float32List.fromList(incoming[i]));
    }
    return Spectrogram._(
      buf,
      length: buf.length,
      nMels: nMels,
      dbFloor: dbFloor,
      hopS: hopS,
      visibleFrames: visibleFrames,
      revision: revision + 1,
    );
  }

  @override
  bool operator ==(Object other) =>
      other is Spectrogram && other.revision == revision && other.nMels == nMels && other.dbFloor == dbFloor;

  @override
  int get hashCode => Object.hash(revision, nMels, dbFloor);
}
