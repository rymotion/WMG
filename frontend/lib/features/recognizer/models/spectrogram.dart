/// Rolling window of live mel frames for display. Immutable: [append] returns a
/// new instance, and equality is by [revision] so BLoC state comparisons stay
/// cheap even with thousands of values.
class Spectrogram {
  const Spectrogram({
    this.frames = const [],
    this.nMels = 64,
    this.dbFloor = -80,
    this.capacity = 320,
    this.revision = 0,
  });

  /// `[frame][mel bin]`, oldest first, bin 0 lowest frequency, absolute dB.
  final List<List<double>> frames;
  final int nMels;
  final double dbFloor;

  /// Frames kept (~10 s at the server's 32 ms hop).
  final int capacity;
  final int revision;

  bool get isEmpty => frames.isEmpty;

  Spectrogram append(List<List<double>> incoming) {
    final all = [...frames, ...incoming];
    final kept = all.length > capacity ? all.sublist(all.length - capacity) : all;
    return Spectrogram(
      frames: List.unmodifiable(kept),
      nMels: nMels,
      dbFloor: dbFloor,
      capacity: capacity,
      revision: revision + 1,
    );
  }

  @override
  bool operator ==(Object other) =>
      other is Spectrogram && other.revision == revision && other.nMels == nMels && other.dbFloor == dbFloor;

  @override
  int get hashCode => Object.hash(revision, nMels, dbFloor);
}
