/// Subscription tiers and what each one unlocks. The server is the authority on a user's tier
/// and remaining quota; these are the rules the app uses to shape its UI.
enum Tier {
  free('free', 'Free'),
  plus('plus', 'Plus'),
  pro('pro', 'Pro Creator');

  const Tier(this.wire, this.displayName);

  final String wire;
  final String displayName;

  static Tier parse(String? wire) =>
      values.firstWhere((t) => t.wire == wire, orElse: () => free);

  Entitlements get entitlements => switch (this) {
    free => Entitlements.free,
    plus => Entitlements.plus,
    pro => Entitlements.pro,
  };
}

enum BillingInterval {
  month('month'),
  year('year');

  const BillingInterval(this.wire);

  final String wire;

  static BillingInterval? parse(String? wire) =>
      values.where((i) => i.wire == wire).firstOrNull;
}

/// Feature gates for one tier. A null limit means unlimited.
class Entitlements {
  const Entitlements({
    required this.monthlyMicIds,
    required this.listenWindow,
    required this.monthlyUploadIds,
    required this.historyRetention,
    required this.searchableHistory,
    required this.playlistExport,
    required this.offlineQueue,
    required this.multiSongScan,
    required this.proMetadata,
    required this.bulkExport,
  });

  static const free = Entitlements(
    monthlyMicIds: 10,
    listenWindow: Duration(seconds: 10),
    monthlyUploadIds: 0,
    historyRetention: Duration(days: 7),
    searchableHistory: false,
    playlistExport: false,
    offlineQueue: false,
    multiSongScan: false,
    proMetadata: false,
    bulkExport: false,
  );

  static const plus = Entitlements(
    monthlyMicIds: null,
    listenWindow: Duration(seconds: 30),
    monthlyUploadIds: 0,
    historyRetention: null,
    searchableHistory: true,
    playlistExport: true,
    offlineQueue: true,
    multiSongScan: false,
    proMetadata: false,
    bulkExport: false,
  );

  static const pro = Entitlements(
    monthlyMicIds: null,
    listenWindow: Duration(seconds: 30),
    monthlyUploadIds: 100,
    historyRetention: null,
    searchableHistory: true,
    playlistExport: true,
    offlineQueue: true,
    multiSongScan: true,
    proMetadata: true,
    bulkExport: true,
  );

  /// Mic identifications per billing month; null: unlimited.
  final int? monthlyMicIds;

  /// How long one tap listens before it stops on its own (it also stops at the first match).
  final Duration listenWindow;

  /// Video/audio file identifications per billing month.
  final int monthlyUploadIds;

  /// How far back history is shown; null: forever.
  final Duration? historyRetention;
  final bool searchableHistory;

  /// Auto-create playlists on Spotify and Apple Music.
  final bool playlistExport;

  /// Record now, identify later.
  final bool offlineQueue;

  /// Timestamped tracklist from a mix/set/video up to an hour long.
  final bool multiSongScan;

  /// BPM, key, energy, release date, ISRC, label.
  final bool proMetadata;

  /// CSV, Rekordbox and Serato export.
  final bool bulkExport;
}
