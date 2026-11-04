import 'package:equatable/equatable.dart';

import 'plan.dart';

/// This billing month's usage, as the server counts it.
class Usage extends Equatable {
  const Usage({
    required this.periodEnd,
    this.micIdsUsed = 0,
    this.micIdsLimit,
    this.uploadIdsUsed = 0,
    this.uploadIdsLimit = 0,
    this.micCredits = 0,
    this.videoCredits = 0,
  });

  factory Usage.fromJson(Map<String, dynamic> json) => Usage(
    periodEnd: DateTime.parse(json['period_end'] as String),
    micIdsUsed: json['mic_ids_used'] as int? ?? 0,
    micIdsLimit: json['mic_ids_limit'] as int?,
    uploadIdsUsed: json['upload_ids_used'] as int? ?? 0,
    uploadIdsLimit: json['upload_ids_limit'] as int? ?? 0,
    micCredits: json['mic_credits'] as int? ?? 0,
    videoCredits: json['video_credits'] as int? ?? 0,
  );

  /// When the monthly allowance resets.
  final DateTime periodEnd;
  final int micIdsUsed;

  /// Null: unlimited.
  final int? micIdsLimit;
  final int uploadIdsUsed;
  final int uploadIdsLimit;

  /// Bought credit packs; spent once the monthly allowance is used up. They don't expire.
  final int micCredits;
  final int videoCredits;

  bool get unlimitedMic => micIdsLimit == null;

  /// Mic IDs left this month, credits included; null: unlimited.
  int? get micIdsLeft => micIdsLimit == null
      ? null
      : (micIdsLimit! - micIdsUsed).clamp(0, micIdsLimit!) + micCredits;

  int get uploadIdsLeft =>
      (uploadIdsLimit - uploadIdsUsed).clamp(0, uploadIdsLimit) + videoCredits;

  bool get canIdentify => micIdsLeft == null || micIdsLeft! > 0;

  Map<String, dynamic> toJson() => {
    'period_end': periodEnd.toUtc().toIso8601String(),
    'mic_ids_used': micIdsUsed,
    'mic_ids_limit': micIdsLimit,
    'upload_ids_used': uploadIdsUsed,
    'upload_ids_limit': uploadIdsLimit,
    'mic_credits': micCredits,
    'video_credits': videoCredits,
  };

  @override
  List<Object?> get props => [
    periodEnd,
    micIdsUsed,
    micIdsLimit,
    uploadIdsUsed,
    uploadIdsLimit,
    micCredits,
    videoCredits,
  ];
}

/// Who paid for the current subscription, so "Manage" opens the right place.
enum BillingSource {
  appStore('app_store'),
  googlePlay('google_play'),
  stripe('stripe');

  const BillingSource(this.wire);

  final String wire;

  static BillingSource? parse(String? wire) =>
      values.where((s) => s.wire == wire).firstOrNull;
}

/// `GET /me`: the signed-in user, their plan and this month's usage.
class Account extends Equatable {
  const Account({
    required this.userId,
    this.email,
    this.tier = Tier.free,
    this.interval,
    this.renewsAt,
    this.billingSource,
    required this.usage,
    this.webCheckout = false,
  });

  factory Account.fromJson(Map<String, dynamic> json) {
    final user = json['user'] as Map<String, dynamic>;
    final renews = json['renews_at'] as String?;
    return Account(
      userId: user['id'] as String,
      email: user['email'] as String?,
      tier: Tier.parse(json['tier'] as String?),
      interval: BillingInterval.parse(json['interval'] as String?),
      renewsAt: renews == null ? null : DateTime.parse(renews),
      billingSource: BillingSource.parse(json['billing_source'] as String?),
      usage: Usage.fromJson(json['usage'] as Map<String, dynamic>),
      webCheckout: json['web_checkout'] as bool? ?? false,
    );
  }

  final String userId;
  final String? email;
  final Tier tier;
  final BillingInterval? interval;
  final DateTime? renewsAt;
  final BillingSource? billingSource;
  final Usage usage;

  /// The server allows a link to Stripe Checkout for this user (by storefront/region).
  final bool webCheckout;

  Entitlements get entitlements => tier.entitlements;

  Map<String, dynamic> toJson() => {
    'user': {'id': userId, 'email': email},
    'tier': tier.wire,
    'interval': interval?.wire,
    'renews_at': renewsAt?.toUtc().toIso8601String(),
    'billing_source': billingSource?.wire,
    'usage': usage.toJson(),
    'web_checkout': webCheckout,
  };

  Account copyWith({
    Tier? tier,
    BillingInterval? interval,
    BillingSource? billingSource,
    Usage? usage,
  }) => Account(
    userId: userId,
    email: email,
    tier: tier ?? this.tier,
    interval: interval ?? this.interval,
    renewsAt: renewsAt,
    billingSource: billingSource ?? this.billingSource,
    usage: usage ?? this.usage,
    webCheckout: webCheckout,
  );

  @override
  List<Object?> get props => [
    userId,
    email,
    tier,
    interval,
    renewsAt,
    billingSource,
    usage,
    webCheckout,
  ];
}

/// One reserved identification. Pass [id] to the live session; the server only runs
/// the paid lookup for a valid, unspent ticket.
class IdTicket extends Equatable {
  const IdTicket({
    required this.id,
    required this.listenWindow,
    required this.usage,
  });

  factory IdTicket.fromJson(Map<String, dynamic> json) => IdTicket(
    id: json['ticket_id'] as String,
    listenWindow: Duration(seconds: json['listen_seconds'] as int),
    usage: Usage.fromJson(json['usage'] as Map<String, dynamic>),
  );

  final String id;
  final Duration listenWindow;

  /// Usage after the reservation.
  final Usage usage;

  @override
  List<Object?> get props => [id, listenWindow, usage];
}
