part of 'account_cubit.dart';

enum AccountStatus { signedOut, loading, ready, failure }

final class AccountState extends Equatable {
  const AccountState({
    this.status = AccountStatus.signedOut,
    this.account,
    this.storePrices = const {},
    this.purchasing = false,
    this.message,
    this.authAvailable = true,
  });

  final AccountStatus status;
  final Account? account;

  /// Localised store prices by product id; missing ones fall back to the catalog.
  final Map<String, String> storePrices;

  /// A store or web purchase is in flight.
  final bool purchasing;

  /// One-off notice for a snackbar (purchase done, failed, restored…).
  final String? message;

  /// False until an identity provider is wired in.
  final bool authAvailable;

  bool get signedIn => account != null;
  Tier get tier => account?.tier ?? Tier.free;
  Entitlements get entitlements => tier.entitlements;
  Usage? get usage => account?.usage;

  /// The in-app price to show for [product].
  String storePrice(Product product) =>
      storePrices[product.id] ??
      '\$${product.storePriceUsd.toStringAsFixed(2)}';

  AccountState copyWith({
    AccountStatus? status,
    Account? Function()? account,
    Map<String, String>? storePrices,
    bool? purchasing,
    String? Function()? message,
    bool? authAvailable,
  }) => AccountState(
    status: status ?? this.status,
    account: account != null ? account() : this.account,
    storePrices: storePrices ?? this.storePrices,
    purchasing: purchasing ?? this.purchasing,
    message: message != null ? message() : this.message,
    authAvailable: authAvailable ?? this.authAvailable,
  );

  @override
  List<Object?> get props => [
    status,
    account,
    storePrices,
    purchasing,
    message,
    authAvailable,
  ];
}

/// Why a tap couldn't name the song.
enum UpsellReason { signedOut, quotaExhausted, unreachable }

/// What one mic tap should do.
sealed class ListenPlan extends Equatable {
  const ListenPlan();

  @override
  List<Object?> get props => [];
}

/// Identify the song with [ticket], listening for at most its window.
final class IdentifyPlan extends ListenPlan {
  const IdentifyPlan(this.ticket);

  final IdTicket ticket;

  @override
  List<Object?> get props => [ticket];
}

/// Run on-device AI detection only (free for everyone); explain [reason] alongside.
final class AiOnlyPlan extends ListenPlan {
  const AiOnlyPlan(this.reason);

  final UpsellReason reason;

  @override
  List<Object?> get props => [reason];
}

/// Nothing can run (no on-device model on this platform): resolve [reason] first.
final class BlockedPlan extends ListenPlan {
  const BlockedPlan(this.reason);

  final UpsellReason reason;

  @override
  List<Object?> get props => [reason];
}
