import 'dart:async';

import 'package:equatable/equatable.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../../data/account/account.dart';
import '../../../data/account/account_api.dart';
import '../../../data/account/catalog.dart';
import '../../../data/account/plan.dart';
import '../../../data/auth/auth_session.dart';
import '../../../data/billing/store_billing.dart';
import '../../../data/usage/network_meter.dart';

part 'account_state.dart';

/// The signed-in user's plan and quota, purchases, and usage reporting.
///
/// Every limit is enforced by the server: a tap that identifies a song first reserves a
/// ticket ([planListen]), and a ticket that finds nothing is refunded ([finishIdentification]).
class AccountCubit extends Cubit<AccountState> {
  AccountCubit({
    required AuthSession auth,
    required AccountApi api,
    required NetworkMeter meter,
    required Future<bool> Function(Uri) openUrl,
    StoreBilling? store,
    this.reportEvery = const Duration(minutes: 5),
  }) : _auth = auth,
       _api = api,
       _meter = meter,
       _openUrl = openUrl,
       _store = store,
       super(AccountState(authAvailable: auth.isAvailable));

  final AuthSession _auth;
  final AccountApi _api;
  final NetworkMeter _meter;
  final Future<bool> Function(Uri) _openUrl;

  /// Null where in-app purchase isn't offered (tests, desktop).
  final StoreBilling? _store;
  final Duration reportEvery;

  StreamSubscription<AuthUser?>? _authSub;
  StreamSubscription<StoreEvent>? _storeSub;
  Timer? _reportTimer;

  Future<void> start() async {
    _authSub = _auth.changes.listen(
      (user) => user == null ? _signedOut() : refresh(),
    );
    _storeSub = _store?.events.listen(_onStoreEvent);
    _reportTimer = Timer.periodic(reportEvery, (_) => reportNetworkUsage());
    final store = _store;
    if (store != null) {
      unawaited(
        store.prices().then((p) {
          if (!isClosed) emit(state.copyWith(storePrices: p));
        }, onError: (Object _) {}),
      );
    }
    if (_auth.currentUser != null) await refresh();
  }

  Future<void> signIn() async {
    if (await _auth.signIn() != null) await refresh();
  }

  Future<void> signOut() async {
    await reportNetworkUsage();
    await _auth.signOut();
    _signedOut();
  }

  void _signedOut() {
    if (!isClosed)
      emit(
        state.copyWith(status: AccountStatus.signedOut, account: () => null),
      );
  }

  Future<void> refresh() async {
    if (_auth.currentUser == null) return _signedOut();
    emit(state.copyWith(status: state.signedIn ? null : AccountStatus.loading));
    try {
      final account = await _api.me();
      emit(state.copyWith(status: AccountStatus.ready, account: () => account));
    } on NotSignedIn {
      _signedOut();
    } catch (e) {
      // Keep showing the last known plan; a failed refresh shouldn't sign anyone out.
      emit(
        state.copyWith(
          status: state.signedIn ? AccountStatus.ready : AccountStatus.failure,
          message: () => '$e',
        ),
      );
    }
  }

  /// Decides what one mic tap does, reserving an identification if it can.
  /// [hasDetector]: the on-device AI model can run here (free, no account needed).
  Future<ListenPlan> planListen({required bool hasDetector}) async {
    ListenPlan fallback(UpsellReason r) =>
        hasDetector ? AiOnlyPlan(r) : BlockedPlan(r);
    if (!state.signedIn) return fallback(UpsellReason.signedOut);
    try {
      final ticket = await _api.reserveMicId();
      _setUsage(ticket.usage);
      return IdentifyPlan(ticket);
    } on QuotaExhausted catch (e) {
      _setUsage(e.usage);
      return fallback(UpsellReason.quotaExhausted);
    } on NotSignedIn {
      _signedOut();
      return fallback(UpsellReason.signedOut);
    } catch (_) {
      return fallback(UpsellReason.unreachable);
    }
  }

  /// Hands back a ticket whose session matched nothing (or failed), so it isn't counted.
  Future<void> finishIdentification(
    IdTicket ticket, {
    required bool matched,
    bool failed = false,
  }) async {
    if (matched) return;
    try {
      _setUsage(
        await _api.releaseTicket(
          ticket.id,
          failed ? TicketRelease.failed : TicketRelease.noMatch,
        ),
      );
    } catch (_) {
      // The server expires unreleased tickets that never reached a lookup.
    }
  }

  void _setUsage(Usage usage) {
    final account = state.account;
    if (account != null && !isClosed)
      emit(state.copyWith(account: () => account.copyWith(usage: usage)));
  }

  /// Sends the network totals gathered since the last report.
  Future<void> reportNetworkUsage() async {
    if (!state.signedIn || _meter.isEmpty) return;
    final entries = _meter.drain();
    try {
      await _api.reportNetworkUsage(entries);
    } catch (_) {
      _meter.restore(entries);
    }
  }

  // Purchases.

  Future<void> buyInApp(Product product) async {
    final store = _store;
    if (store == null) return;
    emit(state.copyWith(purchasing: true, message: () => null));
    try {
      await store.buy(product);
    } catch (e) {
      emit(state.copyWith(purchasing: false, message: () => '$e'));
    }
  }

  /// Opens Stripe Checkout in the browser; the plan updates on the next [refresh] (the app
  /// refreshes when it comes back to the foreground).
  Future<void> buyOnWeb(Product product) async {
    emit(state.copyWith(message: () => null));
    try {
      await _openUrl(await _api.webCheckout(product.id));
    } catch (e) {
      emit(state.copyWith(message: () => 'Couldn\'t open checkout: $e'));
    }
  }

  Future<void> restorePurchases() async {
    await _store?.restore();
  }

  /// Where the current subscription is managed: Stripe's portal or the store's settings.
  Future<void> manageSubscription() async {
    try {
      final url = switch (state.account?.billingSource) {
        BillingSource.stripe => await _api.billingPortal(),
        BillingSource.googlePlay => Uri.parse(
          'https://play.google.com/store/account/subscriptions',
        ),
        _ => Uri.parse('https://apps.apple.com/account/subscriptions'),
      };
      await _openUrl(url);
    } catch (e) {
      emit(state.copyWith(message: () => '$e'));
    }
  }

  Future<void> _onStoreEvent(StoreEvent event) async {
    switch (event) {
      case StorePending():
        emit(state.copyWith(purchasing: true));
      case StoreCancelled():
        emit(state.copyWith(purchasing: false));
      case StoreFailed(:final message):
        emit(state.copyWith(purchasing: false, message: () => message));
      case StorePurchased(:final purchase, :final finish):
        try {
          final account = await _api.applyStorePurchase(purchase);
          // Only finish once the server has it; otherwise the store redelivers it next launch.
          await finish();
          emit(
            state.copyWith(
              status: AccountStatus.ready,
              account: () => account,
              purchasing: false,
              message: () =>
                  'Thanks! ${Catalog.byId(purchase.productId)?.title ?? 'Your purchase'} is active.',
            ),
          );
        } catch (e) {
          emit(
            state.copyWith(
              purchasing: false,
              message: () =>
                  'Payment received, but we couldn\'t confirm it yet: $e',
            ),
          );
        }
    }
  }

  /// Clears [AccountState.message] once shown.
  void messageShown() => emit(state.copyWith(message: () => null));

  @override
  Future<void> close() async {
    _reportTimer?.cancel();
    await reportNetworkUsage();
    await _authSub?.cancel();
    await _storeSub?.cancel();
    return super.close();
  }
}
