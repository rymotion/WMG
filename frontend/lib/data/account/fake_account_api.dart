import '../auth/auth_session.dart';
import '../usage/network_meter.dart';
import 'account.dart';
import 'account_api.dart';
import 'catalog.dart';
import 'plan.dart';

/// In-memory backend with the server's rules, for development until the real endpoints
/// exist (`--dart-define=ACCOUNT_BACKEND=fake`, the default) and for tests.
class FakeAccountApi implements AccountApi {
  FakeAccountApi({
    required AuthSession auth,
    this.tier = Tier.free,
    this.webCheckoutAllowed = true,
    DateTime Function()? now,
  }) : _auth = auth,
       _now = now ?? DateTime.now;

  final AuthSession _auth;
  final DateTime Function() _now;
  Tier tier;
  BillingInterval? interval;
  BillingSource? source;
  bool webCheckoutAllowed;
  int micIdsUsed = 0;
  int micCredits = 0;
  int videoCredits = 0;
  final reported = <NetworkUsage>[];
  final _open = <String, bool>{}; // ticket id → paid from credits
  var _nextTicket = 0;

  Usage get _usage {
    final now = _now();
    final e = tier.entitlements;
    return Usage(
      periodEnd: DateTime(now.year, now.month + 1),
      micIdsUsed: micIdsUsed,
      micIdsLimit: e.monthlyMicIds,
      uploadIdsLimit: e.monthlyUploadIds,
      micCredits: micCredits,
      videoCredits: videoCredits,
    );
  }

  AuthUser _user() => _auth.currentUser ?? (throw const NotSignedIn());

  @override
  Future<Account> me() async {
    final user = _user();
    return Account(
      userId: user.id,
      email: user.email,
      tier: tier,
      interval: interval,
      billingSource: source,
      usage: _usage,
      webCheckout: webCheckoutAllowed,
    );
  }

  @override
  Future<IdTicket> reserveMicId() async {
    _user();
    final usage = _usage;
    if (!usage.canIdentify) throw QuotaExhausted(usage);
    final fromCredits = !usage.unlimitedMic && micIdsUsed >= usage.micIdsLimit!;
    if (fromCredits) {
      micCredits--;
    } else {
      micIdsUsed++;
    }
    final id = 'ticket-${_nextTicket++}';
    _open[id] = fromCredits;
    return IdTicket(
      id: id,
      listenWindow: tier.entitlements.listenWindow,
      usage: _usage,
    );
  }

  @override
  Future<Usage> releaseTicket(String ticketId, TicketRelease reason) async {
    _user();
    final fromCredits = _open.remove(ticketId);
    if (fromCredits == true) micCredits++;
    if (fromCredits == false) micIdsUsed--;
    return _usage;
  }

  @override
  Future<void> reportNetworkUsage(List<NetworkUsage> entries) async {
    _user();
    reported.addAll(entries);
  }

  @override
  Future<Uri> webCheckout(String productId) async {
    _user();
    // The real server answers with a Stripe Checkout session URL; here, apply it at once.
    _apply(Catalog.byId(productId)!, BillingSource.stripe);
    return Uri.parse('https://checkout.stripe.com/test/$productId');
  }

  @override
  Future<Uri> billingPortal() async =>
      Uri.parse('https://billing.stripe.com/test/portal');

  @override
  Future<Account> applyStorePurchase(StorePurchase purchase) async {
    _user();
    final product = Catalog.byId(purchase.productId);
    if (product == null)
      throw AccountException(
        'Unknown product ${purchase.productId}',
        statusCode: 400,
      );
    _apply(
      product,
      purchase.platform == 'google_play'
          ? BillingSource.googlePlay
          : BillingSource.appStore,
    );
    return me();
  }

  void _apply(Product product, BillingSource via) {
    switch (product.kind) {
      case ProductKind.subscription:
        tier = product.tier!;
        interval = product.interval;
        source = via;
      case ProductKind.micCredits:
        micCredits += product.credits;
      case ProductKind.videoCredits:
        videoCredits += product.credits;
    }
  }
}
