import 'plan.dart';

enum ProductKind { subscription, micCredits, videoCredits }

/// Everything sold, keyed by one id shared by the App Store, Google Play and Stripe (the
/// backend maps it to a Stripe Price).
///
/// In-app monthly subscriptions cost 30% more than on the web to cover the store's cut;
/// yearly plans and credit packs cost the same everywhere. The app shows the store's own
/// localised price when it has loaded; these numbers are the USD fallback and the reference
/// for setting up App Store Connect / Play Console.
class Product {
  const Product({
    required this.id,
    required this.kind,
    required this.title,
    required this.webPriceUsd,
    required this.storePriceUsd,
    this.tier,
    this.interval,
    this.credits = 0,
  });

  final String id;
  final ProductKind kind;
  final String title;
  final double webPriceUsd;
  final double storePriceUsd;
  final Tier? tier;
  final BillingInterval? interval;

  /// IDs added by a credit pack.
  final int credits;

  bool get isSubscription => kind == ProductKind.subscription;
}

abstract final class Catalog {
  static const plusMonthly = Product(
    id: 'org.activeie.wmg.plus.monthly',
    kind: ProductKind.subscription,
    title: 'Plus · monthly',
    tier: Tier.plus,
    interval: BillingInterval.month,
    webPriceUsd: 6.99,
    storePriceUsd: 9.09,
  );
  static const plusYearly = Product(
    id: 'org.activeie.wmg.plus.yearly',
    kind: ProductKind.subscription,
    title: 'Plus · yearly',
    tier: Tier.plus,
    interval: BillingInterval.year,
    webPriceUsd: 49.99,
    storePriceUsd: 49.99,
  );
  static const proMonthly = Product(
    id: 'org.activeie.wmg.pro.monthly',
    kind: ProductKind.subscription,
    title: 'Pro Creator · monthly',
    tier: Tier.pro,
    interval: BillingInterval.month,
    webPriceUsd: 14.99,
    storePriceUsd: 19.49,
  );
  static const proYearly = Product(
    id: 'org.activeie.wmg.pro.yearly',
    kind: ProductKind.subscription,
    title: 'Pro Creator · yearly',
    tier: Tier.pro,
    interval: BillingInterval.year,
    webPriceUsd: 119.99,
    storePriceUsd: 119.99,
  );
  static const mic20 = Product(
    id: 'org.activeie.wmg.credits.mic20',
    kind: ProductKind.micCredits,
    title: '20 song IDs',
    credits: 20,
    webPriceUsd: 1.99,
    storePriceUsd: 1.99,
  );
  static const mic100 = Product(
    id: 'org.activeie.wmg.credits.mic100',
    kind: ProductKind.micCredits,
    title: '100 song IDs',
    credits: 100,
    webPriceUsd: 6.99,
    storePriceUsd: 6.99,
  );
  static const video10 = Product(
    id: 'org.activeie.wmg.credits.video10',
    kind: ProductKind.videoCredits,
    title: '10 video IDs',
    credits: 10,
    webPriceUsd: 3.99,
    storePriceUsd: 3.99,
  );

  static const subscriptions = [plusMonthly, plusYearly, proMonthly, proYearly];
  static const creditPacks = [mic20, mic100, video10];
  static const all = [...subscriptions, ...creditPacks];

  static Product? byId(String id) => all.where((p) => p.id == id).firstOrNull;

  static Product subscription(Tier tier, BillingInterval interval) =>
      subscriptions.firstWhere((p) => p.tier == tier && p.interval == interval);
}
