import 'dart:async';
import 'dart:io' show Platform;

import 'package:in_app_purchase/in_app_purchase.dart';

import '../account/account_api.dart';
import '../account/catalog.dart';

/// Something the store reported about a purchase.
sealed class StoreEvent {
  const StoreEvent();
}

/// Paid; send [purchase] to the server, then call [finish] so the store stops redelivering it.
final class StorePurchased extends StoreEvent {
  const StorePurchased(this.purchase, this.finish);

  final StorePurchase purchase;
  final Future<void> Function() finish;
}

/// Waiting on the store (Ask to Buy, a slow card). It arrives later as [StorePurchased].
final class StorePending extends StoreEvent {
  const StorePending();
}

final class StoreFailed extends StoreEvent {
  const StoreFailed(this.message);

  final String message;
}

final class StoreCancelled extends StoreEvent {
  const StoreCancelled();
}

/// App Store / Google Play billing.
abstract interface class StoreBilling {
  Stream<StoreEvent> get events;

  /// Localised prices by product id, for the products the store knows.
  Future<Map<String, String>> prices();

  Future<void> buy(Product product);

  /// Redelivers past subscriptions as [StorePurchased] (required by App Review).
  Future<void> restore();
}

class InAppStoreBilling implements StoreBilling {
  InAppStoreBilling({InAppPurchase? iap})
    : _iap = iap ?? InAppPurchase.instance;

  final InAppPurchase _iap;
  final _details = <String, ProductDetails>{};

  static String get _platform => Platform.isIOS ? 'app_store' : 'google_play';

  @override
  late final Stream<StoreEvent> events = _iap.purchaseStream
      .expand((list) => list.map(_toEvent))
      .asBroadcastStream();

  StoreEvent _toEvent(PurchaseDetails d) => switch (d.status) {
    PurchaseStatus.pending => const StorePending(),
    PurchaseStatus.canceled => const StoreCancelled(),
    PurchaseStatus.error => StoreFailed(
      d.error?.message ?? 'The store reported an error.',
    ),
    PurchaseStatus.purchased || PurchaseStatus.restored => StorePurchased(
      StorePurchase(
        platform: _platform,
        productId: d.productID,
        verificationData: d.verificationData.serverVerificationData,
        transactionId: d.purchaseID,
      ),
      () async {
        if (d.pendingCompletePurchase) await _iap.completePurchase(d);
      },
    ),
  };

  @override
  Future<Map<String, String>> prices() async {
    if (!await _iap.isAvailable()) return const {};
    final response = await _iap.queryProductDetails({
      for (final p in Catalog.all) p.id,
    });
    for (final d in response.productDetails) {
      _details[d.id] = d;
    }
    return {for (final d in response.productDetails) d.id: d.price};
  }

  @override
  Future<void> buy(Product product) async {
    if (_details[product.id] == null) await prices();
    final details = _details[product.id];
    if (details == null)
      throw StateError(
        '${product.title} isn\'t available from the store right now.',
      );
    final param = PurchaseParam(productDetails: details);
    if (product.isSubscription) {
      await _iap.buyNonConsumable(purchaseParam: param);
    } else {
      await _iap.buyConsumable(purchaseParam: param);
    }
  }

  @override
  Future<void> restore() => _iap.restorePurchases();
}
