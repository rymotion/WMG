import 'dart:async';
import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:wmg_frontend/data/account/account.dart';
import 'package:wmg_frontend/data/account/account_api.dart';
import 'package:wmg_frontend/data/account/catalog.dart';
import 'package:wmg_frontend/data/account/fake_account_api.dart';
import 'package:wmg_frontend/data/account/plan.dart';
import 'package:wmg_frontend/data/auth/auth_session.dart';
import 'package:wmg_frontend/data/billing/store_billing.dart';
import 'package:wmg_frontend/data/usage/network_meter.dart';
import 'package:wmg_frontend/features/account/cubit/account_cubit.dart';
import 'package:wmg_frontend/features/results/cubit/results_cubit.dart';
import 'package:wmg_frontend/features/results/models/saved_session.dart';

class _FakeStore implements StoreBilling {
  final _events = StreamController<StoreEvent>.broadcast();
  final bought = <String>[];
  var finished = 0;

  @override
  Stream<StoreEvent> get events => _events.stream;

  @override
  Future<Map<String, String>> prices() async => {Catalog.plusMonthly.id: '\$9.49'};

  @override
  Future<void> buy(Product product) async {
    bought.add(product.id);
    _events.add(
      StorePurchased(
        StorePurchase(platform: 'app_store', productId: product.id, verificationData: 'jws'),
        () async => finished++,
      ),
    );
  }

  @override
  Future<void> restore() async {}
}

void main() {
  group('Usage', () {
    final end = DateTime.utc(2026, 12);

    test('free IDs left include bought credits', () {
      final u = Usage(periodEnd: end, micIdsUsed: 10, micIdsLimit: 10, micCredits: 20);
      expect(u.micIdsLeft, 20);
      expect(u.canIdentify, isTrue);
      expect(Usage(periodEnd: end, micIdsUsed: 10, micIdsLimit: 10).canIdentify, isFalse);
    });

    test('unlimited plans have no count', () {
      final u = Usage(periodEnd: end, micIdsUsed: 500);
      expect(u.micIdsLeft, isNull);
      expect(u.canIdentify, isTrue);
    });

    test('account survives a JSON round trip', () {
      final a = Account(
        userId: 'u',
        email: 'a@b.c',
        tier: Tier.pro,
        interval: BillingInterval.year,
        billingSource: BillingSource.stripe,
        usage: Usage(periodEnd: end, uploadIdsLimit: 100, uploadIdsUsed: 3),
        webCheckout: true,
      );
      expect(Account.fromJson(jsonDecode(jsonEncode(a.toJson())) as Map<String, dynamic>), a);
    });
  });

  group('Catalog', () {
    test('in-app monthly costs ~30% more; yearly and credit packs match the web', () {
      for (final p in [Catalog.plusMonthly, Catalog.proMonthly]) {
        expect(p.storePriceUsd / p.webPriceUsd, closeTo(1.3, 0.01));
      }
      for (final p in [Catalog.plusYearly, Catalog.proYearly, ...Catalog.creditPacks]) {
        expect(p.storePriceUsd, p.webPriceUsd);
      }
    });
  });

  group('AccountCubit', () {
    late DevAuthSession auth;
    late FakeAccountApi api;
    late NetworkMeter meter;
    late _FakeStore store;
    final opened = <Uri>[];

    AccountCubit build() => AccountCubit(
      auth: auth,
      api: api,
      meter: meter,
      store: store,
      openUrl: (u) async {
        opened.add(u);
        return true;
      },
    );

    setUp(() {
      auth = DevAuthSession();
      api = FakeAccountApi(auth: auth);
      meter = NetworkMeter();
      store = _FakeStore();
      opened.clear();
    });

    test('signed out: the tap still checks for AI where the model runs, else asks to sign in', () async {
      final cubit = build();
      await cubit.start();
      expect(await cubit.planListen(hasDetector: true), const AiOnlyPlan(UpsellReason.signedOut));
      expect(await cubit.planListen(hasDetector: false), const BlockedPlan(UpsellReason.signedOut));
      await cubit.close();
    });

    test('free: 10 IDs a month, then AI-only; a miss is refunded', () async {
      final cubit = build();
      await cubit.start();
      await cubit.signIn();
      await pumpEventQueue();
      expect(cubit.state.usage?.micIdsLeft, 10);

      final first = await cubit.planListen(hasDetector: true) as IdentifyPlan;
      expect(first.ticket.listenWindow, const Duration(seconds: 10));
      expect(cubit.state.usage?.micIdsLeft, 9);
      await cubit.finishIdentification(first.ticket, matched: false);
      expect(cubit.state.usage?.micIdsLeft, 10, reason: 'no match: it doesn\'t count');

      for (var i = 0; i < 10; i++) {
        final plan = await cubit.planListen(hasDetector: true) as IdentifyPlan;
        await cubit.finishIdentification(plan.ticket, matched: true);
      }
      expect(cubit.state.usage?.micIdsLeft, 0);
      expect(await cubit.planListen(hasDetector: true), const AiOnlyPlan(UpsellReason.quotaExhausted));
      expect(await cubit.planListen(hasDetector: false), const BlockedPlan(UpsellReason.quotaExhausted));
      await cubit.close();
    });

    test('credit packs are spent after the monthly allowance', () async {
      api
        ..micIdsUsed = 10
        ..micCredits = 2;
      final cubit = build();
      await cubit.start();
      await cubit.signIn();
      expect(await cubit.planListen(hasDetector: false), isA<IdentifyPlan>());
      expect(api.micCredits, 1);
      await cubit.close();
    });

    test('an in-app purchase is verified by the server, then finished with the store', () async {
      final cubit = build();
      await cubit.start();
      await cubit.signIn();
      expect(cubit.state.storePrice(Catalog.plusMonthly), '\$9.49');
      expect(cubit.state.storePrice(Catalog.proMonthly), '\$19.49', reason: 'catalog fallback');

      await cubit.buyInApp(Catalog.plusMonthly);
      await pumpEventQueue();
      expect(cubit.state.tier, Tier.plus);
      expect(cubit.state.account?.billingSource, BillingSource.appStore);
      expect(cubit.state.usage?.unlimitedMic, isTrue);
      expect(store.finished, 1);

      final plan = await cubit.planListen(hasDetector: false) as IdentifyPlan;
      expect(plan.ticket.listenWindow, const Duration(seconds: 30));
      await cubit.close();
    });

    test('web checkout opens the Stripe URL from the server', () async {
      final cubit = build();
      await cubit.start();
      await cubit.signIn();
      await cubit.buyOnWeb(Catalog.proYearly);
      expect(opened.single.host, 'checkout.stripe.com');
      await cubit.close();
    });

    test('network usage is reported while signed in, and kept when the report fails', () async {
      final cubit = build();
      await cubit.start();
      meter.record('GET /audio');
      await cubit.reportNetworkUsage();
      expect(api.reported, isEmpty, reason: 'signed out: nothing sent');

      await cubit.signIn();
      await cubit.reportNetworkUsage();
      expect(api.reported.single.route, 'GET /audio');
      expect(meter.isEmpty, isTrue);

      meter.record('POST /audio', bytesUp: 10);
      await auth.signOut();
      // The cubit still thinks it's signed in until the change lands; the call fails and is kept.
      await cubit.reportNetworkUsage();
      expect(meter.isEmpty, isFalse);
      await cubit.close();
    });
  });

  group('NetworkMeter', () {
    test('routes drop ids and query strings', () {
      expect(
        NetworkMeter.routeOf('GET', Uri.parse('http://h/audio/ef864cac-5be7-4e03-b93f-e579e6ae2059/identification?x=1')),
        'GET /audio/:id/identification',
      );
      expect(NetworkMeter.routeOf('POST', Uri.parse('http://h/usage/tickets/42/release')), 'POST /usage/tickets/:id/release');
    });

    test('the metered client counts calls, failures and bytes', () async {
      final meter = NetworkMeter();
      final client = MeteredHttpClient(
        meter,
        MockClient((r) async => http.Response(r.url.path == '/bad' ? 'no' : 'hello', r.url.path == '/bad' ? 500 : 200)),
      );
      await client.post(Uri.parse('http://h/audio'), body: 'abc');
      await client.post(Uri.parse('http://h/audio'), body: 'de');
      await client.get(Uri.parse('http://h/bad'));
      final usage = {for (final u in meter.drain()) u.route: u};
      expect(usage['POST /audio'], const NetworkUsage(route: 'POST /audio', calls: 2, bytesUp: 5, bytesDown: 10));
      expect(usage['GET /bad']?.failures, 1);
    });
  });

  group('history by plan', () {
    final now = DateTime(2026, 11, 2);
    SavedSession at(int daysAgo) =>
        SavedSession(id: '$daysAgo', startedAt: now.subtract(Duration(days: daysAgo)), durationS: 10);
    final sessions = [at(1), at(6), at(8), at(30)];

    test('free shows the last 7 days and counts the rest as hidden', () {
      final v = visibleSessions(sessions, retention: const Duration(days: 7), now: now);
      expect(v.shown.map((s) => s.id), ['1', '6']);
      expect(v.hidden, 2);
    });

    test('paid plans show everything and can search', () {
      expect(visibleSessions(sessions, now: now).shown, hasLength(4));
      expect(visibleSessions(sessions, query: '2026-10-25', now: now).shown.map((s) => s.id), ['8']);
    });
  });
}
