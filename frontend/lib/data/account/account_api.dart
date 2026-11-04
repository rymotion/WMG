import 'dart:convert';

import 'package:http/http.dart' as http;

import '../auth/auth_session.dart';
import '../usage/network_meter.dart';
import 'account.dart';

class AccountException implements Exception {
  const AccountException(this.message, {this.statusCode});

  final String message;
  final int? statusCode;

  @override
  String toString() => message;
}

/// 401: no or expired token.
class NotSignedIn extends AccountException {
  const NotSignedIn() : super('Sign in to identify songs.', statusCode: 401);
}

/// 402: this month's IDs and any credits are spent.
class QuotaExhausted extends AccountException {
  const QuotaExhausted(this.usage)
    : super('No identifications left this month.', statusCode: 402);

  final Usage usage;
}

/// Why a ticket is handed back unspent; the server refunds it.
enum TicketRelease {
  noMatch('no_match'),
  failed('failed'),
  cancelled('cancelled');

  const TicketRelease(this.wire);

  final String wire;
}

/// A purchase the App Store or Google Play reported, for the server to verify and apply.
class StorePurchase {
  const StorePurchase({
    required this.platform,
    required this.productId,
    required this.verificationData,
    this.transactionId,
  });

  /// `app_store` or `google_play`.
  final String platform;
  final String productId;

  /// StoreKit's signed transaction (JWS) or Play's purchase token.
  final String verificationData;
  final String? transactionId;

  Map<String, dynamic> toJson() => {
    'platform': platform,
    'product_id': productId,
    'verification_data': verificationData,
    'transaction_id': ?transactionId,
  };
}

/// Accounts, quota and billing. The server enforces every limit; the app only reflects it.
/// The contract is written up in `docs/billing-api.md`.
abstract interface class AccountApi {
  /// `GET /me`
  Future<Account> me();

  /// `POST /usage/tickets`: reserves one mic identification. Throws [QuotaExhausted].
  Future<IdTicket> reserveMicId();

  /// `POST /usage/tickets/{id}/release`: refunds a ticket that found nothing.
  Future<Usage> releaseTicket(String ticketId, TicketRelease reason);

  /// `POST /usage/network`
  Future<void> reportNetworkUsage(List<NetworkUsage> entries);

  /// `POST /billing/checkout`: a Stripe Checkout URL for [productId].
  Future<Uri> webCheckout(String productId);

  /// `POST /billing/portal`: Stripe's customer portal, for web subscribers.
  Future<Uri> billingPortal();

  /// `POST /billing/store-purchases`: verifies and applies a store purchase.
  Future<Account> applyStorePurchase(StorePurchase purchase);
}

class HttpAccountApi implements AccountApi {
  HttpAccountApi({
    required this.baseUrl,
    required AuthSession auth,
    http.Client? httpClient,
  }) : _auth = auth,
       _http = httpClient ?? http.Client();

  final Uri baseUrl;
  final AuthSession _auth;
  final http.Client _http;

  @override
  Future<Account> me() async =>
      Account.fromJson(await _send('GET', '/me') as Map<String, dynamic>);

  @override
  Future<IdTicket> reserveMicId() async => IdTicket.fromJson(
    await _send('POST', '/usage/tickets', body: {'kind': 'mic'}, expect: 201)
        as Map<String, dynamic>,
  );

  @override
  Future<Usage> releaseTicket(String ticketId, TicketRelease reason) async =>
      Usage.fromJson(
        await _send(
              'POST',
              '/usage/tickets/$ticketId/release',
              body: {'reason': reason.wire},
            )
            as Map<String, dynamic>,
      );

  @override
  Future<void> reportNetworkUsage(List<NetworkUsage> entries) async => _send(
    'POST',
    '/usage/network',
    body: {
      'entries': [for (final e in entries) e.toJson()],
    },
    expect: 204,
  );

  @override
  Future<Uri> webCheckout(String productId) async => _url(
    await _send('POST', '/billing/checkout', body: {'product_id': productId}),
  );

  @override
  Future<Uri> billingPortal() async =>
      _url(await _send('POST', '/billing/portal'));

  @override
  Future<Account> applyStorePurchase(StorePurchase purchase) async =>
      Account.fromJson(
        await _send('POST', '/billing/store-purchases', body: purchase.toJson())
            as Map<String, dynamic>,
      );

  static Uri _url(Object? json) =>
      Uri.parse((json as Map<String, dynamic>)['url'] as String);

  Future<Object?> _send(
    String method,
    String path, {
    Map<String, dynamic>? body,
    int expect = 200,
  }) async {
    final token = await _auth.token();
    if (token == null) throw const NotSignedIn();
    final request = http.Request(method, baseUrl.resolve(path))
      ..headers['authorization'] = 'Bearer $token';
    if (body != null) {
      request.headers['content-type'] = 'application/json';
      request.body = jsonEncode(body);
    }
    final response = await http.Response.fromStream(await _http.send(request));
    final json = response.bodyBytes.isEmpty ? null : _tryDecode(response);
    if (response.statusCode == expect) return json;
    if (response.statusCode == 401) throw const NotSignedIn();
    if (response.statusCode == 402 &&
        json is Map<String, dynamic> &&
        json['usage'] is Map<String, dynamic>) {
      throw QuotaExhausted(
        Usage.fromJson(json['usage'] as Map<String, dynamic>),
      );
    }
    final message = json is Map && json['error'] != null
        ? '${json['error']}'
        : 'Request failed';
    throw AccountException(message, statusCode: response.statusCode);
  }

  static Object? _tryDecode(http.Response response) {
    try {
      return jsonDecode(utf8.decode(response.bodyBytes));
    } on FormatException {
      return null;
    }
  }
}
