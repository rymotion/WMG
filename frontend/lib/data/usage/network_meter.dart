import 'dart:async';

import 'package:equatable/equatable.dart';
import 'package:http/http.dart' as http;

/// Network use for one route over one reporting period. Routes are templates
/// (`GET /audio/:id/identification`): no ids, query strings or bodies leave the device.
class NetworkUsage extends Equatable {
  const NetworkUsage({
    required this.route,
    this.calls = 0,
    this.failures = 0,
    this.bytesUp = 0,
    this.bytesDown = 0,
  });

  final String route;
  final int calls;

  /// Non-2xx responses and transport errors.
  final int failures;
  final int bytesUp;
  final int bytesDown;

  NetworkUsage add({
    int calls = 0,
    int failures = 0,
    int bytesUp = 0,
    int bytesDown = 0,
  }) => NetworkUsage(
    route: route,
    calls: this.calls + calls,
    failures: this.failures + failures,
    bytesUp: this.bytesUp + bytesUp,
    bytesDown: this.bytesDown + bytesDown,
  );

  Map<String, dynamic> toJson() => {
    'route': route,
    'calls': calls,
    'failures': failures,
    'bytes_up': bytesUp,
    'bytes_down': bytesDown,
  };

  @override
  List<Object?> get props => [route, calls, failures, bytesUp, bytesDown];
}

/// Tallies every backend call the app makes; [drain] hands the totals to the reporter.
class NetworkMeter {
  final _byRoute = <String, NetworkUsage>{};

  void record(
    String route, {
    int calls = 1,
    int failures = 0,
    int bytesUp = 0,
    int bytesDown = 0,
  }) {
    final current = _byRoute[route] ?? NetworkUsage(route: route);
    _byRoute[route] = current.add(
      calls: calls,
      failures: failures,
      bytesUp: bytesUp,
      bytesDown: bytesDown,
    );
  }

  /// One `/live` session: PCM up, JSON messages down (not counted per frame).
  void recordLive({required int bytesUp, required bool failed}) =>
      record('WS /live', bytesUp: bytesUp, failures: failed ? 1 : 0);

  bool get isEmpty => _byRoute.isEmpty;

  /// Totals since the last drain, and a fresh start.
  List<NetworkUsage> drain() {
    final out = _byRoute.values.toList()
      ..sort((a, b) => a.route.compareTo(b.route));
    _byRoute.clear();
    return out;
  }

  /// Puts back totals whose upload failed, so they go out with the next report.
  void restore(List<NetworkUsage> entries) {
    for (final e in entries) {
      record(
        e.route,
        calls: e.calls,
        failures: e.failures,
        bytesUp: e.bytesUp,
        bytesDown: e.bytesDown,
      );
    }
  }

  static final _idSegment = RegExp(r'^([0-9a-fA-F-]{16,}|\d+)$');

  /// `POST /audio/ef86…/identification?x=1` → `POST /audio/:id/identification`.
  static String routeOf(String method, Uri url) {
    final path = url.pathSegments
        .map((s) => _idSegment.hasMatch(s) ? ':id' : s)
        .join('/');
    return '$method /$path';
  }
}

/// An [http.Client] that reports each request to a [NetworkMeter].
class MeteredHttpClient extends http.BaseClient {
  MeteredHttpClient(this._meter, [http.Client? inner])
    : _inner = inner ?? http.Client();

  final NetworkMeter _meter;
  final http.Client _inner;

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) async {
    final route = NetworkMeter.routeOf(request.method, request.url);
    final up = request.contentLength ?? 0;
    final http.StreamedResponse response;
    try {
      response = await _inner.send(request);
    } catch (_) {
      _meter.record(route, failures: 1, bytesUp: up);
      rethrow;
    }
    final failed = response.statusCode < 200 || response.statusCode >= 300;
    var down = 0;
    final counted = response.stream.map((chunk) {
      down += chunk.length;
      return chunk;
    });
    return http.StreamedResponse(
      counted.transform(
        StreamTransformer.fromHandlers(
          handleDone: (sink) {
            _meter.record(
              route,
              failures: failed ? 1 : 0,
              bytesUp: up,
              bytesDown: down,
            );
            sink.close();
          },
        ),
      ),
      response.statusCode,
      contentLength: response.contentLength,
      request: response.request,
      headers: response.headers,
      isRedirect: response.isRedirect,
      persistentConnection: response.persistentConnection,
      reasonPhrase: response.reasonPhrase,
    );
  }

  @override
  void close() => _inner.close();
}
