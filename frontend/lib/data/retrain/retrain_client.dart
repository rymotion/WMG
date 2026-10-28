import 'dart:convert';

import 'package:equatable/equatable.dart';
import 'package:http/http.dart' as http;

/// Client for the Mac's retrain service (`training/wmg_train/retrain.py serve`).
class RetrainClient {
  RetrainClient({required this.baseUrl, http.Client? client}) : _http = client ?? http.Client();

  final Uri baseUrl;
  final http.Client _http;

  Uri _u(String path) => baseUrl.replace(path: path);

  Future<RetrainStatus> status() async {
    final r = await _http.get(_u('/status')).timeout(const Duration(seconds: 5));
    _check(r);
    return RetrainStatus.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  /// Starts label -> cache -> train both models -> compare -> install.
  /// [labels] are this device's labels (`session_id`, `category`, and `clip_index` for a
  /// single clip), in addition to what the service finds on the Mac itself.
  Future<void> start({required List<Map<String, Object>> labels, bool buildApp = true}) =>
      _post('/retrain', {'labels': labels, 'build_app': buildApp});

  Future<void> approve() => _post('/approve');
  Future<void> reject() => _post('/reject');
  Future<void> cancel() => _post('/cancel');

  /// Opens the rebuilt app; the caller quits this one.
  Future<void> relaunch() => _post('/relaunch');

  Future<void> _post(String path, [Map<String, Object?>? body]) async {
    final r = await _http
        .post(_u(path), headers: {'Content-Type': 'application/json'}, body: jsonEncode(body ?? {}))
        .timeout(const Duration(seconds: 10));
    _check(r);
  }

  void _check(http.Response r) {
    if (r.statusCode >= 300) {
      String msg = r.body;
      try {
        msg = (jsonDecode(r.body) as Map)['error'] as String? ?? msg;
      } catch (_) {}
      throw RetrainException(msg);
    }
  }

  void close() => _http.close();
}

class RetrainException implements Exception {
  const RetrainException(this.message);
  final String message;
  @override
  String toString() => message;
}

/// One model's numbers from its cross-validation (see `summary()` in retrain.py).
class ModelChecks extends Equatable {
  const ModelChecks(this.values, this.run);

  factory ModelChecks.fromJson(Map<String, dynamic> j) => ModelChecks({
    for (final e in j.entries)
      if (e.value is num) e.key: (e.value as num).toDouble(),
  }, j['run'] as String?);

  final Map<String, double> values;
  final String? run;

  @override
  List<Object?> get props => [values, run];
}

class ModelComparison extends Equatable {
  const ModelComparison({required this.newChecks, this.current, this.regressions = const []});

  factory ModelComparison.fromJson(Map<String, dynamic> j) => ModelComparison(
    newChecks: ModelChecks.fromJson(j['new'] as Map<String, dynamic>),
    current: j['current'] == null ? null : ModelChecks.fromJson(j['current'] as Map<String, dynamic>),
    regressions: [for (final r in (j['regressions'] as List? ?? const [])) r as String],
  );

  final ModelChecks newChecks;
  final ModelChecks? current;
  final List<String> regressions;

  @override
  List<Object?> get props => [newChecks, current, regressions];
}

class RetrainStep extends Equatable {
  const RetrainStep(this.name, this.state, this.detail);
  final String name;
  final String state;
  final String? detail;
  @override
  List<Object?> get props => [name, state, detail];
}

/// `GET /status`. [state]: idle, running, awaiting_approval, done, rejected, failed, cancelled.
class RetrainStatus extends Equatable {
  const RetrainStatus({
    this.state = 'idle',
    this.step,
    this.steps = const [],
    this.message,
    this.lastLine = '',
    this.comparison = const {},
    this.current = const {},
    this.builtApp = false,
    this.newRecordings = 0,
    this.conflicts = 0,
    this.startedAt,
  });

  factory RetrainStatus.fromJson(Map<String, dynamic> j) {
    final steps = (j['steps'] as Map<String, dynamic>?) ?? const {};
    final cmp = (j['comparison'] ?? (j['pending'] as Map<String, dynamic>?)?['comparison'])
        as Map<String, dynamic>?;
    final labels = j['labels'] as Map<String, dynamic>?;
    return RetrainStatus(
      state: j['state'] as String? ?? 'idle',
      step: j['step'] as String?,
      steps: [
        for (final e in steps.entries)
          RetrainStep(e.key, (e.value as Map)['state'] as String, (e.value as Map)['detail'] as String?),
      ],
      message: j['message'] as String?,
      lastLine: j['last_line'] as String? ?? '',
      comparison: {
        for (final e in (cmp ?? const {}).entries)
          e.key: ModelComparison.fromJson(e.value as Map<String, dynamic>),
      },
      current: {
        for (final e in ((j['current'] as Map<String, dynamic>?) ?? const {}).entries)
          if (e.value != null) e.key: e.value as String,
      },
      builtApp: j['built_app'] as bool? ?? false,
      newRecordings: (labels?['new'] as List?)?.length ?? 0,
      conflicts: (labels?['conflicts'] as List?)?.length ?? 0,
      startedAt: DateTime.tryParse(j['started_at'] as String? ?? ''),
    );
  }

  final String state;
  final String? step;
  final List<RetrainStep> steps;
  final String? message;
  final String lastLine;

  /// `10s` / `5s` -> new vs current.
  final Map<String, ModelComparison> comparison;

  /// Model runs currently installed in the app.
  final Map<String, String> current;
  final bool builtApp;
  final int newRecordings;
  final int conflicts;
  final DateTime? startedAt;

  bool get running => state == 'running';
  bool get awaitingApproval => state == 'awaiting_approval';

  @override
  List<Object?> get props => [
    state, step, steps, message, lastLine, comparison, current, builtApp, newRecordings, conflicts, startedAt,
  ];
}
