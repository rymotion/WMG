import 'dart:async';

import 'package:equatable/equatable.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../../data/retrain/retrain_client.dart';
import '../../results/data/session_history_store.dart';

class RetrainState extends Equatable {
  const RetrainState({this.reachable, this.status = const RetrainStatus(), this.error, this.sending = false});

  /// null until the first check; false when the Mac's retrain service isn't running.
  final bool? reachable;
  final RetrainStatus status;
  final String? error;
  final bool sending;

  RetrainState copyWith({bool? reachable, RetrainStatus? status, String? Function()? error, bool? sending}) =>
      RetrainState(
        reachable: reachable ?? this.reachable,
        status: status ?? this.status,
        error: error != null ? error() : this.error,
        sending: sending ?? this.sending,
      );

  @override
  List<Object?> get props => [reachable, status, error, sending];
}

/// Drives the "Retrain models" button: starts a retrain on the Mac, polls its progress,
/// and handles approval when the new models are worse than the current ones.
class RetrainCubit extends Cubit<RetrainState> {
  RetrainCubit({required RetrainClient client, required SessionHistoryStore history})
    : _client = client,
      _history = history,
      super(const RetrainState());

  final RetrainClient _client;
  final SessionHistoryStore _history;
  Timer? _timer;

  /// Polls every [period] while the page is open; faster polling isn't useful, training
  /// takes over an hour.
  void watch({Duration period = const Duration(seconds: 3)}) {
    _timer?.cancel();
    refresh();
    _timer = Timer.periodic(period, (_) => refresh());
  }

  void unwatch() {
    _timer?.cancel();
    _timer = null;
  }

  Future<void> refresh() async {
    try {
      final s = await _client.status();
      if (!isClosed) emit(state.copyWith(reachable: true, status: s));
    } catch (_) {
      if (!isClosed) emit(state.copyWith(reachable: false));
    }
  }

  /// This device's labels for recordings that went through the backend (their audio is
  /// there): one row for a whole recording, plus one per clip labelled on its own.
  Future<List<Map<String, Object>>> _labels() async => [
    for (final s in await _history.list())
      if (s.serverSessionId != null) ...[
        if (s.label != null) {'session_id': s.serverSessionId!, 'category': s.label!.category.wire, 'id': s.id},
        for (final c in s.clips)
          if (c.label != null)
            {
              'session_id': s.serverSessionId!,
              'category': c.label!.category.wire,
              'clip_index': c.index,
              'id': s.id,
            },
      ],
  ];

  Future<void> start() => _send(() async => _client.start(labels: await _labels()));
  Future<void> approve() => _send(_client.approve);
  Future<void> reject() => _send(_client.reject);
  Future<void> cancel() => _send(_client.cancel);
  Future<void> relaunch() => _send(_client.relaunch);

  Future<void> _send(Future<void> Function() call) async {
    emit(state.copyWith(sending: true, error: () => null));
    try {
      await call();
    } catch (e) {
      emit(state.copyWith(error: () => '$e'));
    } finally {
      if (!isClosed) emit(state.copyWith(sending: false));
      await refresh();
    }
  }

  @override
  Future<void> close() {
    unwatch();
    _client.close();
    return super.close();
  }
}
