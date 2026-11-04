import 'package:equatable/equatable.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../data/session_history_store.dart';
import '../models/saved_session.dart';

/// The list of saved sessions: load, refresh after a session closes, delete.
class ResultsCubit extends Cubit<ResultsState> {
  ResultsCubit({required SessionHistoryStore store})
    : _store = store,
      super(const ResultsState());

  final SessionHistoryStore _store;

  Future<void> load() async {
    try {
      emit(ResultsState(loaded: true, sessions: await _store.list()));
    } catch (e) {
      emit(
        ResultsState(loaded: true, error: 'Could not read saved sessions: $e'),
      );
    }
  }

  Future<void> delete(SavedSession session) async {
    emit(
      ResultsState(
        loaded: true,
        sessions: [...state.sessions]..remove(session),
      ),
    );
    await _store.delete(session.id);
  }
}

final class ResultsState extends Equatable {
  const ResultsState({
    this.loaded = false,
    this.sessions = const [],
    this.error,
  });

  final bool loaded;

  /// Newest first.
  final List<SavedSession> sessions;
  final String? error;

  @override
  List<Object?> get props => [loaded, sessions, error];
}

/// What a plan shows of [sessions] (newest first): those started within [retention]
/// (null: all) that match [query] by song, label or date. [hidden] counts the ones
/// past retention; they stay on the device and come back on upgrade.
({List<SavedSession> shown, int hidden}) visibleSessions(
  List<SavedSession> sessions, {
  Duration? retention,
  String query = '',
  DateTime? now,
}) {
  final cutoff = retention == null
      ? null
      : (now ?? DateTime.now()).subtract(retention);
  final kept = cutoff == null
      ? sessions
      : sessions.where((s) => s.startedAt.isAfter(cutoff)).toList();
  final q = query.trim().toLowerCase();
  final shown = q.isEmpty
      ? kept
      : kept
            .where(
              (s) => [
                ...s.matchedTitles,
                ?s.label?.category.displayName,
                s.startedAt.toIso8601String(),
              ].any((text) => text.toLowerCase().contains(q)),
            )
            .toList();
  return (shown: shown, hidden: sessions.length - kept.length);
}
