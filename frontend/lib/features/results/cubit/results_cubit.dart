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
