part of 'recognizer_bloc.dart';

sealed class RecognizerEvent extends Equatable {
  const RecognizerEvent();

  @override
  List<Object?> get props => [];
}

/// The mic button was tapped: opens a live session and starts streaming, or
/// sends `stop` and waits for the server's final messages.
final class RecognizerMicPressed extends RecognizerEvent {
  const RecognizerMicPressed();
}

/// Drop the session (closing the socket) and return to idle.
final class RecognizerResetRequested extends RecognizerEvent {
  const RecognizerResetRequested();
}

// Internal events fed back into the loop by timers, the mic and the socket.

final class _RecordingTicked extends RecognizerEvent {
  const _RecordingTicked(this.elapsed);

  final Duration elapsed;

  @override
  List<Object?> get props => [elapsed];
}

/// Throttled progress from the PCM pipe (the audio itself bypasses the loop).
final class _PcmChunkSent extends RecognizerEvent {
  const _PcmChunkSent({required this.level, required this.bytesSent});

  final double level;
  final int bytesSent;

  @override
  List<Object?> get props => [level, bytesSent];
}

final class _MicStreamFailed extends RecognizerEvent {
  const _MicStreamFailed(this.reason);

  final String reason;

  @override
  List<Object?> get props => [reason];
}

/// A `session` / `spectrum` / `clip` / `identification` / `error` message.
final class _ServerMessageReceived extends RecognizerEvent {
  const _ServerMessageReceived(this.message);

  final LiveServerMessage message;

  @override
  List<Object?> get props => [message];
}

/// The socket closed (after `stop`, after an `error`, or unexpectedly).
final class _SessionClosed extends RecognizerEvent {
  const _SessionClosed({this.code, this.reason});

  final int? code;
  final String? reason;

  @override
  List<Object?> get props => [code, reason];
}

/// The server didn't close within the finish timeout after `stop`.
final class _FinishTimedOut extends RecognizerEvent {
  const _FinishTimedOut();
}
