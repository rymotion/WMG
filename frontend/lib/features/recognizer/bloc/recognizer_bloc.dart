import 'dart:async';
import 'dart:typed_data';

import 'package:bloc_concurrency/bloc_concurrency.dart';
import 'package:equatable/equatable.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:wmg_detector/wmg_detector.dart';

import '../../../data/api/live_audio_client.dart';
import '../../../data/models/live_message.dart';
import '../../../data/recorder/audio_recorder_service.dart';
import '../../../data/repository/identification_repository.dart';
import '../models/clip_result.dart';
import '../models/on_device_detection.dart';
import '../models/spectrogram.dart';

part 'recognizer_event.dart';
part 'recognizer_state.dart';

/// Drives one live session on `/live`: mic → PCM frames → server, and server
/// messages (`session`, `spectrum`, `clip`, `identification`, `error`) → state.
///
/// Every input — taps, the timer, mic progress, server messages, the socket
/// closing — arrives as an event, and each handler's transformer decides how
/// events of that type are scheduled. The PCM itself is piped straight from the
/// mic into the socket; only throttled progress goes through the loop.
class RecognizerBloc extends Bloc<RecognizerEvent, RecognizerState> {
  RecognizerBloc({
    required AudioRecorderService recorder,
    required IdentificationRepository repository,
    this.sampleRate = 32000,
    this.maxDuration = const Duration(minutes: 60),
    this.finishTimeout = const Duration(seconds: 20),
    WmgDetector? detector,
  }) : _recorder = recorder,
       _detector = detector,
       _repository = repository,
       super(const RecognizerState()) {
    // Taps that arrive while a connect/stop is still in flight are dropped.
    on<RecognizerMicPressed>(_onMicPressed, transformer: droppable());
    on<RecognizerResetRequested>(_onReset, transformer: sequential());
    on<_RecordingTicked>(_onTicked);
    on<_PcmChunkSent>(_onPcmChunkSent);
    on<_MicStreamFailed>(_onMicStreamFailed, transformer: sequential());
    on<_ServerMessageReceived>(_onServerMessage, transformer: sequential());
    on<_SessionClosed>(_onSessionClosed, transformer: sequential());
    on<_FinishTimedOut>(_onFinishTimedOut);
    on<_OnDeviceScored>(_onOnDeviceScored, transformer: sequential());
    on<_OnDeviceFailed>(_onOnDeviceFailed);
    on<_OnDeviceInfo>(_onOnDeviceInfo);
  }

  final AudioRecorderService _recorder;
  final IdentificationRepository _repository;

  /// Scores audio locally; null where no on-device model exists (Android, tests).
  final WmgDetector? _detector;
  final int sampleRate;

  /// Audio scored per on-device call: one 10 s detector window.
  static const _detectSeconds = 10;

  /// A trailing chunk shorter than this isn't scored (matches the 3 s partial-window rule).
  static const _minDetectSeconds = 3;

  /// The server caps sessions at `LIVE_MAX_MINUTES` (60 by default).
  final Duration maxDuration;
  final Duration finishTimeout;

  static const _tick = Duration(milliseconds: 100);
  static const _progressInterval = Duration(milliseconds: 50);

  LiveAudioSession? _session;
  StreamSubscription<LiveServerMessage>? _messageSub;
  StreamSubscription<Uint8List>? _pcmSub;
  Completer<void>? _pcmDrained;
  Timer? _ticker;
  Timer? _finishTimer;

  final _detectBuffer = BytesBuilder(copy: false);
  int _detectedBytes = 0;
  int _detectGeneration = 0;
  Future<void> _detectChain = Future<void>.value();

  Future<void> _onMicPressed(RecognizerMicPressed event, Emitter<RecognizerState> emit) async {
    if (state.isStreaming) {
      await _stop(emit);
    } else if (!state.isBusy) {
      await _start(emit);
    }
  }

  Future<void> _start(Emitter<RecognizerState> emit) async {
    emit(const RecognizerState(status: RecognizerStatus.connecting));

    // 1. Permission before the socket opens.
    if (!await _recorder.hasPermission()) {
      emit(state.copyWith(status: RecognizerStatus.failure, error: () => MicrophonePermissionDenied().toString()));
      return;
    }

    // 2. Socket. One socket is one session.
    final LiveAudioSession session;
    try {
      session = await _repository.openLiveSession();
    } catch (e) {
      emit(state.copyWith(status: RecognizerStatus.failure, error: () => 'Could not reach the server: $e'));
      return;
    }
    _session = session;
    _messageSub = session.messages.listen(
      (message) => add(_ServerMessageReceived(message)),
      onError: (Object e) => add(_ServerMessageReceived(LiveError('Bad message from server: $e'))),
    );
    unawaited(
      session.done.then((_) {
        if (identical(_session, session)) add(_SessionClosed(code: session.closeCode, reason: session.closeReason));
      }),
    );

    // 3. Microphone → socket.
    final Stream<Uint8List> pcm;
    try {
      pcm = await _recorder.startPcmStream(sampleRate: sampleRate);
    } catch (e) {
      await _teardown(closeSocket: true);
      emit(state.copyWith(status: RecognizerStatus.failure, error: () => 'Could not start recording: $e'));
      return;
    }

    emit(
      state.copyWith(
        status: RecognizerStatus.streaming,
        onDevice: OnDeviceDetection(status: _detector == null ? OnDeviceStatus.unavailable : OnDeviceStatus.listening),
      ),
    );
    _detectBuffer.clear();
    _detectedBytes = 0;
    _detectGeneration++;
    final detector = _detector;
    if (detector != null) {
      unawaited(
        detector.info().then(
          (info) => _addIfOpen(_OnDeviceInfo(info)),
          onError: (Object e) => _addIfOpen(_OnDeviceFailed('$e')),
        ),
      );
    }

    final drained = _pcmDrained = Completer<void>();
    var lastProgress = DateTime.fromMillisecondsSinceEpoch(0);
    _pcmSub = pcm.listen(
      (chunk) {
        session.addPcm(chunk);
        _bufferForDetection(chunk);
        final now = DateTime.now();
        if (now.difference(lastProgress) >= _progressInterval) {
          lastProgress = now;
          add(_PcmChunkSent(level: AudioRecorderService.pcmLevel(chunk), bytesSent: session.bytesSent));
        }
      },
      onError: (Object e) => add(_MicStreamFailed('Microphone stream failed: $e')),
      onDone: () {
        if (!drained.isCompleted) drained.complete();
      },
    );

    final started = DateTime.now();
    _ticker = Timer.periodic(_tick, (_) => add(_RecordingTicked(DateTime.now().difference(started))));
  }

  /// Stops capture, then sends `stop` so the server flushes the last clip and
  /// pending identifications before closing.
  Future<void> _stop(Emitter<RecognizerState> emit) async {
    await _stopCapture();
    final session = _session;
    if (session == null || session.isClosed) return; // _SessionClosed handles it
    session.stop();
    emit(state.copyWith(status: RecognizerStatus.finishing, level: 0, bytesStreamed: session.bytesSent));
    _finishTimer = Timer(finishTimeout, () => add(const _FinishTimedOut()));
  }

  void _onTicked(_RecordingTicked event, Emitter<RecognizerState> emit) {
    if (!state.isStreaming) return;
    emit(state.copyWith(elapsed: event.elapsed));
    if (event.elapsed >= maxDuration) add(const RecognizerMicPressed());
  }

  void _onPcmChunkSent(_PcmChunkSent event, Emitter<RecognizerState> emit) {
    if (state.isStreaming) emit(state.copyWith(level: event.level, bytesStreamed: event.bytesSent));
  }

  Future<void> _onMicStreamFailed(_MicStreamFailed event, Emitter<RecognizerState> emit) async {
    if (!state.isStreaming) return;
    await _teardown(closeSocket: true);
    emit(state.copyWith(status: RecognizerStatus.failure, level: 0, error: () => event.reason));
  }

  void _onServerMessage(_ServerMessageReceived event, Emitter<RecognizerState> emit) {
    switch (event.message) {
      case final LiveSessionStarted session:
        emit(
          state.copyWith(
            session: session,
            spectrogram: Spectrogram(nMels: session.nMels, dbFloor: session.dbFloor),
          ),
        );
      case LiveSpectrum(:final frames):
        emit(state.copyWith(spectrogram: state.spectrogram.append(frames)));
      case final LiveClip clip:
        emit(
          state.copyWith(
            clips: _upsertClip(clip.index, clip.id, (c) => c.copyWith(startS: clip.startS, durationS: clip.durationS)),
          ),
        );
      case final LiveClipIdentified result:
        emit(
          state.copyWith(
            clips: _upsertClip(result.index, result.id, (c) => c.copyWith(identification: result.identification)),
          ),
        );
      case LiveError(:final message):
        // Fatal; the server closes with 1008 next and _SessionClosed reports it.
        emit(state.copyWith(error: () => message));
    }
  }

  List<ClipResult> _upsertClip(int index, String id, ClipResult Function(ClipResult) update) {
    final clips = [...state.clips];
    final i = clips.indexWhere((c) => c.index == index);
    if (i >= 0) {
      clips[i] = update(clips[i]);
    } else {
      clips.add(update(ClipResult(index: index, id: id)));
    }
    clips.sort((a, b) => b.index.compareTo(a.index));
    return List.unmodifiable(clips);
  }

  Future<void> _onSessionClosed(_SessionClosed event, Emitter<RecognizerState> emit) async {
    final wasFinishing = state.status == RecognizerStatus.finishing;
    await _teardown(closeSocket: false);

    if (state.error != null) {
      emit(state.copyWith(status: RecognizerStatus.failure, level: 0));
    } else if (wasFinishing) {
      emit(state.copyWith(status: RecognizerStatus.completed, level: 0));
    } else {
      final code = event.code == null ? '' : ' (code ${event.code})';
      emit(
        state.copyWith(
          status: RecognizerStatus.failure,
          level: 0,
          error: () => 'Connection closed$code. Clips already saved are kept — tap the mic to start a new session.',
        ),
      );
    }
  }

  Future<void> _onFinishTimedOut(_FinishTimedOut event, Emitter<RecognizerState> emit) async {
    if (state.status != RecognizerStatus.finishing) return;
    // Closing makes session.done fire, which completes via _SessionClosed.
    await _session?.close();
  }

  Future<void> _onReset(RecognizerResetRequested event, Emitter<RecognizerState> emit) async {
    await _teardown(closeSocket: true);
    emit(const RecognizerState());
  }

  Future<void> _stopCapture() async {
    _ticker?.cancel();
    _ticker = null;
    if (_pcmSub == null) return;
    await _recorder.stop();
    // Let the last chunks the plugin flushes on stop reach the socket.
    await _pcmDrained?.future.timeout(const Duration(seconds: 1), onTimeout: () {});
    await _pcmSub?.cancel();
    _pcmSub = null;
    _flushDetection();
  }

  // On-device detection: the same PCM that goes to the server is cut into 10 s chunks and
  // scored locally, one call at a time, without blocking the stream.

  void _bufferForDetection(Uint8List chunk) {
    if (_detector == null) return;
    _detectBuffer.add(chunk);
    final chunkBytes = sampleRate * 2 * _detectSeconds;
    while (_detectBuffer.length >= chunkBytes) {
      final all = _detectBuffer.takeBytes();
      _scoreChunk(Uint8List.sublistView(all, 0, chunkBytes));
      if (all.length > chunkBytes) _detectBuffer.add(Uint8List.sublistView(all, chunkBytes));
    }
  }

  void _flushDetection() {
    if (_detector == null || _detectBuffer.length < sampleRate * 2 * _minDetectSeconds) {
      _detectBuffer.clear();
      return;
    }
    _scoreChunk(_detectBuffer.takeBytes());
  }

  void _scoreChunk(Uint8List pcm) {
    final detector = _detector!;
    final offsetS = _detectedBytes / (sampleRate * 2);
    final generation = _detectGeneration;
    _detectedBytes += pcm.length;
    _detectChain = _detectChain.then(
      (_) => detector.scorePcm16(pcm, sampleRate).then(
        (r) => _addIfOpen(_OnDeviceScored(r, offsetS, generation)),
        onError: (Object e) => _addIfOpen(_OnDeviceFailed('$e')),
      ),
    );
  }

  void _addIfOpen(RecognizerEvent event) {
    if (!isClosed) add(event);
  }

  void _onOnDeviceInfo(_OnDeviceInfo event, Emitter<RecognizerState> emit) {
    final info = event.info;
    emit(
      state.copyWith(
        onDevice: info.compatible
            ? state.onDevice.copyWith(modelRun: info.modelRun)
            : state.onDevice.copyWith(
                status: OnDeviceStatus.failed,
                modelRun: info.modelRun,
                error: () => 'Model ${info.modelRun} expects features ${info.modelSpecHash}, app has ${info.featureSpecHash}',
              ),
      ),
    );
  }

  void _onOnDeviceScored(_OnDeviceScored event, Emitter<RecognizerState> emit) {
    if (event.generation != _detectGeneration ||
        state.status == RecognizerStatus.idle ||
        state.onDevice.status == OnDeviceStatus.failed) {
      return;
    }
    final shifted = [
      for (final w in event.result.windows)
        WindowScore(startS: w.startS + event.offsetS, logit: w.logit, bands: w.bands),
    ];
    emit(
      state.copyWith(
        onDevice: state.onDevice.copyWith(
          status: state.onDevice.windows.isEmpty && shifted.isEmpty ? OnDeviceStatus.listening : OnDeviceStatus.ready,
          windows: [...state.onDevice.windows, ...shifted],
          inputRmsDbfs: event.result.inputRmsDbfs,
        ),
      ),
    );
  }

  void _onOnDeviceFailed(_OnDeviceFailed event, Emitter<RecognizerState> emit) {
    if (state.status == RecognizerStatus.idle) return;
    emit(state.copyWith(onDevice: state.onDevice.copyWith(status: OnDeviceStatus.failed, error: () => event.reason)));
  }

  /// Stops the mic and forgets the session; optionally closes the socket.
  Future<void> _teardown({required bool closeSocket}) async {
    _finishTimer?.cancel();
    _finishTimer = null;
    await _stopCapture();
    final session = _session;
    _session = null;
    await _messageSub?.cancel();
    _messageSub = null;
    if (closeSocket) await session?.close();
  }

  @override
  Future<void> close() async {
    await _teardown(closeSocket: true);
    return super.close();
  }
}
