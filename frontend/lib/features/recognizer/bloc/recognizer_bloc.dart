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

  /// On-device scores update every 5 s: the last 5 s go to the 5 s model and the last 10 s
  /// (this hop + the previous one) to the 10 s model.
  static const _hopSeconds = 5;

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
  Uint8List? _prevHop;
  bool _has5s = false;
  int _detectGeneration = 0;
  Future<void> _detectChain = Future<void>.value();

  Future<void> _onMicPressed(
    RecognizerMicPressed event,
    Emitter<RecognizerState> emit,
  ) async {
    if (state.isStreaming) {
      await _stop(emit);
    } else if (!state.isBusy) {
      await _start(emit, offline: event.offline, identify: event.identify);
    }
  }

  Future<void> _start(
    Emitter<RecognizerState> emit, {
    required bool offline,
    required bool identify,
  }) async {
    emit(
      RecognizerState(status: RecognizerStatus.connecting, offline: offline),
    );

    if (offline && _detector == null) {
      emit(
        state.copyWith(
          status: RecognizerStatus.failure,
          error: () =>
              'Offline mode needs the on-device model, which this platform doesn\'t have yet.',
        ),
      );
      return;
    }

    // 1. Permission before the socket opens.
    if (!await _recorder.hasPermission()) {
      emit(
        state.copyWith(
          status: RecognizerStatus.failure,
          error: () => MicrophonePermissionDenied().toString(),
        ),
      );
      return;
    }

    // 2. Socket. One socket is one session; offline there is none.
    LiveAudioSession? session;
    if (!offline) {
      final LiveAudioSession live;
      try {
        live = await _repository.openLiveSession(identify: identify);
      } catch (e) {
        emit(
          state.copyWith(
            status: RecognizerStatus.failure,
            error: () => 'Could not reach the server: $e',
          ),
        );
        return;
      }
      session = _session = live;
      _messageSub = live.messages.listen(
        (message) => add(_ServerMessageReceived(message)),
        onError: (Object e) => add(
          _ServerMessageReceived(LiveError('Bad message from server: $e')),
        ),
      );
      unawaited(
        live.done.then((_) {
          if (identical(_session, live))
            add(_SessionClosed(code: live.closeCode, reason: live.closeReason));
        }),
      );
    }

    // 3. Microphone → socket.
    final Stream<Uint8List> pcm;
    try {
      pcm = await _recorder.startPcmStream(sampleRate: sampleRate);
    } catch (e) {
      await _teardown(closeSocket: true);
      emit(
        state.copyWith(
          status: RecognizerStatus.failure,
          error: () => 'Could not start recording: $e',
        ),
      );
      return;
    }

    emit(
      state.copyWith(
        status: RecognizerStatus.streaming,
        onDevice: OnDeviceDetection(
          status: _detector == null
              ? OnDeviceStatus.unavailable
              : OnDeviceStatus.listening,
        ),
      ),
    );
    _detectBuffer.clear();
    _detectedBytes = 0;
    _prevHop = null;
    _has5s = false;
    _detectGeneration++;
    final detector = _detector;
    if (detector != null) {
      unawaited(
        detector.info().then(
          (info) => _addIfOpen(_OnDeviceInfo(info, model: '10s')),
          onError: (Object e) => _addIfOpen(_OnDeviceFailed('$e')),
        ),
      );
      // The 5 s model is optional; without it the 10 s model runs alone.
      unawaited(
        detector.info(model: '5s').then((info) {
          if (info.compatible) _has5s = true;
          _addIfOpen(_OnDeviceInfo(info, model: '5s'));
        }, onError: (Object _) {}),
      );
    }

    final drained = _pcmDrained = Completer<void>();
    var lastProgress = DateTime.fromMillisecondsSinceEpoch(0);
    var captured = 0;
    _pcmSub = pcm.listen(
      (chunk) {
        session?.addPcm(chunk);
        captured += chunk.length;
        _bufferForDetection(chunk);
        final now = DateTime.now();
        if (now.difference(lastProgress) >= _progressInterval) {
          lastProgress = now;
          add(
            _PcmChunkSent(
              level: AudioRecorderService.pcmLevel(chunk),
              bytesSent: session?.bytesSent ?? captured,
            ),
          );
        }
      },
      onError: (Object e) =>
          add(_MicStreamFailed('Microphone stream failed: $e')),
      onDone: () {
        if (!drained.isCompleted) drained.complete();
      },
    );

    final started = DateTime.now();
    _ticker = Timer.periodic(
      _tick,
      (_) => add(_RecordingTicked(DateTime.now().difference(started))),
    );
  }

  /// Stops capture, then sends `stop` so the server flushes the last clip and
  /// pending identifications before closing.
  Future<void> _stop(Emitter<RecognizerState> emit) async {
    await _stopCapture();
    if (state.offline) {
      emit(state.copyWith(status: RecognizerStatus.finishing, level: 0));
      // Wait for the last chunk to be scored locally.
      await _detectChain;
      if (state.status == RecognizerStatus.finishing) {
        emit(state.copyWith(status: RecognizerStatus.completed));
      }
      return;
    }
    final session = _session;
    if (session == null || session.isClosed)
      return; // _SessionClosed handles it
    session.stop();
    emit(
      state.copyWith(
        status: RecognizerStatus.finishing,
        level: 0,
        bytesStreamed: session.bytesSent,
      ),
    );
    _finishTimer = Timer(finishTimeout, () => add(const _FinishTimedOut()));
  }

  void _onTicked(_RecordingTicked event, Emitter<RecognizerState> emit) {
    if (!state.isStreaming) return;
    emit(state.copyWith(elapsed: event.elapsed));
    if (event.elapsed >= maxDuration) add(const RecognizerMicPressed());
  }

  void _onPcmChunkSent(_PcmChunkSent event, Emitter<RecognizerState> emit) {
    if (state.isStreaming)
      emit(state.copyWith(level: event.level, bytesStreamed: event.bytesSent));
  }

  Future<void> _onMicStreamFailed(
    _MicStreamFailed event,
    Emitter<RecognizerState> emit,
  ) async {
    if (!state.isStreaming) return;
    await _teardown(closeSocket: true);
    emit(
      state.copyWith(
        status: RecognizerStatus.failure,
        level: 0,
        error: () => event.reason,
      ),
    );
  }

  void _onServerMessage(
    _ServerMessageReceived event,
    Emitter<RecognizerState> emit,
  ) {
    switch (event.message) {
      case final LiveSessionStarted session:
        emit(
          state.copyWith(
            session: session,
            spectrogram: Spectrogram(
              nMels: session.nMels,
              dbFloor: session.dbFloor,
              hopS: session.hopS,
            ),
          ),
        );
      case LiveSpectrum(:final t, :final frames):
        emit(state.copyWith(spectrogram: state.spectrogram.append(t, frames)));
      case final LiveClip clip:
        emit(
          state.copyWith(
            clips: _upsertClip(
              clip.index,
              clip.id,
              (c) => c.copyWith(startS: clip.startS, durationS: clip.durationS),
            ),
          ),
        );
      case final LiveClipIdentified result:
        emit(
          state.copyWith(
            clips: _upsertClip(
              result.index,
              result.id,
              (c) => c.copyWith(identification: result.identification),
            ),
            clipMetaData: _upsertClipMeta(
              result.index,
              result.id,
              (c) => c.copyWith(identification: result.identification),
            ),
          ),
        );
      case LiveError(:final message):
        // Fatal; the server closes with 1008 next and _SessionClosed reports it.
        emit(state.copyWith(error: () => message));
    }
  }

  List<ClipResult> _upsertClip(
    int index,
    String id,
    ClipResult Function(ClipResult) update,
  ) {
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

  List<ClipMetaData> _upsertClipMeta(
    int index,
    String id,
    ClipMetaData Function(ClipMetaData) update,
  ) {
    final clips = [...state.clipMetaData];
    final i = clips.indexWhere((c) => c.id == id);
    if (i >= 0) {
      clips[i] = update(clips[i]);
    } else {
      clips.add(
        update(
          ClipMetaData(
            prediction: '',
            id: id,
            predictionWindow: PredictionWindow(
              window10: '',
              window5: '',
              delta: '',
            ),
          ),
        ),
      );
    }
    clips.sort((a, b) => b.id.compareTo(a.id));
    return List.unmodifiable(clips);
  }

  void updateClipMeta(
    int index,
    String id,
    ClipMetaData metaData,
    Emitter<RecognizerState> emit,
  ) {
    final clipMeta = state.clipMetaData.firstWhere((x) {
      return x.id == id;
    });

    state.clipMetaData.removeWhere((x) => x.id == id);

    emit(
      state.copyWith(
        clipMetaData: [
          ...state.clipMetaData,
          clipMeta.copyWith(
            prediction: metaData.prediction,
            predictionWindow: metaData.predictionWindow,
            identification: metaData.identification,
          ),
        ],
      ),
    );
  }

  Future<void> _onSessionClosed(
    _SessionClosed event,
    Emitter<RecognizerState> emit,
  ) async {
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
          error: () =>
              'Connection closed$code. Clips already saved are kept — tap the mic to start a new session.',
        ),
      );
    }
  }

  Future<void> _onFinishTimedOut(
    _FinishTimedOut event,
    Emitter<RecognizerState> emit,
  ) async {
    if (state.status != RecognizerStatus.finishing) return;
    // Closing makes session.done fire, which completes via _SessionClosed.
    await _session?.close();
  }

  Future<void> _onReset(
    RecognizerResetRequested event,
    Emitter<RecognizerState> emit,
  ) async {
    await _teardown(closeSocket: true);
    emit(const RecognizerState());
  }

  Future<void> _stopCapture() async {
    _ticker?.cancel();
    _ticker = null;
    if (_pcmSub == null) return;
    await _recorder.stop();
    // Let the last chunks the plugin flushes on stop reach the socket.
    await _pcmDrained?.future.timeout(
      const Duration(seconds: 1),
      onTimeout: () {},
    );
    await _pcmSub?.cancel();
    _pcmSub = null;
    _flushDetection();
  }

  // On-device detection: the same PCM that goes to the server is cut into 5 s hops and scored
  // locally, one call at a time, without blocking the stream. Each hop gives a 5 s window to
  // the 5 s model and, from the second hop on, the last 10 s to the 10 s model.

  int get _hopBytes => sampleRate * 2 * _hopSeconds;

  void _bufferForDetection(Uint8List chunk) {
    if (_detector == null) return;
    _detectBuffer.add(chunk);
    while (_detectBuffer.length >= _hopBytes) {
      final all = _detectBuffer.takeBytes();
      _onHop(Uint8List.sublistView(all, 0, _hopBytes));
      if (all.length > _hopBytes)
        _detectBuffer.add(Uint8List.sublistView(all, _hopBytes));
    }
  }

  void _onHop(Uint8List hop) {
    final offsetS = _detectedBytes / (sampleRate * 2);
    _detectedBytes += hop.length;
    if (_has5s) _scoreChunk(hop, '5s', offsetS);
    final prev = _prevHop;
    if (prev != null)
      _scoreChunk(_concat(prev, hop), '10s', offsetS - _hopSeconds);
    _prevHop = hop;
  }

  /// Scores what's left when recording stops, if it is at least 3 s.
  void _flushDetection() {
    if (_detector == null ||
        _detectBuffer.length < sampleRate * 2 * _minDetectSeconds) {
      _detectBuffer.clear();
      return;
    }
    final rest = _detectBuffer.takeBytes();
    final offsetS = _detectedBytes / (sampleRate * 2);
    _detectedBytes += rest.length;
    if (_has5s) _scoreChunk(rest, '5s', offsetS);
    final prev = _prevHop;
    _scoreChunk(
      prev == null ? rest : _concat(prev, rest),
      '10s',
      prev == null ? offsetS : offsetS - _hopSeconds,
    );
  }

  static Uint8List _concat(Uint8List a, Uint8List b) =>
      (BytesBuilder(copy: false)
            ..add(a)
            ..add(b))
          .takeBytes();

  void _scoreChunk(Uint8List pcm, String model, double offsetS) {
    final detector = _detector!;
    final generation = _detectGeneration;
    _detectChain = _detectChain.then(
      (_) => detector
          .scorePcm16(pcm, sampleRate, model: model)
          .then(
            (r) => _addIfOpen(
              _OnDeviceScored(r, offsetS, generation, model: model),
            ),
            onError: (Object e) => _addIfOpen(_OnDeviceFailed('$e')),
          ),
    );
  }

  void _addIfOpen(RecognizerEvent event) {
    if (!isClosed) add(event);
  }

  void _onOnDeviceInfo(_OnDeviceInfo event, Emitter<RecognizerState> emit) {
    final info = event.info;
    if (event.model == '5s') {
      if (info.compatible)
        emit(
          state.copyWith(
            onDevice: state.onDevice.copyWith(modelRun5s: info.modelRun),
          ),
        );
      return;
    }
    emit(
      state.copyWith(
        onDevice: info.compatible
            ? state.onDevice.copyWith(modelRun: info.modelRun)
            : state.onDevice.copyWith(
                status: OnDeviceStatus.failed,
                modelRun: info.modelRun,
                error: () =>
                    'Model ${info.modelRun} expects features ${info.modelSpecHash}, app has ${info.featureSpecHash}',
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
        WindowScore(
          startS: w.startS + event.offsetS,
          logit: w.logit,
          bands: w.bands,
          embedding: w.embedding, // kept for Training-mode labelling
        ),
    ];
    final d = state.onDevice;
    final windows = event.model == '5s'
        ? d.windows
        : [...d.windows, ...shifted];
    final windows5s = event.model == '5s'
        ? [...d.windows5s, ...shifted]
        : d.windows5s;
    emit(
      state.copyWith(
        onDevice: d.copyWith(
          status: windows.isEmpty && windows5s.isEmpty
              ? OnDeviceStatus.listening
              : OnDeviceStatus.ready,
          windows: windows,
          windows5s: windows5s,
          inputRmsDbfs: event.result.inputRmsDbfs,
        ),
      ),
    );
  }

  void _onOnDeviceFailed(_OnDeviceFailed event, Emitter<RecognizerState> emit) {
    if (state.status == RecognizerStatus.idle) return;
    emit(
      state.copyWith(
        onDevice: state.onDevice.copyWith(
          status: OnDeviceStatus.failed,
          error: () => event.reason,
        ),
      ),
    );
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
