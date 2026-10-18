import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../../core/app_config.dart';
import '../../../data/recorder/audio_recorder_service.dart';
import '../../../data/repository/identification_repository.dart';
import '../bloc/recognizer_bloc.dart';
import '../widgets/mic_button.dart';
import '../widgets/result_card.dart';
import '../widgets/spectrogram_view.dart';

class RecognizerPage extends StatelessWidget {
  const RecognizerPage({super.key});

  @override
  Widget build(BuildContext context) {
    final config = context.read<AppConfig>();
    return BlocProvider(
      create: (context) => RecognizerBloc(
        recorder: context.read<AudioRecorderService>(),
        repository: context.read<IdentificationRepository>(),
        sampleRate: config.liveSampleRate,
        maxDuration: config.liveMaxDuration,
        finishTimeout: config.finishTimeout,
      ),
      child: const RecognizerView(),
    );
  }
}

class RecognizerView extends StatelessWidget {
  const RecognizerView({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Audio Identifier'),
        actions: [
          BlocSelector<RecognizerBloc, RecognizerState, bool>(
            selector: (s) => s.status != RecognizerStatus.idle,
            builder: (context, canReset) => IconButton(
              tooltip: 'Reset',
              icon: const Icon(Icons.refresh_rounded),
              onPressed: canReset ? () => context.read<RecognizerBloc>().add(const RecognizerResetRequested()) : null,
            ),
          ),
        ],
      ),
      body: SafeArea(
        child: BlocBuilder<RecognizerBloc, RecognizerState>(
          builder: (context, state) => Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(24),
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 520),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    MicButton(
                      recording: state.isStreaming,
                      busy: state.isBusy,
                      level: state.level,
                      progress: state.clipProgress,
                      onPressed: state.isBusy
                          ? null
                          : () => context.read<RecognizerBloc>().add(const RecognizerMicPressed()),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      _statusLine(state),
                      style: Theme.of(context).textTheme.titleMedium,
                      textAlign: TextAlign.center,
                    ),
                    if (state.error != null) ...[
                      const SizedBox(height: 12),
                      Text(
                        state.error!,
                        textAlign: TextAlign.center,
                        style: TextStyle(color: Theme.of(context).colorScheme.error),
                      ),
                    ],
                    if (state.session != null) ...[
                      const SizedBox(height: 24),
                      SpectrogramView(spectrogram: state.spectrogram),
                    ],
                    if (state.status == RecognizerStatus.completed && state.clips.isEmpty) ...[
                      const SizedBox(height: 16),
                      const Text('No clip was saved — the server only keeps clips of at least 3 s.'),
                    ],
                    for (final clip in state.clips) ...[const SizedBox(height: 12), ClipResultCard(clip: clip)],
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  static String _statusLine(RecognizerState s) {
    final t = s.elapsed.inSeconds;
    final clock = '${t ~/ 60}:${(t % 60).toString().padLeft(2, '0')}';
    return switch (s.status) {
      RecognizerStatus.idle => 'Tap the mic and play some audio',
      RecognizerStatus.connecting => 'Connecting…',
      RecognizerStatus.streaming =>
        'Streaming $clock · ${_kb(s.bytesStreamed)} · clip ${(s.elapsed.inMilliseconds / 1000) ~/ s.clipSeconds + 1}'
            ' — tap to stop',
      RecognizerStatus.finishing => 'Saving last clip and finishing identification…',
      RecognizerStatus.completed => 'Session finished — tap the mic to start a new one',
      RecognizerStatus.failure => 'Something went wrong',
    };
  }

  static String _kb(int bytes) => '${(bytes / 1024).round()} KB';
}
