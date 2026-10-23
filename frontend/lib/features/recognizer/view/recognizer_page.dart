import 'dart:io' show Platform;

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:wmg_detector/wmg_detector.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../../core/app_config.dart';
import '../../../data/recorder/audio_recorder_service.dart';
import '../../../data/repository/identification_repository.dart';
import '../../../data/api/spectro_api_client.dart';
import '../../../data/personalization/personalization_store.dart';
import '../../personalization/cubit/personalization_cubit.dart';
import '../../personalization/view/personalization_page.dart';
import '../bloc/recognizer_bloc.dart';
import '../models/on_device_detection.dart';
import '../widgets/mic_button.dart';
import '../widgets/on_device_card.dart';
import '../widgets/result_card.dart';
import '../widgets/spectrogram_view.dart';

class RecognizerPage extends StatelessWidget {
  const RecognizerPage({super.key});

  @override
  Widget build(BuildContext context) {
    final config = context.read<AppConfig>();
    // Core ML + the Rust feature library ship for iOS and macOS so far.
    const detector = WmgDetector();
    final hasDetector = !kIsWeb && (Platform.isIOS || Platform.isMacOS);
    return MultiBlocProvider(
      providers: [
        BlocProvider(
          create: (context) => PersonalizationCubit(
            store: PersonalizationStore(),
            detector: hasDetector ? detector : null,
            api: context.read<SpectroApiClient>(),
          )..load(),
        ),
        BlocProvider(
          create: (context) => RecognizerBloc(
            recorder: context.read<AudioRecorderService>(),
            repository: context.read<IdentificationRepository>(),
            sampleRate: config.liveSampleRate,
            maxDuration: config.liveMaxDuration,
            finishTimeout: config.finishTimeout,
            detector: hasDetector ? detector : null,
          ),
        ),
      ],
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
          IconButton(
            tooltip: 'On-device training',
            icon: const Icon(Icons.tune_rounded),
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute<void>(
                builder: (_) => BlocProvider.value(
                  value: context.read<PersonalizationCubit>(),
                  child: const PersonalizationPage(),
                ),
              ),
            ),
          ),
          BlocSelector<RecognizerBloc, RecognizerState, bool>(
            selector: (s) => s.status != RecognizerStatus.idle,
            builder: (context, canReset) => IconButton(
              tooltip: 'Reset',
              icon: const Icon(Icons.refresh_rounded),
              onPressed: canReset
                  ? () => context.read<RecognizerBloc>().add(
                      const RecognizerResetRequested(),
                    )
                  : null,
            ),
          ),
        ],
      ),
      body: SafeArea(
        child: BlocBuilder<RecognizerBloc, RecognizerState>(
          builder: (context, state) => Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  ConstrainedBox(
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
                              : () => context.read<RecognizerBloc>().add(
                                  const RecognizerMicPressed(),
                                ),
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
                            style: TextStyle(
                              color: Theme.of(context).colorScheme.error,
                            ),
                          ),
                        ],
                        if (state.session != null) ...[
                          const SizedBox(height: 24),
                          SpectrogramView(spectrogram: state.spectrogram),
                        ],
                        if (state.status != RecognizerStatus.idle &&
                            state.onDevice.status != OnDeviceStatus.unavailable)
                          BlocBuilder<
                            PersonalizationCubit,
                            PersonalizationState
                          >(
                            builder: (context, p) {
                              final head = p.activeHead;
                              final recordingId = state.session?.sessionId;
                              final canLabel =
                                  p.trainingMode &&
                                  state.status == RecognizerStatus.completed &&
                                  state.onDevice.hasScore &&
                                  recordingId != null;
                              return Column(
                                children: [
                                  const SizedBox(height: 16),
                                  OnDeviceCard(
                                    detection: head == null
                                        ? state.onDevice
                                        : state.onDevice.personalised(head),
                                    personalisedId: head?.id,
                                  ),
                                  if (canLabel) ...[
                                    const SizedBox(height: 8),
                                    _LabelCard(
                                      label: p.labelled[recordingId],
                                      windows: state.onDevice.windows.length,
                                      onLabel: (l) => context
                                          .read<PersonalizationCubit>()
                                          .labelRecording(
                                            recordingId,
                                            l,
                                            state.onDevice.windows,
                                          ),
                                    ),
                                  ],
                                ],
                              );
                            },
                          ),
                        if (state.status == RecognizerStatus.completed &&
                            state.clips.isEmpty) ...[
                          const SizedBox(height: 16),
                          const Text(
                            'No clip was saved — the server only keeps clips of at least 3 s.',
                          ),
                        ],
                      ],
                    ),
                  ),
                  if (state.clips.isNotEmpty) ...[
                    const SizedBox(height: 16),
                    _buildClipResults(context, state),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  /// Clip results laid out as a grid whose column count scales with the
  /// available width (up to [_maxClipColumns]). Rows size to their content
  /// so cards with several matches expand instead of overflowing.
  Widget _buildClipResults(BuildContext context, RecognizerState state) {
    const spacing = 8.0;
    return LayoutBuilder(
      builder: (context, constraints) {
        final width = constraints.maxWidth;
        final columns = (width / _minClipCardWidth).floor().clamp(
          1,
          _maxClipColumns,
        );
        final cardWidth = (width - spacing * (columns - 1)) / columns;
        return Wrap(
          spacing: spacing,
          runSpacing: spacing,
          children: [
            for (final clip in state.clips)
              SizedBox(
                width: cardWidth,
                child: ClipResultCard(context: context, clip: clip),
              ),
          ],
        );
      },
    );
  }

  static const _maxClipColumns = 3;
  static const _minClipCardWidth = 320.0;

  static String _statusLine(RecognizerState s) {
    final t = s.elapsed.inSeconds;
    final clock = '${t ~/ 60}:${(t % 60).toString().padLeft(2, '0')}';
    return switch (s.status) {
      RecognizerStatus.idle => 'Tap the mic and play some audio',
      RecognizerStatus.connecting => 'Connecting…',
      RecognizerStatus.streaming =>
        'Streaming $clock · ${_kb(s.bytesStreamed)} · clip ${(s.elapsed.inMilliseconds / 1000) ~/ s.clipSeconds + 1}'
            ' — tap to stop',
      RecognizerStatus.finishing =>
        'Saving last clip and finishing identification…',
      RecognizerStatus.completed =>
        'Session finished — tap the mic to start a new one',
      RecognizerStatus.failure => 'Something went wrong',
    };
  }

  static String _kb(int bytes) => '${(bytes / 1024).round()} KB';
}

/// Training mode: tag the finished recording as human or AI for on-device retraining.
class _LabelCard extends StatelessWidget {
  const _LabelCard({
    required this.label,
    required this.windows,
    required this.onLabel,
  });

  final int? label;
  final int windows;
  final ValueChanged<int> onLabel;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      color: theme.colorScheme.secondaryContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: label != null
            ? Row(
                children: [
                  const Icon(Icons.check_circle_rounded),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Saved $windows windows as ${label == 1 ? 'AI' : 'human'} for training.',
                    ),
                  ),
                ],
              )
            : Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Training mode: what was playing?',
                    style: theme.textTheme.titleSmall,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Labels $windows windows (embeddings only, no audio) for on-device retraining.',
                    style: theme.textTheme.bodySmall,
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: FilledButton.tonalIcon(
                          icon: const Icon(Icons.person_rounded),
                          label: const Text('Human-made'),
                          onPressed: () => onLabel(0),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: FilledButton.tonalIcon(
                          icon: const Icon(Icons.smart_toy_rounded),
                          label: const Text('AI-generated'),
                          onPressed: () => onLabel(1),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
      ),
    );
  }
}
