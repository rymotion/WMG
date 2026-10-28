import 'dart:io' show Platform;

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:wmg_detector/wmg_detector.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../../core/app_config.dart';
import '../../../data/recorder/audio_recorder_service.dart';
import '../../../data/repository/identification_repository.dart';
import '../../../data/api/spectro_api_client.dart';
import '../../../data/personalization/personal_model.dart' show PersonalHead;
import '../../../data/personalization/personalization_store.dart';
import '../../personalization/cubit/personalization_cubit.dart';
import '../../personalization/view/personalization_page.dart';
import '../../../data/retrain/retrain_client.dart';
import '../../retrain/cubit/retrain_cubit.dart';
import '../../../data/labels/recording_label.dart';
import '../bloc/recognizer_bloc.dart';
import '../models/clip_result.dart';
import '../models/on_device_detection.dart';
import '../widgets/mic_button.dart';
import '../widgets/on_device_card.dart';
import '../../results/data/session_history_store.dart';
import '../../results/view/results_page.dart';
import '../widgets/clip_results_grid.dart';
import '../widgets/labelable_clip_grid.dart';
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
            history: context.read<SessionHistoryStore>(),
          ),
        ),
      ],
      child: const RecognizerView(),
    );
  }
}

class RecognizerView extends StatefulWidget {
  const RecognizerView({super.key});

  @override
  State<RecognizerView> createState() => _RecognizerViewState();
}

class _RecognizerViewState extends State<RecognizerView> {
  final _spectrogram = SpectrogramController();
  final _spectrogramKey = GlobalKey();

  /// [ClipResult.index] of the clip highlighted on the spectrogram.
  int? _selectedClip;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Audio Detector'),
        actions: [
          IconButton(
            tooltip: 'Saved sessions',
            icon: const Icon(Icons.history_rounded),
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute<void>(builder: (_) => const ResultsPage()),
            ),
          ),
          IconButton(
            tooltip: 'On-device training',
            icon: const Icon(Icons.tune_rounded),
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute<void>(
                builder: (_) => MultiBlocProvider(
                  providers: [
                    BlocProvider.value(value: context.read<PersonalizationCubit>()),
                    BlocProvider(
                      create: (_) => RetrainCubit(
                        client: RetrainClient(baseUrl: context.read<AppConfig>().trainerBaseUrl),
                        history: context.read<SessionHistoryStore>(),
                      ),
                    ),
                  ],
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
      body: BlocBuilder<RecognizerBloc, RecognizerState>(
        // The mic level and clock change ~20×/s; only the mic row rebuilds for those.
        buildWhen: (a, b) => !_onlyProgressChanged(a, b),
        builder: (context, state) => Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 520),
                  child: BlocBuilder<RecognizerBloc, RecognizerState>(
                    builder: (context, state) => recordingFeedback(state),
                  ),
                ),
                _buildSpectrogram(state),
                _buildError(state),
                ...buildEstimateResults(state),
                if (state.clips.isNotEmpty) ...[
                  const SizedBox(height: 16),
                  _buildClipResults(context, state),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildClipResults(BuildContext context, RecognizerState state) {
    return LabelableClipGrid(
      clips: state.clips,
      detection: _detection(
        state,
        context.select((PersonalizationCubit c) => c.state.activeHead),
      ),
      selectedIndex: _selectedClip,
      onTap: _showClip,
      recordingLabel: state.label,
      onLabel: (indexes, c) => context.read<RecognizerBloc>().add(
        RecognizerClipLabelled(indexes, c == null ? null : RecordingLabel.now(c)),
      ),
    );
  }

  Widget recordingFeedback(RecognizerState state) {
    // No Expanded here: it's only valid directly inside a Row/Column/Flex, and
    // this sits in a ConstrainedBox. That box's maxWidth bounds the Flexible.
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        MicButton(
          recording: state.isStreaming,
          busy: state.isBusy,
          level: state.level,
          progress: state.clipProgress,
          onPressed: state.isBusy ? null : () => _onMicPressed(context),
        ),

        Flexible(
          child: Text(
            _statusLine(state),
            softWrap: true,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w600,
              overflow: TextOverflow.fade,
            ),
            textAlign: TextAlign.center,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
          ),
        ),
      ],
    );
  }

  Widget _buildError(RecognizerState state) {
    if (state.error == null) return const SizedBox.shrink();
    return Column(
      children: [
        const SizedBox(height: 12),
        Text(
          state.error!,
          textAlign: TextAlign.center,
          style: TextStyle(color: Theme.of(context).colorScheme.error),
        ),
      ],
    );
  }

  Widget _buildSpectrogram(RecognizerState state) {
    // Offline sessions have no server spectrogram but still get the P(AI) chart.
    if (state.session == null && !state.onDevice.hasScore) {
      return const SizedBox.shrink();
    }
    return BlocSelector<
      PersonalizationCubit,
      PersonalizationState,
      PersonalHead?
    >(
      selector: (p) => p.activeHead,
      builder: (context, head) {
        // Same scores as the OnDeviceCard headline.
        final d = head == null
            ? state.onDevice
            : state.onDevice.personalised(head);
        final colors = Theme.of(context).colorScheme;
        return SpectrogramView(
          key: _spectrogramKey,
          spectrogram: state.spectrogram,
          controller: _spectrogram,
          regions: [for (final c in state.clips) ?_region(c)],
          highlight: _selectedRegion(state),
          scores: scoreSeries(d, colors),
          label: state.label?.category,
          width: MediaQuery.of(context).size.width,
        );
      },
    );
  }

  List<Widget> buildEstimateResults(RecognizerState state) {
    return [
      if (state.status != RecognizerStatus.idle &&
          state.onDevice.status != OnDeviceStatus.unavailable)
        BlocBuilder<PersonalizationCubit, PersonalizationState>(
          builder: (context, p) {
            final head = p.activeHead;
            // Server session id, or the local recording id when offline.
            final recordingId = state.session?.sessionId ?? state.recordingId;
            final canLabel =
                state.status == RecognizerStatus.completed &&
                state.onDevice.hasScore;
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
                    label: state.label?.category,
                    trainingMode: p.trainingMode && recordingId != null,
                    windows: state.onDevice.windows.length,
                    onLabel: (category) {
                      context.read<RecognizerBloc>().add(
                        RecognizerLabelled(RecordingLabel.now(category)),
                      );
                      if (p.trainingMode && recordingId != null) {
                        context.read<PersonalizationCubit>().labelRecording(
                          recordingId,
                          category,
                          state.onDevice.windows,
                        );
                      }
                    },
                  ),
                ],
              ],
            );
          },
        ),
      if (state.status == RecognizerStatus.completed &&
          !state.offline &&
          state.clips.isEmpty) ...[
        const SizedBox(height: 16),
        const Text(
          'No clip was saved — the server only keeps clips of at least 3 s.',
        ),
      ],
    ];
  }

  static bool _onlyProgressChanged(RecognizerState a, RecognizerState b) =>
      a.copyWith(
        level: b.level,
        elapsed: b.elapsed,
        bytesStreamed: b.bytesStreamed,
      ) ==
      b;

  /// Starts a session with the current settings, or stops the running one.
  void _onMicPressed(BuildContext context) {
    final settings = context.read<PersonalizationCubit>().state;
    context.read<RecognizerBloc>().add(
      RecognizerMicPressed(
        offline: settings.offlineMode,
        identify: settings.identifyMode,
      ),
    );
  }

  /// Highlights [clip] on the spectrogram and scrolls the page back up to it.
  void _showClip(ClipResult clip) {
    setState(() => _selectedClip = clip.index);
    _spectrogram.seekTo(clip.startS!);
    final target = _spectrogramKey.currentContext;
    if (target != null) {
      Scrollable.ensureVisible(
        target,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOutCubic,
        alignmentPolicy: ScrollPositionAlignmentPolicy.keepVisibleAtStart,
      );
    }
  }

  SpectrogramRegion? _selectedRegion(RecognizerState state) {
    for (final c in state.clips) {
      if (c.index == _selectedClip) return _region(c);
    }
    // Cleared by a reset or new session.
    return null;
  }

  /// On-device scores as the OnDeviceCard headline shows them (personalised
  /// when a fine-tuned head is active); null where there's no detector.
  static OnDeviceDetection? _detection(
    RecognizerState state,
    PersonalHead? head,
  ) {
    final d = state.onDevice;
    if (d.status == OnDeviceStatus.unavailable) return null;
    return head == null ? d : d.personalised(head);
  }

  static SpectrogramRegion? _region(ClipResult clip) =>
      ClipResultsGrid.regionOf(clip);

  static String _statusLine(RecognizerState s) {
    final t = s.elapsed.inSeconds;
    final clock = '${t ~/ 60}:${(t % 60).toString().padLeft(2, '0')}';
    return switch (s.status) {
      RecognizerStatus.idle => 'Tap the mic and play some audio',
      RecognizerStatus.connecting => 'Connecting…',
      RecognizerStatus.streaming =>
        'Streaming $clock · ${_kb(s.bytesStreamed)} · clip ${(s.elapsed.inMilliseconds / 1000) ~/ s.clipSeconds + 1}'
            ' — tap to stop',
      RecognizerStatus.finishing when s.offline =>
        'Scoring the last few seconds on device…',
      RecognizerStatus.finishing =>
        'Saving last clip and finishing identification…',
      RecognizerStatus.completed =>
        'Session finished — tap the mic to start a new one',
      RecognizerStatus.failure => 'Something went wrong',
    };
  }

  static String _kb(int bytes) => '${(bytes / 1024).round()} KB';
}

/// After a recording: label it Human-made / AI-assisted / AI-generated. The label is saved
/// with the session (and its audio); in Training mode the recording's windows are also kept
/// for on-device retraining (AI-assisted ones are stored but not used by Retrain yet).
class _LabelCard extends StatelessWidget {
  const _LabelCard({
    required this.label,
    required this.trainingMode,
    required this.windows,
    required this.onLabel,
  });

  final AiCategory? label;
  final bool trainingMode;
  final int windows;
  final ValueChanged<AiCategory> onLabel;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      color: theme.colorScheme.secondaryContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('What was playing?', style: theme.textTheme.titleSmall),
            const SizedBox(height: 4),
            Text(
              label == null
                  ? 'Label this recording; it\'s saved with the session and its audio'
                        '${trainingMode ? ', and its $windows windows are kept for on-device training' : ''}.'
                  : 'Labelled ${label!.displayName} '
                        '(about ${(label!.aiShareMin * 100).round()}–${(label!.aiShareMax * 100).round()}% AI). '
                        'Tap another to change it.',
              style: theme.textTheme.bodySmall,
            ),
            const SizedBox(height: 12),
            SegmentedButton<AiCategory>(
              segments: [
                for (final c in AiCategory.values)
                  ButtonSegment(value: c, icon: Icon(c.icon), label: Text(c.displayName)),
              ],
              selected: {if (label != null) label!},
              emptySelectionAllowed: true,
              showSelectedIcon: false,
              onSelectionChanged: (sel) {
                if (sel.isNotEmpty) onLabel(sel.first);
              },
            ),
          ],
        ),
      ),
    );
  }
}
