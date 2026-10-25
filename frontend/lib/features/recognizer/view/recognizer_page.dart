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
import '../bloc/recognizer_bloc.dart';
import '../models/clip_result.dart';
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
      body: BlocBuilder<RecognizerBloc, RecognizerState>(
        builder: (context, state) => Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 520),
                  child: recordingFeedback(state),
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
        final detection = _detection(
          state,
          context.select((PersonalizationCubit c) => c.state.activeHead),
        );
        return Wrap(
          spacing: spacing,
          runSpacing: spacing,
          children: [
            for (final clip in state.clips)
              SizedBox(
                width: cardWidth,
                child: ClipResultCard(
                  context: context,
                  clip: clip,
                  selected: clip.index == _selectedClip,
                  onTap: _region(clip) == null ? null : () => _showClip(clip),
                  detection: switch ((detection, _region(clip))) {
                    (final d?, final r?) => d.slice(r.startS, r.endS),
                    _ => null,
                  },
                ),
              ),
          ],
        );
      },
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
                        .labelRecording(recordingId, l, state.onDevice.windows),
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

  static SpectrogramRegion? _region(ClipResult clip) {
    final start = clip.startS;
    final duration = clip.durationS;
    if (start == null || duration == null) return null;
    return (startS: start, endS: start + duration);
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
