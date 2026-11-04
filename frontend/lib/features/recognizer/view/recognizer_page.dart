import 'dart:io' show Platform;

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:wmg_detector/wmg_detector.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../../core/app_config.dart';
import '../../../data/account/account.dart';
import '../../../data/models/identification.dart';
import '../../../data/usage/network_meter.dart';
import '../../account/cubit/account_cubit.dart';
import '../../account/view/account_page.dart';
import '../../account/view/paywall_sheet.dart';
import '../../account/widgets/usage_chip.dart';
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
      child: RecognizerView(hasDetector: hasDetector),
    );
  }
}

class RecognizerView extends StatefulWidget {
  const RecognizerView({super.key, this.hasDetector = false});

  /// The on-device AI model runs here, so a tap can always at least check for AI.
  final bool hasDetector;

  @override
  State<RecognizerView> createState() => _RecognizerViewState();
}

class _RecognizerViewState extends State<RecognizerView> {
  final _spectrogram = SpectrogramController();
  final _spectrogramKey = GlobalKey();

  /// [ClipResult.index] of the clip highlighted on the spectrogram.
  int? _selectedClip;

  /// The identification the running session was paid with, until it's settled.
  IdTicket? _ticket;

  /// Waiting on the server to reserve an ID; further taps are ignored.
  bool _planning = false;

  late final _config = context.read<AppConfig>();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Audio Detector'),
        actions: [
          if (_config.billingEnabled)
            IconButton(
              tooltip: 'Account',
              icon: const Icon(Icons.person_rounded),
              onPressed: () => Navigator.of(context).push(AccountPage.route()),
            ),
          IconButton(
            tooltip: 'Saved sessions',
            icon: const Icon(Icons.history_rounded),
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute<void>(builder: (_) => const ResultsPage()),
            ),
          ),
          if (_config.devTools)
            IconButton(
              tooltip: 'On-device training',
              icon: const Icon(Icons.tune_rounded),
              onPressed: () => Navigator.of(context).push(
                MaterialPageRoute<void>(
                  builder: (_) => MultiBlocProvider(
                    providers: [
                      BlocProvider.value(
                        value: context.read<PersonalizationCubit>(),
                      ),
                      BlocProvider(
                        create: (_) => RetrainCubit(
                          client: RetrainClient(
                            baseUrl: context.read<AppConfig>().trainerBaseUrl,
                          ),
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
      body: BlocConsumer<RecognizerBloc, RecognizerState>(
        listenWhen: (a, b) => _sessionEnded(a.status, b.status),
        listener: _onSessionEnded,
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
                if (_config.billingEnabled) ...[
                  const SizedBox(height: 12),
                  const UsageChip(),
                ],
                if (_config.devTools) _buildSpectrogram(state),
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
    if (!_config.devTools) {
      final head = context.select(
        (PersonalizationCubit c) => c.state.activeHead,
      );
      return ClipResultsGrid(
        clips: state.clips,
        detection: _detection(state, head),
      );
    }
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
        RecognizerClipLabelled(
          indexes,
          c == null ? null : RecordingLabel.now(c),
        ),
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
          busy: state.isBusy || _planning,
          level: state.level,
          progress: state.clipProgress,
          onPressed: state.isBusy || _planning
              ? null
              : () => _onMicPressed(context),
        ),

        Flexible(
          child: Text(
            _statusLine(state, detailed: _config.devTools),
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
                if (canLabel && _config.devTools) ...[
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
  Future<void> _onMicPressed(BuildContext context) async {
    final recognizer = context.read<RecognizerBloc>();
    if (_config.billingEnabled && !recognizer.state.isStreaming)
      return _startMetered(context);
    final settings = context.read<PersonalizationCubit>().state;
    context.read<RecognizerBloc>().add(
      RecognizerMicPressed(
        offline: settings.offlineMode,
        identify: settings.identifyMode,
      ),
    );
  }

  /// One tap = one ID: reserve it, then listen for the plan's window and stop at the first
  /// match. Without an ID left (or an account), the tap still checks for AI on device.
  Future<void> _startMetered(BuildContext context) async {
    final account = context.read<AccountCubit>();
    final recognizer = context.read<RecognizerBloc>();
    final window = account.state.entitlements.listenWindow;
    setState(() => _planning = true);
    final plan = await account.planListen(hasDetector: widget.hasDetector);
    if (!context.mounted) return;
    setState(() => _planning = false);
    switch (plan) {
      case IdentifyPlan(:final ticket):
        _ticket = ticket;
        recognizer.add(
          RecognizerMicPressed(
            ticket: ticket.id,
            listenFor: ticket.listenWindow,
            stopOnMatch: true,
          ),
        );
      case AiOnlyPlan(:final reason):
        recognizer.add(RecognizerMicPressed(offline: true, listenFor: window));
        _explainAiOnly(context, reason);
      case BlockedPlan(reason: UpsellReason.signedOut):
        await showSignInSheet(context);
      case BlockedPlan(reason: UpsellReason.quotaExhausted):
        await showPaywall(context, reason: UpsellReason.quotaExhausted);
      case BlockedPlan(reason: UpsellReason.unreachable):
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
              'Can\'t reach the server. Check your connection and try again.',
            ),
          ),
        );
    }
  }

  static void _explainAiOnly(BuildContext context, UpsellReason reason) {
    final (text, label, action) = switch (reason) {
      UpsellReason.signedOut => (
        'Checking for AI on this device. Sign in to name the song too.',
        'Sign in',
        () => showSignInSheet(context),
      ),
      UpsellReason.quotaExhausted => (
        'Out of song IDs this month, so this is an AI check only.',
        'Get more',
        () => showPaywall(context, reason: UpsellReason.quotaExhausted),
      ),
      UpsellReason.unreachable => (
        'Offline: checking for AI on this device.',
        null,
        null,
      ),
    };
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(text),
        action: label == null
            ? null
            : SnackBarAction(label: label, onPressed: action!),
      ),
    );
  }

  static bool _isEnd(RecognizerStatus s) =>
      s == RecognizerStatus.completed ||
      s == RecognizerStatus.failure ||
      s == RecognizerStatus.idle;

  /// A session that was running has completed, failed or been reset.
  static bool _sessionEnded(RecognizerStatus from, RecognizerStatus to) =>
      !_isEnd(from) && _isEnd(to);

  /// Settles the ticket (refunded when nothing matched) and meters the stream.
  void _onSessionEnded(BuildContext context, RecognizerState state) {
    final failed = state.status != RecognizerStatus.completed;
    if (!state.offline) {
      context.read<NetworkMeter>().recordLive(
        bytesUp: state.bytesStreamed,
        failed: failed,
      );
    }
    final ticket = _ticket;
    _ticket = null;
    if (ticket == null) return;
    final matched = state.clips.any(
      (c) => c.identification?.status == IdentificationStatus.matched,
    );
    context.read<AccountCubit>().finishIdentification(
      ticket,
      matched: matched,
      failed: failed,
    );
    if (!matched && state.status == RecognizerStatus.completed) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            'No match this time, and it didn\'t count. Try closer to the speaker.',
          ),
        ),
      );
    }
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

  /// [detailed] adds the byte and clip counters (research builds).
  static String _statusLine(RecognizerState s, {bool detailed = true}) {
    final t = s.elapsed.inSeconds;
    final clock = '${t ~/ 60}:${(t % 60).toString().padLeft(2, '0')}';
    return switch (s.status) {
      RecognizerStatus.idle => 'Tap the mic and play some audio',
      RecognizerStatus.connecting => 'Connecting…',
      RecognizerStatus.streaming when !detailed =>
        s.offline ? 'Checking for AI… $clock' : 'Listening… $clock',
      RecognizerStatus.finishing when !detailed && !s.offline =>
        'Naming the song…',
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
                  ButtonSegment(
                    value: c,
                    icon: Icon(c.icon),
                    label: Text(c.displayName),
                  ),
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
