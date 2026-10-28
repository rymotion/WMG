import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:just_audio/just_audio.dart';

import '../../../data/labels/recording_label.dart';
import '../../recognizer/models/clip_result.dart';
import '../../recognizer/models/spectrogram.dart';
import '../../recognizer/widgets/clip_results_grid.dart';
import '../../recognizer/widgets/labelable_clip_grid.dart';
import '../../recognizer/widgets/on_device_card.dart';
import '../../recognizer/widgets/spectrogram_view.dart';
import '../data/session_history_store.dart';
import '../models/saved_session.dart';
import 'results_page.dart';

/// One saved session, laid out like the live view: the spectrogram and P(AI) chart,
/// the on-device estimate and the clip cards, plus playback of the recording. Tapping
/// a clip card or the spectrogram moves both the view and playback there.
class SessionDetailPage extends StatefulWidget {
  const SessionDetailPage({super.key, required this.session});

  final SavedSession session;

  @override
  State<SessionDetailPage> createState() => _SessionDetailPageState();
}

class _SessionDetailPageState extends State<SessionDetailPage> {
  final _spectrogram = SpectrogramController();
  final _player = AudioPlayer();
  final _subs = <StreamSubscription<Object?>>[];

  Spectrogram? _spec;
  bool _hasAudio = false;
  String? _audioError;
  Duration _position = Duration.zero;
  bool _playing = false;
  int? _selectedClip;

  /// The session as last saved; replaced when its label changes.
  late SavedSession _current = widget.session;

  SavedSession get _session => _current;

  Future<void> _setLabel(AiCategory? category) async {
    final store = context.read<SessionHistoryStore>();
    try {
      final updated = await store.setLabel(
        _session.id,
        category == null ? null : RecordingLabel.now(category),
      );
      if (mounted) setState(() => _current = updated);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not save the label: $e')),
        );
      }
    }
  }

  Future<void> _setClipLabels(Set<int> clipIndexes, AiCategory? category) async {
    final store = context.read<SessionHistoryStore>();
    try {
      final updated = await store.setClipLabels(
        _session.id,
        clipIndexes,
        category == null ? null : RecordingLabel.now(category),
      );
      if (mounted) setState(() => _current = updated);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not save the clip labels: $e')),
        );
      }
    }
  }

  @override
  void initState() {
    super.initState();
    _subs
      ..add(_player.positionStream.listen((p) => setState(() => _position = p)))
      ..add(_player.playingStream.listen((p) => setState(() => _playing = p)))
      ..add(
        _player.processingStateStream.listen((s) {
          // Back to the start when it plays out, ready to play again.
          if (s == ProcessingState.completed) {
            _player
              ..pause()
              ..seek(Duration.zero);
          }
        }),
      );
    unawaited(_load());
  }

  Future<void> _load() async {
    final store = context.read<SessionHistoryStore>();
    final spec = await store.loadSpectrogram(_session);
    final audio = await store.audioFile(_session);
    if (!mounted) return;
    setState(() => _spec = spec);
    if (audio == null) return;
    try {
      await _player.setFilePath(audio.path);
      if (mounted) setState(() => _hasAudio = true);
    } catch (e) {
      if (mounted) {
        setState(() => _audioError = 'Could not open the recording: $e');
      }
    }
  }

  @override
  void dispose() {
    for (final s in _subs) {
      s.cancel();
    }
    _player.dispose();
    super.dispose();
  }

  void _seekAudio(double seconds) {
    if (_hasAudio) {
      _player.seek(Duration(milliseconds: (seconds * 1000).round()));
    }
  }

  void _showClip(ClipResult clip) {
    setState(() => _selectedClip = clip.index);
    _spectrogram.seekTo(clip.startS!);
    if (_hasAudio) {
      _seekAudio(clip.startS!);
      _player.play();
    }
  }

  Future<void> _delete() async {
    if (!await confirmDelete(context) || !mounted) return;
    await _player.stop();
    if (!mounted) return;
    await context.read<SessionHistoryStore>().delete(_session.id);
    if (mounted) Navigator.of(context).pop(true);
  }

  @override
  Widget build(BuildContext context) {
    final s = _session;
    final spec = _spec;
    SpectrogramRegion? selected;
    for (final c in s.clips) {
      if (c.index == _selectedClip) selected = ClipResultsGrid.regionOf(c);
    }
    return Scaffold(
      appBar: AppBar(
        title: Text(formatStartedAt(s.startedAt)),
        actions: [
          IconButton(
            tooltip: 'Delete session',
            icon: const Icon(Icons.delete_outline_rounded),
            onPressed: _delete,
          ),
        ],
      ),
      body: spec == null
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _buildPlayer(context),
                  const SizedBox(height: 12),
                  _SessionLabelPicker(label: s.label, onChanged: _setLabel),
                  const SizedBox(height: 12),
                  SpectrogramView(
                    spectrogram: spec,
                    controller: _spectrogram,
                    regions: [
                      for (final c in s.clips) ?ClipResultsGrid.regionOf(c),
                    ],
                    highlight: selected,
                    scores: scoreSeries(
                      s.onDevice,
                      Theme.of(context).colorScheme,
                    ),
                    playheadS: _hasAudio
                        ? _position.inMilliseconds / 1000
                        : null,
                    onTapTime: _hasAudio ? _seekAudio : null,
                    endLabel: 'Jump to end',
                    label: s.label?.category,
                  ),
                  if (s.onDevice.hasScore) ...[
                    const SizedBox(height: 16),
                    OnDeviceCard(detection: s.onDevice),
                  ],
                  if (s.clips.isNotEmpty) ...[
                    const SizedBox(height: 16),
                    LabelableClipGrid(
                      clips: s.clips,
                      detection: s.onDevice.hasScore ? s.onDevice : null,
                      selectedIndex: _selectedClip,
                      onTap: _showClip,
                      recordingLabel: s.label,
                      onLabel: _setClipLabels,
                    ),
                  ],
                ],
              ),
            ),
    );
  }

  Widget _buildPlayer(BuildContext context) {
    final theme = Theme.of(context);
    if (!_hasAudio) {
      return Text(
        _audioError ?? 'No recording was kept for this session.',
        style: theme.textTheme.bodySmall?.copyWith(
          color: theme.colorScheme.onSurfaceVariant,
        ),
      );
    }
    final total =
        _player.duration ??
        Duration(milliseconds: (_session.durationS * 1000).round());
    final totalS = math.max(total.inMilliseconds / 1000, 0.001);
    final posS = (_position.inMilliseconds / 1000).clamp(0.0, totalS);
    return Row(
      children: [
        IconButton.filledTonal(
          tooltip: _playing ? 'Pause' : 'Play',
          icon: Icon(_playing ? Icons.pause_rounded : Icons.play_arrow_rounded),
          onPressed: _playing ? _player.pause : _player.play,
        ),
        Expanded(
          child: Slider(value: posS, max: totalS, onChanged: _seekAudio),
        ),
        Text(
          '${formatDuration(posS)} / ${formatDuration(totalS)}',
          style: theme.textTheme.labelMedium?.copyWith(
            fontFeatures: const [FontFeature.tabularFigures()],
          ),
        ),
      ],
    );
  }
}

/// Human-made / AI-assisted / AI-generated for the whole recording; saved with the session
/// and its audio, and drawn on the P(AI) chart.
class _SessionLabelPicker extends StatelessWidget {
  const _SessionLabelPicker({required this.label, required this.onChanged});

  final RecordingLabel? label;
  final ValueChanged<AiCategory?> onChanged;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final c = label?.category;
    return Card(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 8, 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(child: Text('Label', style: theme.textTheme.titleSmall)),
                if (c != null)
                  TextButton(onPressed: () => onChanged(null), child: const Text('Clear')),
              ],
            ),
            Text(
              c == null
                  ? 'What is this recording? The label is kept with its audio for retraining. '
                        'Clips that differ can be labelled on their own cards below.'
                  : '${c.displayName}: about ${(c.aiShareMin * 100).round()}–${(c.aiShareMax * 100).round()}% AI'
                        '${c.binaryLabel == null ? ' (kept for the AI-share model; not used in Human-vs-AI training yet)' : ''}.',
              style: theme.textTheme.bodySmall,
            ),
            const SizedBox(height: 8),
            SegmentedButton<AiCategory>(
              segments: [
                for (final cat in AiCategory.values)
                  ButtonSegment(value: cat, icon: Icon(cat.icon), label: Text(cat.displayName)),
              ],
              selected: {?c},
              emptySelectionAllowed: true,
              showSelectedIcon: false,
              onSelectionChanged: (sel) {
                if (sel.isNotEmpty) onChanged(sel.first);
              },
            ),
          ],
        ),
      ),
    );
  }
}
