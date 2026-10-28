import 'package:flutter/material.dart';

import '../../../data/labels/recording_label.dart';
import 'package:wmg_frontend/common/star_builder.dart';

import '../../../data/models/identification.dart';
import '../../../data/models/track_match.dart';
import '../models/clip_result.dart';
import '../models/on_device_detection.dart';
import 'spectrogram_view.dart';

/// One saved clip of the live session and its identification result.
class ClipResultCard extends StatelessWidget {
  const ClipResultCard({
    super.key,
    required this.context,
    required this.clip,
    this.selected = false,
    this.detection,
    this.onTap,
    this.label,
    this.labelInherited = false,
    this.onLabel,
    this.checked,
  });

  final BuildContext context;
  final ClipResult clip;

  /// On-device scores for just this clip's time span ([OnDeviceDetection.slice]);
  /// null where there's no detector or the clip's timing hasn't arrived.
  final OnDeviceDetection? detection;

  /// Outlined, e.g. while its span is highlighted on the spectrogram.
  final bool selected;
  final VoidCallback? onTap;

  /// The clip's label: its own, or the recording's when [labelInherited].
  final RecordingLabel? label;
  final bool labelInherited;

  /// Sets this clip's own label; null goes back to the recording's. Without it the label is
  /// only shown.
  final ValueChanged<AiCategory?>? onLabel;

  /// Non-null while clips are being selected: shows a checkbox in this state.
  final bool? checked;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final id = clip.identification;
    final status = id?.status ?? IdentificationStatus.pending;

    return Card(
      clipBehavior: Clip.antiAlias,
      shape: selected
          ? RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
              side: BorderSide(color: theme.colorScheme.primary, width: 2),
            )
          : null,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildScoreCardHeader(status, theme, clip),
              _buildStarRating(context),
              ..._buildProbability(theme),
              if (label != null || onLabel != null) ...[
                const SizedBox(height: 8),
                _ClipLabelRow(label: label, inherited: labelInherited, onLabel: onLabel),
              ],
              if (id != null && id.matches.isNotEmpty) ...[
                const SizedBox(height: 8),
                for (final match in id.matches) _MatchTile(match: match),
              ] else if (id?.message != null) ...[
                const SizedBox(height: 8),
                Text(id!.message!, style: theme.textTheme.bodyMedium),
              ],
              const SizedBox(height: 8),
              Text(
                _timing(),
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStarRating(BuildContext context) {
    try {
      final d = detection;
      return StarCount(score: d?.pAi ?? 0, height: 45, width: 45);
    } catch (e) {
      return const SizedBox();
    }
  }

  /// This clip's on-device P(AI): headline, both models and their gap, and
  /// the window scores over the clip.
  List<Widget> _buildProbability(ThemeData theme) {
    final d = detection;
    final start = clip.startS;
    final duration = clip.durationS;
    if (d == null || start == null || duration == null) return const [];
    final muted = theme.textTheme.bodySmall?.copyWith(
      color: theme.colorScheme.onSurfaceVariant,
    );
    if (!d.hasScore) {
      return [
        const SizedBox(height: 8),
        Text('No on-device score for this clip yet.', style: muted),
      ];
    }
    String pct(double? p) => p == null ? '–' : '${(p * 100).round()}%';
    final delta = d.delta;
    return [
      const SizedBox(height: 12),
      Text(
        'P(AI) ${pct(d.pAi)} · ${_generatedProbability(d.pAi)}',
        style: theme.textTheme.titleSmall,
      ),
      const SizedBox(height: 2),
      Text(
        [
          '5 s ${pct(d.p5)}',
          '10 s ${pct(d.p10)}',
          if (delta != null)
            'Δ ${delta >= 0 ? '+' : ''}${delta.toStringAsFixed(2)}',
        ].join(' · '),
        style: muted,
      ),
      const SizedBox(height: 8),
      WindowScoreChart(
        scores: scoreSeries(d, theme.colorScheme),
        startS: start,
        endS: start + duration,
        label: label?.category,
      ),
    ];
  }

  Widget _buildScoreCardHeader(
    IdentificationStatus status,
    ThemeData theme,
    ClipResult result,
  ) {
    final id = result.identification;
    return Row(
      children: [
        if (checked != null)
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: Icon(
              checked! ? Icons.check_box_rounded : Icons.check_box_outline_blank_rounded,
              color: checked! ? theme.colorScheme.primary : theme.colorScheme.outline,
            ),
          ),
        if (id == null)
          const SizedBox.square(
            dimension: 20,
            child: CircularProgressIndicator(strokeWidth: 2),
          )
        else
          Icon(_icon(status), color: _color(status, theme.colorScheme)),
        const SizedBox(width: 8),
        Expanded(
          child: Text(_headline(status), style: theme.textTheme.titleMedium),
        ),
        Text('Clip ${clip.index + 1}', style: theme.textTheme.labelMedium),
      ],
    );
  }

  String _timing() {
    final start = clip.startS;
    final duration = clip.durationS;
    if (start == null || duration == null) return clip.id;
    return '${_mmss(start)}–${_mmss(start + duration)} · ${clip.id}';
  }

  static String _mmss(double s) =>
      '${s ~/ 60}:${(s % 60).floor().toString().padLeft(2, '0')}';

  static String _headline(IdentificationStatus s) => switch (s) {
    IdentificationStatus.pending => 'Identifying…',
    IdentificationStatus.matched => 'Match found',
    IdentificationStatus.possibleMatch => 'Possible match (low confidence)',
    IdentificationStatus.noMatch => 'No match',
    IdentificationStatus.error => 'Identification failed',
    IdentificationStatus.disabled => 'Identification disabled on server',
    IdentificationStatus.skipped => 'Identification skipped',
    IdentificationStatus.unknown => 'Unknown status',
  };

  String _generatedProbability(double confidence) => switch (confidence) {
    >= 0.9 => 'Fully AI-generated',
    >= 0.7 => 'Mostly AI-generated',
    >= 0.5 => 'Partially AI-generated',
    >= 0.2 => 'Computers used in production',
    _ => 'Not AI-generated',
  };

  static IconData _icon(IdentificationStatus s) => switch (s) {
    IdentificationStatus.matched => Icons.check_circle_rounded,
    IdentificationStatus.possibleMatch => Icons.help_rounded,
    IdentificationStatus.noMatch => Icons.search_off_rounded,
    IdentificationStatus.error => Icons.error_rounded,
    _ => Icons.info_rounded,
  };

  static Color _color(IdentificationStatus s, ColorScheme c) => switch (s) {
    IdentificationStatus.matched => Colors.green,
    IdentificationStatus.possibleMatch => Colors.amber.shade700,
    IdentificationStatus.error => c.error,
    _ => c.onSurfaceVariant,
  };
}

/// "AI-generated (recording)" plus a menu to give this clip its own label.
class _ClipLabelRow extends StatelessWidget {
  const _ClipLabelRow({required this.label, required this.inherited, required this.onLabel});

  final RecordingLabel? label;
  final bool inherited;
  final ValueChanged<AiCategory?>? onLabel;

  static const _inherit = 'inherit';

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final c = label?.category;
    final muted = theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant);
    final text = Text.rich(
      TextSpan(
        text: c?.displayName ?? 'Unlabelled clip',
        children: [if (c != null && inherited) TextSpan(text: ' · from recording', style: muted)],
      ),
      style: theme.textTheme.labelLarge?.copyWith(color: c?.color),
    );
    final row = Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(c?.icon ?? Icons.label_outline_rounded, size: 18, color: c?.color ?? theme.colorScheme.outline),
        const SizedBox(width: 6),
        Flexible(child: text),
        if (onLabel != null) Icon(Icons.arrow_drop_down_rounded, color: theme.colorScheme.outline),
      ],
    );
    if (onLabel == null) return row;
    return PopupMenuButton<String>(
      tooltip: 'Label this clip',
      onSelected: (v) => onLabel!(v == _inherit ? null : AiCategory.fromWire(v)),
      itemBuilder: (_) => [
        for (final cat in AiCategory.values)
          CheckedPopupMenuItem(
            value: cat.wire,
            checked: !inherited && c == cat,
            child: Text(cat.displayName),
          ),
        if (label != null && !inherited) ...[
          const PopupMenuDivider(),
          const PopupMenuItem(value: _inherit, child: Text('Same as recording')),
        ],
      ],
      child: row,
    );
  }
}

class _MatchTile extends StatelessWidget {
  const _MatchTile({required this.match});

  final TrackMatch match;

  @override
  Widget build(BuildContext context) {
    final details = [
      if (match.album != null) match.album!,
      if (match.releaseDate != null) match.releaseDate!,
      if (match.label != null) match.label!,
    ].join(' · ');

    return ListTile(
      contentPadding: EdgeInsets.zero,
      leading: const CircleAvatar(child: Icon(Icons.music_note_rounded)),
      title: Text(match.displayTitle),
      subtitle: Text(
        details.isEmpty
            ? match.displayArtists
            : '${match.displayArtists}\n$details',
      ),
      isThreeLine: details.isNotEmpty,
      trailing: match.score == null ? null : Text('${match.score!.round()}%'),
    );
  }
}
