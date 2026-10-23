import 'package:flutter/material.dart';

import '../../../data/models/identification.dart';
import '../../../data/models/track_match.dart';
import '../models/clip_result.dart';

/// One saved clip of the live session and its identification result.
class ClipResultCard extends StatelessWidget {
  const ClipResultCard({super.key, required this.context, required this.clip});

  final BuildContext context;
  final ClipResult clip;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final id = clip.identification;
    final status = id?.status ?? IdentificationStatus.pending;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                if (id == null)
                  const SizedBox.square(
                    dimension: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                else
                  Icon(_icon(status), color: _color(status, theme.colorScheme)),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    _headline(status),
                    style: theme.textTheme.titleMedium,
                  ),
                ),
                Text(
                  'Clip ${clip.index + 1}',
                  style: theme.textTheme.labelMedium,
                ),
              ],
            ),
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
    >= 0.9 => 'High',
    >= 0.7 => 'Medium',
    >= 0.5 => 'Low',
    _ => 'Very Low',
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
