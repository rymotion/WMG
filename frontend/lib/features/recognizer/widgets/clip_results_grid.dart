import 'package:flutter/material.dart';

import '../../../data/labels/recording_label.dart';
import '../models/clip_result.dart';
import '../models/on_device_detection.dart';
import 'result_card.dart';
import 'spectrogram_view.dart';

/// Clip results laid out as a grid whose column count scales with the
/// available width (up to [maxColumns]). Rows size to their content so cards
/// with several matches expand instead of overflowing. Used live and when
/// reopening a saved session.
class ClipResultsGrid extends StatelessWidget {
  const ClipResultsGrid({
    super.key,
    required this.clips,
    this.detection,
    this.selectedIndex,
    this.onTap,
    this.recordingLabel,
    this.onLabel,
    this.selection,
    this.onToggle,
  });

  final List<ClipResult> clips;

  /// The whole session's on-device scores; each card gets its clip's slice.
  final OnDeviceDetection? detection;

  /// [ClipResult.index] of the highlighted card.
  final int? selectedIndex;

  /// Called for cards whose timing is known.
  final ValueChanged<ClipResult>? onTap;

  /// The whole recording's label; clips without their own take it.
  final RecordingLabel? recordingLabel;

  /// Sets a clip's own label (null: back to the recording's). Without it labels are only shown.
  final void Function(ClipResult clip, AiCategory? category)? onLabel;

  /// [ClipResult.index]es ticked while selecting; null when not selecting. While selecting,
  /// a tap toggles a card via [onToggle] instead of calling [onTap].
  final Set<int>? selection;
  final ValueChanged<ClipResult>? onToggle;

  static const maxColumns = 3;
  static const minCardWidth = 320.0;
  static const _spacing = 8.0;

  /// The clip's span of stream time, once the server has sent it.
  static SpectrogramRegion? regionOf(ClipResult clip) {
    final start = clip.startS;
    final duration = clip.durationS;
    if (start == null || duration == null) return null;
    return (startS: start, endS: start + duration);
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final width = constraints.maxWidth;
        final columns = (width / minCardWidth).floor().clamp(1, maxColumns);
        final cardWidth = (width - _spacing * (columns - 1)) / columns;
        return Wrap(
          spacing: _spacing,
          runSpacing: _spacing,
          children: [
            for (final clip in clips)
              SizedBox(
                width: cardWidth,
                child: ClipResultCard(
                  context: context,
                  clip: clip,
                  selected: clip.index == selectedIndex,
                  label: clip.label ?? recordingLabel,
                  labelInherited: clip.label == null && recordingLabel != null,
                  checked: selection?.contains(clip.index),
                  onLabel: onLabel == null || selection != null ? null : (c) => onLabel!(clip, c),
                  onTap: selection != null
                      ? (onToggle == null ? null : () => onToggle!(clip))
                      : onTap == null || regionOf(clip) == null
                      ? null
                      : () => onTap!(clip),
                  detection: switch ((detection, regionOf(clip))) {
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
}
