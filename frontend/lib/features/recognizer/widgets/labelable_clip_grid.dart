import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../../data/labels/recording_label.dart';
import '../models/clip_result.dart';
import '../models/on_device_detection.dart';
import 'clip_results_grid.dart';

/// [ClipResultsGrid] with clip labelling: each card's own label menu, plus "Select clips" to
/// tick several cards (shift-click for a range) and label them all at once.
class LabelableClipGrid extends StatefulWidget {
  const LabelableClipGrid({
    super.key,
    required this.clips,
    required this.onLabel,
    this.detection,
    this.selectedIndex,
    this.onTap,
    this.recordingLabel,
  });

  final List<ClipResult> clips;
  final OnDeviceDetection? detection;
  final int? selectedIndex;
  final ValueChanged<ClipResult>? onTap;
  final RecordingLabel? recordingLabel;

  /// Gives these clips their own label; null takes them back to the recording's.
  final void Function(Set<int> clipIndexes, AiCategory? category) onLabel;

  @override
  State<LabelableClipGrid> createState() => _LabelableClipGridState();
}

class _LabelableClipGridState extends State<LabelableClipGrid> {
  /// Ticked clip indexes; null when not selecting.
  Set<int>? _selection;

  /// Last clip ticked by a plain click, the start of a shift-click range.
  int? _anchor;

  static const _inherit = 'inherit';

  void _toggle(ClipResult clip) {
    final sel = {...?_selection};
    final order = [for (final c in widget.clips) c.index];
    final anchor = _anchor;
    if (HardwareKeyboard.instance.isShiftPressed && anchor != null && order.contains(anchor)) {
      final a = order.indexOf(anchor), b = order.indexOf(clip.index);
      sel.addAll(order.sublist(a < b ? a : b, (a < b ? b : a) + 1));
    } else {
      if (!sel.remove(clip.index)) sel.add(clip.index);
      _anchor = clip.index;
    }
    setState(() => _selection = sel);
  }

  void _apply(String value) {
    final sel = _selection;
    if (sel == null || sel.isEmpty) return;
    widget.onLabel(sel, value == _inherit ? null : AiCategory.fromWire(value));
    setState(() {
      _selection = null;
      _anchor = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final sel = _selection;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        if (sel == null)
          Align(
            alignment: Alignment.centerRight,
            child: TextButton.icon(
              icon: const Icon(Icons.checklist_rounded),
              label: const Text('Select clips'),
              onPressed: widget.clips.isEmpty ? null : () => setState(() => _selection = {}),
            ),
          )
        else
          Card(
            color: theme.colorScheme.secondaryContainer,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              child: Wrap(
                crossAxisAlignment: WrapCrossAlignment.center,
                spacing: 8,
                children: [
                  Text('${sel.length} selected', style: theme.textTheme.titleSmall),
                  TextButton(
                    onPressed: () => setState(
                      () => _selection = sel.length == widget.clips.length
                          ? {}
                          : {for (final c in widget.clips) c.index},
                    ),
                    child: Text(sel.length == widget.clips.length ? 'Select none' : 'Select all'),
                  ),
                  PopupMenuButton<String>(
                    enabled: sel.isNotEmpty,
                    tooltip: 'Label the selected clips',
                    onSelected: _apply,
                    itemBuilder: (_) => [
                      for (final cat in AiCategory.values)
                        PopupMenuItem(
                          value: cat.wire,
                          child: ListTile(
                            dense: true,
                            leading: Icon(cat.icon, color: cat.color),
                            title: Text(cat.displayName),
                          ),
                        ),
                      const PopupMenuDivider(),
                      const PopupMenuItem(value: _inherit, child: Text('Same as recording')),
                    ],
                    child: IgnorePointer(
                      child: FilledButton.icon(
                        icon: const Icon(Icons.label_rounded),
                        label: const Text('Label as…'),
                        onPressed: sel.isEmpty ? null : () {},
                      ),
                    ),
                  ),
                  TextButton(
                    onPressed: () => setState(() {
                      _selection = null;
                      _anchor = null;
                    }),
                    child: const Text('Done'),
                  ),
                  Text(
                    'Tap clips to select; shift-click for a range.',
                    style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSecondaryContainer),
                  ),
                ],
              ),
            ),
          ),
        const SizedBox(height: 8),
        ClipResultsGrid(
          clips: widget.clips,
          detection: widget.detection,
          selectedIndex: widget.selectedIndex,
          onTap: widget.onTap,
          recordingLabel: widget.recordingLabel,
          onLabel: (clip, c) => widget.onLabel({clip.index}, c),
          selection: sel,
          onToggle: _toggle,
        ),
      ],
    );
  }
}
