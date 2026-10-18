import 'package:flutter/material.dart';

/// Round mic button with a halo that grows with the input [level] while recording.
class MicButton extends StatelessWidget {
  const MicButton({
    super.key,
    required this.recording,
    required this.busy,
    required this.level,
    required this.progress,
    required this.onPressed,
  });

  final bool recording;
  final bool busy;
  final double level;

  /// Fraction of the maximum recording time used, 0..1.
  final double progress;
  final VoidCallback? onPressed;

  static const _size = 140.0;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final color = recording ? scheme.error : scheme.primary;

    return SizedBox.square(
      dimension: _size + 80,
      child: Stack(
        alignment: Alignment.center,
        children: [
          AnimatedContainer(
            duration: const Duration(milliseconds: 100),
            width: _size + (recording ? 80 * level : 0),
            height: _size + (recording ? 80 * level : 0),
            decoration: BoxDecoration(shape: BoxShape.circle, color: color.withValues(alpha: 0.18)),
          ),
          if (recording || busy)
            SizedBox.square(
              dimension: _size + 12,
              child: CircularProgressIndicator(value: busy ? null : progress, strokeWidth: 4, color: color),
            ),
          Semantics(
            button: true,
            label: recording ? 'Stop recording' : 'Start recording',
            child: Material(
              shape: const CircleBorder(),
              color: busy ? scheme.surfaceContainerHighest : color,
              elevation: recording ? 6 : 2,
              child: InkWell(
                customBorder: const CircleBorder(),
                onTap: onPressed,
                child: SizedBox.square(
                  dimension: _size,
                  child: Icon(
                    recording ? Icons.stop_rounded : Icons.mic_rounded,
                    size: 64,
                    color: busy ? scheme.onSurfaceVariant : scheme.onPrimary,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
