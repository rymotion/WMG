import 'package:flutter/material.dart';

import '../models/spectrogram.dart';

/// Scrolling live mel spectrogram: newest frame on the right, low bins at the bottom.
class SpectrogramView extends StatelessWidget {
  const SpectrogramView({super.key, required this.spectrogram, this.height = 128});

  final Spectrogram spectrogram;
  final double height;

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: SizedBox(
        height: height,
        width: double.infinity,
        child: ColoredBox(
          color: Colors.black,
          child: spectrogram.isEmpty
              ? Center(
                  child: Text('Waiting for spectrum…', style: TextStyle(color: Colors.white.withValues(alpha: 0.6))),
                )
              : CustomPaint(painter: _SpectrogramPainter(spectrogram)),
        ),
      ),
    );
  }
}

class _SpectrogramPainter extends CustomPainter {
  _SpectrogramPainter(this.spec);

  final Spectrogram spec;

  @override
  void paint(Canvas canvas, Size size) {
    final frames = spec.frames;
    final colWidth = size.width / spec.capacity;
    // Right-align so the newest frame sits at the right edge, like a scrolling canvas.
    final x0 = size.width - frames.length * colWidth;
    final paint = Paint()..isAntiAlias = false;
    final floor = spec.dbFloor;

    for (var f = 0; f < frames.length; f++) {
      final frame = frames[f];
      final bins = frame.length;
      if (bins == 0) continue;
      final rowHeight = size.height / bins;
      final left = x0 + f * colWidth;
      for (var b = 0; b < bins; b++) {
        // Absolute dB: floor → 0, ~0 dBFS → 1. Same mapping as the guide's canvas example.
        final v = ((frame[b] - floor) / -floor).clamp(0.0, 1.0);
        paint.color = HSLColor.fromAHSL(1, (280 - v * 240) % 360, 0.9, v * 0.6).toColor();
        canvas.drawRect(Rect.fromLTWH(left, size.height - (b + 1) * rowHeight, colWidth + 0.5, rowHeight + 0.5), paint);
      }
    }
  }

  @override
  bool shouldRepaint(_SpectrogramPainter old) => old.spec != spec;
}
