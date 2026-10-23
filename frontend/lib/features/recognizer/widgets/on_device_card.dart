import 'package:flutter/material.dart';

import '../models/on_device_detection.dart';

/// On-device AI estimate: overall probability, per-band evidence and how much audio it covers.
class OnDeviceCard extends StatelessWidget {
  const OnDeviceCard({super.key, required this.detection, this.personalisedId});

  final OnDeviceDetection detection;

  /// Set when scores come from an on-device fine-tuned layer.
  final String? personalisedId;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final d = detection;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.phone_iphone_rounded, size: 20),
                const SizedBox(width: 8),
                Text('On-device AI estimate', style: theme.textTheme.titleMedium),
                if (personalisedId != null) ...[
                  const SizedBox(width: 8),
                  Tooltip(
                    message: 'Personalised layer $personalisedId',
                    child: Chip(
                      label: const Text('personalised'),
                      visualDensity: VisualDensity.compact,
                      avatar: const Icon(Icons.tune_rounded, size: 16),
                    ),
                  ),
                ],
              ],
            ),
            const SizedBox(height: 12),
            switch (d.status) {
              OnDeviceStatus.unavailable => const Text('Not available on this platform yet.'),
              OnDeviceStatus.failed => Text(d.error ?? 'On-device detection failed', style: TextStyle(color: theme.colorScheme.error)),
              OnDeviceStatus.listening when !d.hasScore => Row(
                children: [
                  const SizedBox.square(dimension: 16, child: CircularProgressIndicator(strokeWidth: 2)),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      (d.inputRmsDbfs ?? 0) < -60
                          ? 'Too quiet to score (${d.inputRmsDbfs!.toStringAsFixed(0)} dBFS) — move closer to the music'
                          : 'Listening — first estimate after 10 s',
                    ),
                  ),
                ],
              ),
              _ => _score(context, d),
            },
          ],
        ),
      ),
    );
  }

  Widget _score(BuildContext context, OnDeviceDetection d) {
    final theme = Theme.of(context);
    final p = d.pAi;
    final color = Color.lerp(Colors.green, Colors.red, p)!;
    final seconds = d.windows.length * 10;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text('${(p * 100).toStringAsFixed(0)}%', style: theme.textTheme.displaySmall?.copyWith(color: color)),
            const SizedBox(width: 8),
            Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Text(p >= 0.5 ? 'resembles AI-generated music' : 'resembles human-made music'),
            ),
          ],
        ),
        const SizedBox(height: 4),
        LinearProgressIndicator(value: p, color: color, minHeight: 6, borderRadius: BorderRadius.circular(3)),
        const SizedBox(height: 16),
        Text('Where the evidence is', style: theme.textTheme.labelLarge),
        const SizedBox(height: 8),
        for (final (i, name) in const ['Bass (20–250 Hz)', 'Mid (250 Hz–4 kHz)', 'Treble (4–16 kHz)'].indexed)
          Padding(
            padding: const EdgeInsets.only(bottom: 6),
            child: Row(
              children: [
                SizedBox(width: 150, child: Text(name, style: theme.textTheme.bodySmall)),
                Expanded(child: LinearProgressIndicator(value: d.bands[i], minHeight: 4)),
                SizedBox(width: 44, child: Text('${(d.bands[i] * 100).round()}%', textAlign: TextAlign.end, style: theme.textTheme.bodySmall)),
              ],
            ),
          ),
        const SizedBox(height: 8),
        Text(
          '${d.windows.length} × 10 s windows (~${seconds ~/ 60}:${(seconds % 60).toString().padLeft(2, '0')})'
          '${d.inputRmsDbfs == null ? '' : ' · input ${d.inputRmsDbfs!.toStringAsFixed(0)} dBFS'}'
          '${d.modelRun == null ? '' : ' · model ${d.modelRun}'}',
          style: theme.textTheme.bodySmall,
        ),
        const SizedBox(height: 4),
        Text(
          'Estimates resemblance to the AI generators it was trained on; it is not proof either way.',
          style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant),
        ),
      ],
    );
  }
}
