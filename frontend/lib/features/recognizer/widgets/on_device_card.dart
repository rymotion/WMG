import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

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
                Text(
                  'AI detection in session',
                  style: theme.textTheme.titleMedium,
                ),
                if (personalisedId != null) ...[
                  const SizedBox(width: 8),
                  Tooltip(
                    message: 'AI waveform match $personalisedId',
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
              OnDeviceStatus.unavailable => const Text(
                'Not available on this platform yet.',
              ),
              OnDeviceStatus.failed => Text(
                d.error ?? 'On-device detection failed',
                style: TextStyle(color: theme.colorScheme.error),
              ),
              OnDeviceStatus.listening when !d.hasScore => Row(
                children: [
                  const SizedBox.square(
                    dimension: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      (d.inputRmsDbfs ?? 0) < -60
                          ? 'Too quiet to score (${d.inputRmsDbfs!.toStringAsFixed(0)} dBFS) — move closer to the music'
                          : 'Listening — first estimate after 5 s',
                    ),
                  ),
                ],
              ),
              _ => _score(context, d),
            },
            MaterialButton(onPressed: () {}, child: const Text('More details')),
          ],
        ),
      ),
    );
  }

  Widget _score(BuildContext context, OnDeviceDetection d) {
    final theme = Theme.of(context);
    final p = d.pAi;

    String pct(double? v) =>
        v == null ? '—' : '${(v * 100).toStringAsFixed(0)}%';
    final covered = [
      for (final w in d.windows) w.startS + 10,
      for (final w in d.windows5s) w.startS + 5,
    ].fold<double>(0, (a, b) => b > a ? b : a).round();

    final agree = d.agree;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildScoreComponent(p),
        Text(
          'Probability you caught AI generated sound: ${pct(p)}',
          style: theme.textTheme.displaySmall?.copyWith(color: Colors.white),
        ),
        const SizedBox(width: 8),
        Padding(
          padding: const EdgeInsets.only(bottom: 6),
          child: Text(
            p >= 0.5
                ? 'resembles AI-generated music'
                : 'resembles human-made music',
          ),
        ),
        const SizedBox(height: 8),
        Text(
          d.provisional
              ? 'Early estimate from the 5 s model; the 10 s model joins after 10 s of audio.'
              : d.delta == null
              ? '10 s model only.'
              : 'Average of the 10 s and 5 s models.',
          style: theme.textTheme.bodySmall,
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            _ModelPill(label: '10 s model', value: pct(d.p10)),
            _ModelPill(label: '5 s model', value: pct(d.p5)),
            if (d.delta != null)
              _ModelPill(
                label: 'Δ 5 s − 10 s',
                value:
                    '${d.delta! >= 0 ? '+' : '−'}${(d.delta!.abs() * 100).toStringAsFixed(0)} pts',
              ),
            if (agree != null)
              Chip(
                avatar: Icon(
                  agree ? Icons.check_circle_rounded : Icons.help_rounded,
                  size: 18,
                  color: agree ? Colors.green : theme.colorScheme.error,
                ),
                label: Text(
                  agree ? 'Models agree' : 'Models disagree — uncertain',
                ),
                visualDensity: VisualDensity.compact,
              ),
          ],
        ),
        const SizedBox(height: 16),
        Text('Where the evidence is', style: theme.textTheme.labelLarge),
        const SizedBox(height: 8),
        for (final (i, name) in const [
          'Bass (20–250 Hz)',
          'Mid (250 Hz–4 kHz)',
          'Treble (4–16 kHz)',
        ].indexed)
          Padding(
            padding: const EdgeInsets.only(bottom: 6),
            child: Row(
              children: [
                SizedBox(
                  width: 150,
                  child: Text(name, style: theme.textTheme.bodySmall),
                ),
                Expanded(
                  child: LinearProgressIndicator(
                    value: d.bands[i],
                    minHeight: 4,
                  ),
                ),
                SizedBox(
                  width: 44,
                  child: Text(
                    '${(d.bands[i] * 100).round()}%',
                    textAlign: TextAlign.end,
                    style: theme.textTheme.bodySmall,
                  ),
                ),
              ],
            ),
          ),
        const SizedBox(height: 8),
        Text(
          '~${covered ~/ 60}:${(covered % 60).toString().padLeft(2, '0')} scored · updates every 5 s'
          '${d.inputRmsDbfs == null ? '' : ' · input ${d.inputRmsDbfs!.toStringAsFixed(0)} dBFS'}'
          '${d.modelRun5s == null ? '' : ' · 5 s ${d.modelRun5s}'}'
          '${d.modelRun == null ? '' : ' · 10 s ${d.modelRun}'}',
          style: theme.textTheme.bodySmall,
        ),
        const SizedBox(height: 4),
        Text(
          'Estimates resemblance to the data and music the model was trained on; This is an automated system based on frequencies that pattern matched generated content. '
          'When the two models disagree by more than ${(OnDeviceDetection.agreementThreshold * 100).round()} points, '
          'treat the result as uncertain.',
          style: theme.textTheme.bodySmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
      ],
    );
  }

  Widget _buildScoreComponent(double p) {
    List<Widget> stars = _buildStarCount(((p * 100.0) / 20.0).toInt());

    if (p == 1.0) {
      stars.add(Icon(Icons.star, color: Colors.white, size: 12));
    }

    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        SvgPicture.asset(
          'assets/music-disclosure.svg',
          height: 250,
          width: 250,
          colorFilter: ColorFilter.mode(Colors.white, BlendMode.srcIn),
          imageBuilder: (context, child) {
            return Container(color: Colors.transparent, child: child);
          },
        ),
        ...stars,
      ],
    );
  }

  List<Widget> _buildStarCount(int starCount) => List.generate(
    5,
    (index) => Icon(
      index < starCount ? Icons.star : Icons.star_outline,
      color: Colors.white,
      size: 72,
    ),
  );
}

class _ModelPill extends StatelessWidget {
  const _ModelPill({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text.rich(
        TextSpan(
          children: [
            TextSpan(text: '$label  ', style: theme.textTheme.bodySmall),
            TextSpan(text: value, style: theme.textTheme.labelLarge),
          ],
        ),
      ),
    );
  }
}
