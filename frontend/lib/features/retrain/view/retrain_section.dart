import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../../data/retrain/retrain_client.dart';
import '../cubit/retrain_cubit.dart';

/// "Retrain models" on the Training page: retrains both detector models on the Mac from every
/// labelled recording, compares them with the installed ones, and installs them if they are not
/// worse. Worse models wait here for approval.
class RetrainSection extends StatefulWidget {
  const RetrainSection({super.key});

  @override
  State<RetrainSection> createState() => _RetrainSectionState();
}

class _RetrainSectionState extends State<RetrainSection> {
  late final RetrainCubit _cubit = context.read<RetrainCubit>();

  @override
  void initState() {
    super.initState();
    _cubit.watch();
  }

  @override
  void dispose() {
    _cubit.unwatch();
    super.dispose();
  }

  static const _stepNames = {
    'labels': 'Collect labelled recordings',
    'cache': 'Rebuild phone-audio cache',
    'train_10s': 'Train 10 s model',
    'train_5s': 'Train 5 s model',
    'compare': 'Compare with current models',
    'install': 'Install into the app',
    'build_app': 'Rebuild the macOS app',
  };

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<RetrainCubit, RetrainState>(
      builder: (context, st) {
        final theme = Theme.of(context);
        final s = st.status;
        final cubit = context.read<RetrainCubit>();
        final idle = st.reachable == true && !s.running && !s.awaitingApproval && !st.sending;
        return Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text('Retrain on the Mac', style: theme.textTheme.titleMedium),
              const SizedBox(height: 4),
              Text(
                'Retrains the 10 s and 5 s models from every labelled recording (about 75 minutes), '
                'compares them with the current ones and installs them if they are not worse.',
                style: theme.textTheme.bodySmall,
              ),
              if (s.current.isNotEmpty) ...[
                const SizedBox(height: 4),
                Text(
                  'In the app: 10 s ${s.current['10s'] ?? '-'}, 5 s ${s.current['5s'] ?? '-'}',
                  style: theme.textTheme.bodySmall,
                ),
              ],
              const SizedBox(height: 12),
              if (st.reachable == false)
                _Note(
                  icon: Icons.power_off_rounded,
                  color: theme.colorScheme.outline,
                  text:
                      'The retrain service isn\'t running. On the Mac, double-click '
                      'training/Retrain Service.command, then try again.',
                  action: TextButton(onPressed: cubit.refresh, child: const Text('Try again')),
                ),
              if (st.error != null)
                _Note(icon: Icons.error_outline_rounded, color: theme.colorScheme.error, text: st.error!),
              if (s.running) ...[
                const LinearProgressIndicator(),
                const SizedBox(height: 8),
              ],
              if (s.steps.isNotEmpty && s.state != 'idle')
                for (final step in s.steps) _StepRow(name: _stepNames[step.name] ?? step.name, step: step),
              if (s.running && s.lastLine.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(top: 4),
                  child: Text(
                    s.lastLine,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: theme.textTheme.bodySmall?.copyWith(fontFamily: 'Menlo', fontSize: 11),
                  ),
                ),
              if (!s.running && s.message != null && s.state != 'idle')
                _Note(
                  icon: switch (s.state) {
                    'done' => Icons.check_circle_rounded,
                    'awaiting_approval' => Icons.warning_amber_rounded,
                    _ => Icons.info_outline_rounded,
                  },
                  color: switch (s.state) {
                    'done' => Colors.green.shade700,
                    'awaiting_approval' => Colors.orange.shade800,
                    'failed' => theme.colorScheme.error,
                    _ => theme.colorScheme.outline,
                  },
                  text: s.message!,
                ),
              if (s.comparison.isNotEmpty) _ComparisonTable(comparison: s.comparison),
              const SizedBox(height: 12),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  if (s.awaitingApproval) ...[
                    FilledButton.icon(
                      icon: const Icon(Icons.check_rounded),
                      label: const Text('Install anyway'),
                      onPressed: st.sending ? null : () => _confirmApprove(context, cubit),
                    ),
                    OutlinedButton.icon(
                      icon: const Icon(Icons.undo_rounded),
                      label: const Text('Keep current models'),
                      onPressed: st.sending ? null : cubit.reject,
                    ),
                  ] else if (s.running)
                    OutlinedButton.icon(
                      icon: const Icon(Icons.stop_rounded),
                      label: const Text('Stop'),
                      onPressed: st.sending ? null : cubit.cancel,
                    )
                  else
                    FilledButton.icon(
                      icon: const Icon(Icons.autorenew_rounded),
                      label: const Text('Retrain models'),
                      onPressed: idle ? cubit.start : null,
                    ),
                  if (Platform.isMacOS && s.state == 'done' && s.builtApp && _stepDone(s, 'build_app'))
                    FilledButton.tonalIcon(
                      icon: const Icon(Icons.restart_alt_rounded),
                      label: const Text('Restart with new models'),
                      onPressed: st.sending
                          ? null
                          : () async {
                              await cubit.relaunch();
                              if (cubit.state.error == null) exit(0);
                            },
                    ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }

  static bool _stepDone(RetrainStatus s, String name) =>
      s.steps.any((x) => x.name == name && x.state == 'done');

  Future<void> _confirmApprove(BuildContext context, RetrainCubit cubit) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Install worse models?'),
        content: const Text(
          'The new models scored worse than the current ones on the checks shown. '
          'The current models stay in training/runs, so you can reinstall them later.',
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('Install')),
        ],
      ),
    );
    if (ok == true) await cubit.approve();
  }
}

class _StepRow extends StatelessWidget {
  const _StepRow({required this.name, required this.step});
  final String name;
  final RetrainStep step;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final (icon, color) = switch (step.state) {
      'done' => (Icons.check_circle_rounded, Colors.green.shade700),
      'running' => (Icons.sync_rounded, theme.colorScheme.primary),
      'failed' => (Icons.cancel_rounded, theme.colorScheme.error),
      'waiting' => (Icons.pause_circle_rounded, Colors.orange.shade800),
      _ => (Icons.radio_button_unchecked_rounded, theme.colorScheme.outline),
    };
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 18, color: color),
          const SizedBox(width: 8),
          Expanded(
            child: Text.rich(
              TextSpan(
                text: name,
                children: [
                  if (step.detail != null)
                    TextSpan(text: ' · ${step.detail}', style: TextStyle(color: theme.colorScheme.outline)),
                ],
              ),
              style: theme.textTheme.bodyMedium,
            ),
          ),
        ],
      ),
    );
  }
}

class _ComparisonTable extends StatelessWidget {
  const _ComparisonTable({required this.comparison});
  final Map<String, ModelComparison> comparison;

  /// key, label, higher is better, shown as a percentage.
  static const _rows = [
    ('song_auc', 'Clean files AUC', true, false),
    ('ai_recall', 'AI caught', true, true),
    ('human_fpr', 'Human flagged as AI', false, true),
    ('phone_auc', 'Through a speaker AUC', true, false),
    ('phone_ai_recall', 'Speaker: AI caught', true, true),
    ('phone_human_fpr', 'Speaker: human flagged', false, true),
  ];

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    String fmt(double? v, bool pct) =>
        v == null ? '–' : (pct ? '${(v * 100).toStringAsFixed(1)}%' : v.toStringAsFixed(3));
    return Padding(
      padding: const EdgeInsets.only(top: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          for (final e in comparison.entries) ...[
            Text(
              '${e.key == '10s' ? '10 s' : '5 s'} model: ${e.value.current?.run ?? 'none'} → ${e.value.newChecks.run}',
              style: theme.textTheme.labelLarge,
            ),
            Table(
              columnWidths: const {0: FlexColumnWidth(2.2), 1: FlexColumnWidth(), 2: FlexColumnWidth()},
              children: [
                TableRow(
                  children: [
                    for (final h in ['', 'Current', 'New'])
                      Text(h, style: theme.textTheme.labelSmall?.copyWith(color: theme.colorScheme.outline)),
                  ],
                ),
                for (final (key, label, higherBetter, pct) in _rows)
                  () {
                    final a = e.value.current?.values[key], b = e.value.newChecks.values[key];
                    final better = a == null || b == null || a == b ? null : (b > a) == higherBetter;
                    return TableRow(
                      children: [
                        Text(label, style: theme.textTheme.bodySmall),
                        Text(fmt(a, pct), style: theme.textTheme.bodySmall),
                        Text(
                          fmt(b, pct),
                          style: theme.textTheme.bodySmall?.copyWith(
                            fontWeight: FontWeight.w600,
                            color: better == null
                                ? null
                                : better
                                ? Colors.green.shade700
                                : Colors.red.shade700,
                          ),
                        ),
                      ],
                    );
                  }(),
              ],
            ),
            for (final r in e.value.regressions)
              Padding(
                padding: const EdgeInsets.only(top: 2),
                child: Text('Worse: $r', style: theme.textTheme.bodySmall?.copyWith(color: Colors.red.shade700)),
              ),
            const SizedBox(height: 8),
          ],
        ],
      ),
    );
  }
}

class _Note extends StatelessWidget {
  const _Note({required this.icon, required this.color, required this.text, this.action});
  final IconData icon;
  final Color color;
  final String text;
  final Widget? action;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 6),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 20, color: color),
        const SizedBox(width: 8),
        Expanded(child: Text(text, style: Theme.of(context).textTheme.bodyMedium)),
        ?action,
      ],
    ),
  );
}
