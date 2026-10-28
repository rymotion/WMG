import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../retrain/view/retrain_section.dart';
import '../cubit/personalization_cubit.dart';

/// Training mode, on-device retraining, model choice and sending the result to the computer.
class PersonalizationPage extends StatelessWidget {
  const PersonalizationPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('On-device training')),
      body: BlocConsumer<PersonalizationCubit, PersonalizationState>(
        listenWhen: (a, b) => b.message != null && a.message != b.message,
        listener: (context, s) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(s.message!),
              duration: const Duration(seconds: 6),
            ),
          );
          context.read<PersonalizationCubit>().clearMessage();
        },
        builder: (context, s) {
          final cubit = context.read<PersonalizationCubit>();
          final theme = Theme.of(context);
          final c = s.counts;
          final busy = s.busy != null;
          final canTrain =
              !busy &&
              s.base?.headWeights != null &&
              c.aiWindows >= 6 &&
              c.humanWindows >= 6;
          final r = s.head?.report;
          String pct(double? v) => v == null ? 'n/a' : '${(v * 100).round()}%';
          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              if (busy) const LinearProgressIndicator(),
              SwitchListTile(
                title: const Text('Training mode'),
                subtitle: const Text(
                  'After each recording, label it Human or AI to teach the model this device.',
                ),
                value: s.trainingMode,
                onChanged: s.base == null ? null : cubit.setTrainingMode,
              ),
              SwitchListTile(
                title: const Text('Offline mode'),
                subtitle: const Text(
                  'Use the on-device model without sending data to the server.',
                ),
                value: s.offlineMode,
                // Offline needs the bundled model; disabled where there is none.
                onChanged: s.base == null ? null : cubit.setOfflineMode,
              ),
              SwitchListTile(
                title: const Text('Identify mode'),
                subtitle: const Text(
                  'Look up each saved clip with ACRCloud on the server.',
                ),
                value: s.identifyMode,
                // A server setting, so it works without the on-device model.
                onChanged: cubit.setIdentifyMode,
              ),
              const Divider(),
              ListTile(
                title: const Text('Labelled audio'),
                subtitle: Text(
                  'AI: ${c.aiWindows} windows (${c.aiRecordings} recordings)\n'
                  'Human: ${c.humanWindows} windows (${c.humanRecordings} recordings)'
                  '${c.otherModelWindows > 0 ? '\n${c.otherModelWindows} windows from an older model (not used)' : ''}',
                ),
                isThreeLine: true,
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Text(
                  'Retraining needs at least 6 windows (about 1 minute) of each class, with neither class above 80%.',
                  style: theme.textTheme.bodySmall,
                ),
              ),
              const SizedBox(height: 12),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: FilledButton.icon(
                  icon: const Icon(Icons.model_training_rounded),
                  label: const Text('Retrain on this device'),
                  onPressed: canTrain ? cubit.retrain : null,
                ),
              ),
              const Divider(height: 32),
              if (s.head == null)
                const ListTile(
                  title: Text('No personalised model yet'),
                  subtitle: Text('Using the bundled model.'),
                )
              else ...[
                ListTile(
                  title: Text('Personalised layer ${s.head!.id}'),
                  subtitle: Text(
                    'Base model ${s.head!.baseModelRun}${s.headUsable ? '' : ' (different from the bundled model, not usable)'}\n'
                    'Accuracy on labelled windows ${pct(r!.accuracyBefore)} → ${pct(r.accuracyAfter)}; '
                    'held-out recordings ${pct(r.heldOutAccuracyBefore)} → ${pct(r.heldOutAccuracyAfter)}',
                  ),
                  isThreeLine: true,
                ),
                SwitchListTile(
                  title: const Text('Use personalised model'),
                  subtitle: Text(
                    s.usePersonal && s.headUsable
                        ? 'Scores use the fine-tuned layer.'
                        : 'Scores use the bundled model.',
                  ),
                  value: s.usePersonal && s.headUsable,
                  onChanged: s.headUsable ? cubit.setUsePersonal : null,
                ),
                const SizedBox(height: 8),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: [
                      FilledButton.tonalIcon(
                        icon: const Icon(Icons.cloud_upload_rounded),
                        label: const Text('Send to computer'),
                        onPressed: busy
                            ? null
                            : () => cubit.sendToComputer(
                                origin: _origin(context),
                              ),
                      ),
                      OutlinedButton.icon(
                        icon: const Icon(Icons.ios_share_rounded),
                        label: const Text('Save / share file'),
                        onPressed: busy
                            ? null
                            : () => cubit.exportFile(origin: _origin(context)),
                      ),
                    ],
                  ),
                ),
              ],
              const Divider(height: 32),
              const RetrainSection(),
              const Divider(height: 32),
              TextButton.icon(
                icon: const Icon(Icons.delete_outline_rounded),
                label: const Text(
                  'Delete labelled data and personalised layer',
                ),
                onPressed: busy ? null : () => _confirmReset(context),
              ),
            ],
          );
        },
      ),
    );
  }

  static Rect? _origin(BuildContext context) {
    final box = context.findRenderObject() as RenderBox?;
    return box == null ? null : box.localToGlobal(Offset.zero) & box.size;
  }

  Future<void> _confirmReset(BuildContext context) async {
    final cubit = context.read<PersonalizationCubit>();
    final ok = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete on-device training data?'),
        content: const Text(
          'Removes every labelled recording and the personalised layer from this device. '
          'Anything already sent to the computer is kept there.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (ok == true) await cubit.resetAll();
  }
}
