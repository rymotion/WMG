import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../../data/account/plan.dart';
import '../cubit/account_cubit.dart';
import 'paywall_sheet.dart';

/// Plan, this month's usage, billing and sign-out.
class AccountPage extends StatelessWidget {
  const AccountPage({super.key});

  static Route<void> route() =>
      MaterialPageRoute(builder: (_) => const AccountPage());

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('Account')),
      body: BlocBuilder<AccountCubit, AccountState>(
        builder: (context, state) {
          final cubit = context.read<AccountCubit>();
          final account = state.account;
          if (account == null) {
            return Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      'You\'re not signed in.',
                      style: theme.textTheme.titleMedium,
                    ),
                    const SizedBox(height: 12),
                    FilledButton(
                      onPressed: () => showSignInSheet(context),
                      child: const Text('Sign in'),
                    ),
                  ],
                ),
              ),
            );
          }
          final usage = account.usage;
          final limit = usage.micIdsLimit;
          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              ListTile(
                title: Text(
                  account.tier.displayName,
                  style: theme.textTheme.titleLarge,
                ),
                subtitle: Text(
                  [
                    ?account.email,
                    if (account.renewsAt != null)
                      'Renews ${formatDay(account.renewsAt!)}',
                  ].join(' · '),
                ),
              ),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('This month', style: theme.textTheme.titleSmall),
                      const SizedBox(height: 8),
                      if (limit == null)
                        const Text('Unlimited song IDs')
                      else ...[
                        Text(
                          '${usage.micIdsUsed} of $limit song IDs used · resets ${formatDay(usage.periodEnd)}',
                        ),
                        const SizedBox(height: 8),
                        LinearProgressIndicator(
                          value: limit == 0
                              ? 1
                              : (usage.micIdsUsed / limit).clamp(0, 1),
                        ),
                      ],
                      if (usage.uploadIdsLimit > 0) ...[
                        const SizedBox(height: 12),
                        Text(
                          '${usage.uploadIdsUsed} of ${usage.uploadIdsLimit} file IDs used',
                        ),
                      ],
                      if (usage.micCredits > 0 || usage.videoCredits > 0) ...[
                        const SizedBox(height: 12),
                        Text(
                          'Credits: ${usage.micCredits} song IDs · ${usage.videoCredits} video IDs',
                        ),
                      ],
                      const SizedBox(height: 8),
                      Text(
                        'No match? It doesn\'t count.',
                        style: theme.textTheme.bodySmall,
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 8),
              if (account.tier != Tier.pro)
                FilledButton(
                  onPressed: () => showPaywall(context),
                  child: Text(
                    account.tier == Tier.free ? 'Upgrade' : 'Go Pro Creator',
                  ),
                ),
              if (account.billingSource != null)
                OutlinedButton(
                  onPressed: cubit.manageSubscription,
                  child: const Text('Manage subscription'),
                ),
              TextButton(
                onPressed: cubit.restorePurchases,
                child: const Text('Restore purchases'),
              ),
              const Divider(height: 32),
              TextButton(
                onPressed: cubit.signOut,
                child: const Text('Sign out'),
              ),
              const SizedBox(height: 8),
              Text(
                'We never sell your listening data. We count identifications and network use only to '
                'enforce your plan and keep the service running.',
                style: theme.textTheme.bodySmall,
                textAlign: TextAlign.center,
              ),
            ],
          );
        },
      ),
    );
  }
}
