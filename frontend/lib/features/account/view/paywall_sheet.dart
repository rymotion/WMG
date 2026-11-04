import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../../data/account/catalog.dart';
import '../../../data/account/plan.dart';
import '../cubit/account_cubit.dart';

/// Plans and credit packs. [reason] tailors the headline to what the user just ran into.
Future<void> showPaywall(BuildContext context, {UpsellReason? reason}) =>
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      showDragHandle: true,
      builder: (_) => BlocProvider.value(
        value: context.read<AccountCubit>(),
        child: PaywallSheet(reason: reason),
      ),
    );

/// Asks a signed-out user to sign in; returns true once they have.
Future<bool> showSignInSheet(BuildContext context) async {
  final cubit = context.read<AccountCubit>();
  await showModalBottomSheet<void>(
    context: context,
    useSafeArea: true,
    showDragHandle: true,
    builder: (_) =>
        BlocProvider.value(value: cubit, child: const _SignInSheet()),
  );
  return cubit.state.signedIn;
}

class PaywallSheet extends StatefulWidget {
  const PaywallSheet({super.key, this.reason});

  final UpsellReason? reason;

  @override
  State<PaywallSheet> createState() => _PaywallSheetState();
}

class _PaywallSheetState extends State<PaywallSheet> {
  var _interval = BillingInterval.year;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return BlocBuilder<AccountCubit, AccountState>(
      builder: (context, state) {
        final usage = state.usage;
        final exhausted = widget.reason == UpsellReason.quotaExhausted;
        return SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                exhausted
                    ? 'You\'ve used your 10 free IDs'
                    : 'Name every song. Even in the loudest room.',
                style: theme.textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                exhausted && usage != null
                    ? 'They reset ${formatDay(usage.periodEnd)}. Go unlimited now, or grab a credit pack. '
                          'The AI check stays free either way.'
                    : 'Bars, clubs, live sets, remixes and covers: where other apps give up.',
                style: theme.textTheme.bodyMedium,
              ),
              const SizedBox(height: 16),
              SegmentedButton<BillingInterval>(
                segments: const [
                  ButtonSegment(
                    value: BillingInterval.month,
                    label: Text('Monthly'),
                  ),
                  ButtonSegment(
                    value: BillingInterval.year,
                    label: Text('Yearly · best value'),
                  ),
                ],
                selected: {_interval},
                showSelectedIcon: false,
                onSelectionChanged: (s) => setState(() => _interval = s.first),
              ),
              const SizedBox(height: 12),
              _PlanCard(
                product: Catalog.subscription(Tier.plus, _interval),
                current: state.tier == Tier.plus,
                features: const [
                  'Unlimited song IDs',
                  'Noisy Mode: listens up to 30 s in loud rooms',
                  'Searchable history, kept forever',
                  'Auto-playlists on Spotify & Apple Music',
                  'Record now, identify later',
                ],
              ),
              const SizedBox(height: 12),
              _PlanCard(
                product: Catalog.subscription(Tier.pro, _interval),
                current: state.tier == Tier.pro,
                highlighted: true,
                features: const [
                  'Everything in Plus',
                  '100 video/audio file IDs a month',
                  'Full timestamped tracklists from 1-hour mixes',
                  'BPM, key, energy, ISRC, label, release date',
                  'Export to CSV, Rekordbox and Serato',
                ],
              ),
              if (state.tier == Tier.free) ...[
                const SizedBox(height: 20),
                Text(
                  'Not ready to subscribe?',
                  style: theme.textTheme.titleSmall,
                ),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    for (final p in Catalog.creditPacks)
                      _CreditButton(product: p),
                  ],
                ),
                const SizedBox(height: 4),
                Text('Credits never expire.', style: theme.textTheme.bodySmall),
              ],
              const SizedBox(height: 16),
              TextButton(
                onPressed: state.purchasing
                    ? null
                    : () => context.read<AccountCubit>().restorePurchases(),
                child: const Text('Restore purchases'),
              ),
              Text(
                'We never sell your listening data. Subscriptions renew until cancelled; '
                'cancel anytime in your account settings.',
                style: theme.textTheme.bodySmall,
                textAlign: TextAlign.center,
              ),
            ],
          ),
        );
      },
    );
  }
}

class _PlanCard extends StatelessWidget {
  const _PlanCard({
    required this.product,
    required this.features,
    this.current = false,
    this.highlighted = false,
  });

  final Product product;
  final List<String> features;
  final bool current;
  final bool highlighted;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final state = context.watch<AccountCubit>().state;
    final per = product.interval == BillingInterval.year ? '/yr' : '/mo';
    final webPrice = '\$${product.webPriceUsd.toStringAsFixed(2)}$per';
    final storePrice = '${state.storePrice(product)}$per';
    return Card(
      color: highlighted ? theme.colorScheme.primaryContainer : null,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    product.tier!.displayName,
                    style: theme.textTheme.titleLarge,
                  ),
                ),
                Text(storePrice, style: theme.textTheme.titleMedium),
              ],
            ),
            const SizedBox(height: 8),
            for (final f in features)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 2),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(
                      Icons.check_rounded,
                      size: 18,
                      color: theme.colorScheme.primary,
                    ),
                    const SizedBox(width: 8),
                    Expanded(child: Text(f, style: theme.textTheme.bodyMedium)),
                  ],
                ),
              ),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: current || state.purchasing
                    ? null
                    : () => _buy(context, web: false),
                child: Text(
                  current ? 'Your plan' : 'Start ${product.tier!.displayName}',
                ),
              ),
            ),
            if (!current && state.account?.webCheckout == true)
              Center(
                child: TextButton(
                  onPressed: state.purchasing
                      ? null
                      : () => _buy(context, web: true),
                  child: Text('Or $webPrice on the web'),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Future<void> _buy(BuildContext context, {required bool web}) async {
    final cubit = context.read<AccountCubit>();
    if (!cubit.state.signedIn && !await showSignInSheet(context)) return;
    await (web ? cubit.buyOnWeb(product) : cubit.buyInApp(product));
  }
}

class _CreditButton extends StatelessWidget {
  const _CreditButton({required this.product});

  final Product product;

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AccountCubit>().state;
    return OutlinedButton(
      onPressed: state.purchasing
          ? null
          : () async {
              final cubit = context.read<AccountCubit>();
              if (!cubit.state.signedIn && !await showSignInSheet(context))
                return;
              await cubit.buyInApp(product);
            },
      child: Text('${product.title} · ${state.storePrice(product)}'),
    );
  }
}

class _SignInSheet extends StatelessWidget {
  const _SignInSheet();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final state = context.watch<AccountCubit>().state;
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            'Sign in to name the song',
            style: theme.textTheme.headlineSmall?.copyWith(
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            state.authAvailable
                ? '10 free song IDs every month, and your history saved. '
                      'Checking a track for AI stays free, no account needed.'
                : 'Accounts are coming soon. Until then, the AI check works without one.',
            style: theme.textTheme.bodyMedium,
          ),
          const SizedBox(height: 16),
          if (state.authAvailable)
            FilledButton(
              onPressed: () async {
                await context.read<AccountCubit>().signIn();
                if (context.mounted) Navigator.of(context).pop();
              },
              child: const Text('Continue'),
            )
          else
            OutlinedButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('OK'),
            ),
        ],
      ),
    );
  }
}

const _months = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec',
];

/// e.g. `Dec 1`.
String formatDay(DateTime t) {
  final local = t.toLocal();
  return '${_months[local.month - 1]} ${local.day}';
}
