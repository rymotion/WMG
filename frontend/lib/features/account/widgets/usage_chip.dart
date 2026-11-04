import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../../data/account/plan.dart';
import '../cubit/account_cubit.dart';
import '../view/account_page.dart';
import '../view/paywall_sheet.dart';

/// Under the mic: how many song IDs are left, and the way to more.
class UsageChip extends StatelessWidget {
  const UsageChip({super.key});

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AccountCubit>().state;
    final usage = state.usage;
    final (label, onTap) = switch (state) {
      _ when !state.signedIn => (
        'Sign in for 10 free song IDs a month',
        () => showSignInSheet(context),
      ),
      _ when usage == null || usage.unlimitedMic => (
        'Unlimited IDs · ${state.tier.displayName}',
        () => Navigator.of(context).push(AccountPage.route()),
      ),
      _ => (
        '${usage.micIdsLeft} song ID${usage.micIdsLeft == 1 ? '' : 's'} left · resets ${formatDay(usage.periodEnd)}',
        () => showPaywall(context),
      ),
    };
    final low = usage != null && !usage.unlimitedMic && usage.micIdsLeft! <= 2;
    final colors = Theme.of(context).colorScheme;
    return ActionChip(
      avatar: Icon(
        state.tier == Tier.free
            ? Icons.bolt_rounded
            : Icons.all_inclusive_rounded,
        size: 18,
        color: low ? colors.error : colors.primary,
      ),
      label: Text(label),
      onPressed: onTap,
    );
  }
}
