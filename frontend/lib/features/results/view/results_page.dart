import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../../core/app_config.dart';
import '../../../data/account/plan.dart';
import '../../account/cubit/account_cubit.dart';
import '../../account/view/paywall_sheet.dart';
import '../cubit/results_cubit.dart';
import '../data/session_history_store.dart';
import '../models/saved_session.dart';
import 'session_detail_page.dart';

/// Saved live sessions, newest first. Tap one to reopen it; swipe to delete.
class ResultsPage extends StatelessWidget {
  const ResultsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (context) =>
          ResultsCubit(store: context.read<SessionHistoryStore>())..load(),
      child: const ResultsView(),
    );
  }
}

class ResultsView extends StatefulWidget {
  const ResultsView({super.key});

  @override
  State<ResultsView> createState() => _ResultsViewState();
}

class _ResultsViewState extends State<ResultsView> {
  var _query = '';

  @override
  Widget build(BuildContext context) {
    // Handsets apply the plan's history rules; macOS shows everything.
    final entitlements = context.read<AppConfig>().billingEnabled
        ? context.select((AccountCubit c) => c.state.entitlements)
        : Entitlements.pro;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Saved sessions'),
        bottom: entitlements.searchableHistory
            ? PreferredSize(
                preferredSize: const Size.fromHeight(64),
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
                  child: SearchBar(
                    hintText: 'Search songs, artists, labels',
                    leading: const Icon(Icons.search_rounded),
                    onChanged: (q) => setState(() => _query = q),
                  ),
                ),
              )
            : null,
      ),
      body: BlocBuilder<ResultsCubit, ResultsState>(
        builder: (context, state) {
          if (!state.loaded) {
            return const Center(child: CircularProgressIndicator());
          }
          if (state.error != null) return Center(child: Text(state.error!));
          if (state.sessions.isEmpty) {
            return const Center(
              child: Text('Finished sessions are saved here.'),
            );
          }
          final (:shown, :hidden) = visibleSessions(
            state.sessions,
            retention: entitlements.historyRetention,
            query: _query,
          );
          return ListView.separated(
            padding: const EdgeInsets.symmetric(vertical: 8),
            itemCount: shown.length + (hidden > 0 ? 1 : 0),
            separatorBuilder: (_, _) => const Divider(height: 1),
            itemBuilder: (context, i) => i < shown.length
                ? _SessionTile(session: shown[i])
                : _HiddenHistoryTile(hidden: hidden),
          );
        },
      ),
    );
  }
}

class _SessionTile extends StatelessWidget {
  const _SessionTile({required this.session});

  final SavedSession session;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final s = session;
    final titles = s.matchedTitles;
    final p = s.onDevice.hasScore ? s.onDevice.pAi : null;
    return Dismissible(
      key: ValueKey(s.id),
      direction: DismissDirection.endToStart,
      background: Container(
        color: theme.colorScheme.errorContainer,
        alignment: Alignment.centerRight,
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Icon(
          Icons.delete_rounded,
          color: theme.colorScheme.onErrorContainer,
        ),
      ),
      confirmDismiss: (_) => confirmDelete(context),
      onDismissed: (_) => context.read<ResultsCubit>().delete(s),
      child: ListTile(
        leading: Icon(
          s.offline ? Icons.cloud_off_rounded : Icons.graphic_eq_rounded,
        ),
        title: Text(formatStartedAt(s.startedAt)),
        subtitle: Text(
          [
            formatDuration(s.durationS),
            ?s.label?.category.displayName,
            if (s.clips.isNotEmpty)
              '${s.clips.length} clip${s.clips.length == 1 ? '' : 's'}',
            if (titles.isNotEmpty)
              titles.take(2).join(', ') + (titles.length > 2 ? '…' : ''),
          ].join(' · '),
          maxLines: 2,
          overflow: TextOverflow.ellipsis,
        ),
        trailing: p == null
            ? null
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    '${(p * 100).round()}%',
                    style: theme.textTheme.titleMedium,
                  ),
                  Text('P(AI)', style: theme.textTheme.labelSmall),
                ],
              ),
        onTap: () async {
          await Navigator.of(context).push<bool>(
            MaterialPageRoute(builder: (_) => SessionDetailPage(session: s)),
          );
          // Reload after deletes and label changes made on the detail page.
          if (context.mounted) await context.read<ResultsCubit>().load();
        },
      ),
    );
  }
}

/// Free keeps 7 days of history; older sessions stay on the device until an upgrade shows them.
class _HiddenHistoryTile extends StatelessWidget {
  const _HiddenHistoryTile({required this.hidden});

  final int hidden;

  @override
  Widget build(BuildContext context) => ListTile(
    leading: const Icon(Icons.lock_clock_rounded),
    title: Text('$hidden older session${hidden == 1 ? '' : 's'} hidden'),
    subtitle: const Text(
      'Free keeps 7 days. Plus keeps your whole history, searchable.',
    ),
    trailing: FilledButton.tonal(
      onPressed: () => showPaywall(context),
      child: const Text('Upgrade'),
    ),
  );
}

/// Asks before a session and its recording are deleted.
Future<bool> confirmDelete(BuildContext context) async =>
    await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete this session?'),
        content: const Text(
          'Its results, spectrogram and recording are removed from this device.',
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
    ) ??
    false;

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

/// e.g. `Oct 26, 01:22`.
String formatStartedAt(DateTime t) =>
    '${_months[t.month - 1]} ${t.day}, ${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}';

/// e.g. `4:50`.
String formatDuration(double seconds) {
  final s = seconds.round();
  return '${s ~/ 60}:${(s % 60).toString().padLeft(2, '0')}';
}
