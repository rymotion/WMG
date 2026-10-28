import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:wmg_frontend/data/labels/recording_label.dart';
import 'package:wmg_frontend/data/retrain/retrain_client.dart';
import 'package:wmg_frontend/features/recognizer/models/clip_result.dart';
import 'package:wmg_frontend/features/recognizer/models/spectrogram.dart';
import 'package:wmg_frontend/features/results/data/session_history_store.dart';
import 'package:wmg_frontend/features/results/models/saved_session.dart';
import 'package:wmg_frontend/features/retrain/cubit/retrain_cubit.dart';
import 'package:wmg_frontend/features/retrain/view/retrain_section.dart';

const _held = {
  'state': 'awaiting_approval',
  'message': 'New models are worse on: 10s: song auc 0.943 -> 0.920',
  'steps': {
    'labels': {'state': 'done', 'detail': '6 new'},
    'install': {'state': 'waiting'},
  },
  'labels': {'new': [{}, {}], 'conflicts': [{}]},
  'current': {'10s': '20261027-005541', '5s': '20261027-014414'},
  'pending': {
    'comparison': {
      '10s': {
        'new': {'run': 'n10', 'song_auc': 0.92, 'gate_auc': 0.45},
        'current': {'run': '20261027-005541', 'song_auc': 0.943},
        'regressions': ['song auc 0.943 -> 0.920 (more than 0.01 worse)'],
      },
    },
  },
  'built_app': false,
};

void main() {
  test('status: a held run shows its comparison and regressions', () {
    final s = RetrainStatus.fromJson(jsonDecode(jsonEncode(_held)) as Map<String, dynamic>);
    expect(s.awaitingApproval, isTrue);
    expect(s.steps.map((x) => x.name), ['labels', 'install']);
    expect(s.newRecordings, 2);
    expect(s.conflicts, 1);
    final c = s.comparison['10s']!;
    expect(c.current!.values['song_auc'], 0.943);
    expect(c.newChecks.run, 'n10');
    expect(c.regressions.single, startsWith('song auc'));
  });

  test('start sends this device\'s labelled backend sessions; errors surface', () async {
    final tmp = await Directory.systemTemp.createTemp('retrain');
    addTearDown(() => tmp.delete(recursive: true));
    final store = SessionHistoryStore(root: tmp);
    SavedSession s(String id, String? server, AiCategory? c) => SavedSession(
      id: id,
      startedAt: DateTime.utc(2026, 10, 27),
      durationS: 60,
      serverSessionId: server,
      label: c == null ? null : RecordingLabel.now(c),
    );
    await store.save(s('a', 'srv-a', AiCategory.aiAssisted), const Spectrogram());
    await store.save(s('b', null, AiCategory.ai), const Spectrogram()); // offline: no audio on the server
    await store.save(s('c', 'srv-c', null), const Spectrogram()); // unlabelled
    await store.save(
      SavedSession(
        id: 'd',
        startedAt: DateTime.utc(2026, 10, 27),
        durationS: 60,
        serverSessionId: 'srv-d',
        clips: [ClipResult(index: 2, id: 'x', label: RecordingLabel.now(AiCategory.ai))],
      ),
      const Spectrogram(),
    ); // only one clip labelled

    final posts = <(String, Object?)>[];
    var busy = false;
    final mock = MockClient((r) async {
      if (r.method == 'GET') return http.Response(jsonEncode({'state': 'idle'}), 200);
      posts.add((r.url.path, jsonDecode(r.body)));
      return busy ? http.Response(jsonEncode({'error': 'a retrain is already running'}), 409) : http.Response('{}', 202);
    });
    final cubit = RetrainCubit(
      client: RetrainClient(baseUrl: Uri.parse('http://localhost:9100'), client: mock),
      history: store,
    );
    await cubit.start();
    expect(posts.single.$1, '/retrain');
    expect(
      (posts.single.$2 as Map)['labels'],
      unorderedEquals([
        {'session_id': 'srv-a', 'category': 'ai_assisted', 'id': 'a'},
        {'session_id': 'srv-d', 'category': 'ai', 'clip_index': 2, 'id': 'd'},
      ]),
    );
    expect(cubit.state.reachable, isTrue);
    expect(cubit.state.error, isNull);

    busy = true;
    await cubit.start();
    expect(cubit.state.error, 'a retrain is already running');
    await cubit.close();
  });

  testWidgets('a held run shows the comparison and both decisions', (tester) async {
    final mock = MockClient((r) async => http.Response(jsonEncode(_held), 200));
    final cubit = RetrainCubit(
      client: RetrainClient(baseUrl: Uri.parse('http://localhost:9100'), client: mock),
      history: SessionHistoryStore(root: Directory.systemTemp),
    );
    await tester.pumpWidget(MaterialApp(
      home: Scaffold(
        body: BlocProvider.value(value: cubit, child: const SingleChildScrollView(child: RetrainSection())),
      ),
    ));
    await tester.runAsync(() => cubit.refresh());
    await tester.pump();
    expect(find.text('Install anyway'), findsOneWidget);
    expect(find.text('Keep current models'), findsOneWidget);
    expect(find.text('Clean files AUC'), findsOneWidget);
    expect(find.textContaining('Worse: song auc'), findsOneWidget);
    cubit.unwatch();
  });
}
