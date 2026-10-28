import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:wmg_frontend/data/labels/recording_label.dart';
import 'package:wmg_frontend/features/recognizer/models/clip_result.dart';
import 'package:wmg_frontend/features/recognizer/widgets/labelable_clip_grid.dart';

void main() {
  testWidgets('select clips, shift-click a range, label them together', (tester) async {
    final calls = <(Set<int>, AiCategory?)>[];
    // Newest first, as in the app.
    final clips = [for (var i = 5; i >= 0; i--) ClipResult(index: i, id: 'c$i')];
    await tester.binding.setSurfaceSize(const Size(1400, 2400));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    await tester.pumpWidget(MaterialApp(
      home: Scaffold(
        body: SingleChildScrollView(
          child: LabelableClipGrid(
            clips: clips,
            recordingLabel: RecordingLabel.now(AiCategory.human),
            onLabel: (ix, c) => calls.add((ix, c)),
          ),
        ),
      ),
    ));
    expect(find.text('Human-made · from recording', findRichText: true), findsNWidgets(6)); // every clip shows the recording's label

    await tester.tap(find.text('Select clips'));
    await tester.pump();
    await tester.tap(find.text('Clip 6')); // index 5
    await tester.pump();
    await tester.sendKeyDownEvent(LogicalKeyboardKey.shiftLeft);
    await tester.tap(find.text('Clip 3')); // index 2: range 5..2
    await tester.sendKeyUpEvent(LogicalKeyboardKey.shiftLeft);
    await tester.pump();
    await tester.tap(find.text('Clip 1')); // plain click adds index 0
    await tester.pump();
    expect(find.text('5 selected'), findsOneWidget);

    // Cards still identifying show a spinner, so pump for the menu animation instead of settling.
    await tester.tap(find.byTooltip('Label the selected clips'));
    await tester.pump();
    await tester.pump(const Duration(seconds: 1));
    await tester.tap(find.text('AI-generated').last);
    await tester.pump(const Duration(milliseconds: 500));
    expect(calls.single.$1, {5, 4, 3, 2, 0});
    expect(calls.single.$2, AiCategory.ai);
    expect(find.text('Select clips'), findsOneWidget); // back out of selecting

    await tester.tap(find.text('Select clips'));
    await tester.pump();
    await tester.tap(find.text('Select all'));
    await tester.pump();
    expect(find.text('6 selected'), findsOneWidget);
  });
}
