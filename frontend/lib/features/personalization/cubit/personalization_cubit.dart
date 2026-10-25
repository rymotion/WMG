import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'dart:ui';

import 'package:equatable/equatable.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:wmg_detector/wmg_detector.dart';

import '../../../data/api/spectro_api_client.dart';
import '../../../data/personalization/head_trainer.dart';
import '../../../data/personalization/model_exporter.dart';
import '../../../data/personalization/personal_model.dart';
import '../../../data/personalization/personalization_store.dart';

part 'personalization_state.dart';

/// On-device personalisation: label recordings, re-fit the model's final layer on them, use the
/// result for scoring, and send it back to the computer (backend upload or a file).
class PersonalizationCubit extends Cubit<PersonalizationState> {
  PersonalizationCubit({
    required PersonalizationStore store,
    required WmgDetector? detector,
    required SpectroApiClient api,
    ModelExporter exporter = const ModelExporter(),
    HeadTrainer trainer = const HeadTrainer(),
  }) : _store = store,
       _detector = detector,
       _api = api,
       _exporter = exporter,
       _trainer = trainer,
       super(const PersonalizationState());

  final PersonalizationStore _store;
  final WmgDetector? _detector;
  final SpectroApiClient _api;
  final ModelExporter _exporter;
  final HeadTrainer _trainer;
  List<LabelledSample> _samples = [];

  Future<void> load() async {
    final detector = _detector;
    if (detector == null) {
      emit(
        state.copyWith(
          loaded: true,
          message: () => 'On-device personalisation needs iOS or macOS.',
        ),
      );
      return;
    }
    try {
      final base = await detector.info();
      final settings = await _store.loadSettings();
      _samples = await _store.loadSamples();
      final head = await _store.loadHead();
      emit(
        state.copyWith(
          loaded: true,
          base: base,
          head: () => head,
          trainingMode: settings['training_mode'] as bool? ?? false,
          usePersonal: settings['use_personal'] as bool? ?? false,
          offlineMode: settings['offline_mode'] as bool? ?? false,
          identifyMode: settings['identify_mode'] as bool? ?? false,
          counts: _countFor(base.modelRun),
        ),
      );
    } catch (e) {
      emit(
        state.copyWith(
          loaded: true,
          message: () => 'Could not load personalisation: $e',
        ),
      );
    }
  }

  SampleCounts _countFor(String modelRun) {
    final s = _samples.where((x) => x.modelRun == modelRun);
    return SampleCounts(
      aiWindows: s.where((x) => x.label == 1).length,
      humanWindows: s.where((x) => x.label == 0).length,
      aiRecordings: s
          .where((x) => x.label == 1)
          .map((x) => x.sessionId)
          .toSet()
          .length,
      humanRecordings: s
          .where((x) => x.label == 0)
          .map((x) => x.sessionId)
          .toSet()
          .length,
      otherModelWindows: _samples.length - s.length,
    );
  }

  Future<void> _saveSettings() => _store.saveSettings({
    'training_mode': state.trainingMode,
    'use_personal': state.usePersonal,
    'offline_mode': state.offlineMode,
    'identify_mode': state.identifyMode,
  });

  Future<void> setTrainingMode(bool on) async {
    emit(state.copyWith(trainingMode: on));
    await _saveSettings();
  }

  Future<void> setOfflineMode(bool on) async {
    emit(state.copyWith(offlineMode: on));
    await _saveSettings();
  }

  Future<void> setIdentifyMode(bool on) async {
    emit(state.copyWith(identifyMode: on));
    await _saveSettings();
  }

  Future<void> setUsePersonal(bool on) async {
    emit(state.copyWith(usePersonal: on && state.headUsable));
    await _saveSettings();
  }

  /// Stores the recording's windows (embeddings only) with the user's label.
  Future<void> labelRecording(
    String recordingId,
    int label,
    List<WindowScore> windows,
  ) async {
    final base = state.base;
    if (base == null || state.labelled.containsKey(recordingId)) return;
    final now = DateTime.now().toUtc();
    final samples = [
      for (final w in windows)
        if (w.embedding != null)
          LabelledSample(
            sessionId: recordingId,
            label: label,
            embedding: w.embedding!,
            modelRun: base.modelRun,
            recordedAt: now,
          ),
    ];
    if (samples.isEmpty) {
      emit(
        state.copyWith(
          message: () =>
              'Nothing to save: no scored windows in this recording.',
        ),
      );
      return;
    }
    await _store.addSamples(samples);
    _samples = [..._samples, ...samples];

    print('Saved ${samples.length} windows as ${label == 1 ? 'AI' : 'human'}');
    emit(
      state.copyWith(
        counts: _countFor(base.modelRun),
        labelled: {...state.labelled, recordingId: label},
        message: () =>
            'Saved ${samples.length} windows as ${label == 1 ? 'AI' : 'human'}.',
      ),
    );
  }

  /// Re-fits the final layer on this device's labelled windows (off the UI thread).
  Future<void> retrain() async {
    final base = state.base;
    final w0 = base?.headWeights;
    if (base == null || w0 == null) return;
    emit(state.copyWith(busy: () => 'Retraining…'));
    try {
      final samples = _samples;
      final trainer = _trainer;
      final head = await Isolate.run(
        () => trainer.train(
          samples: samples,
          w0: Float64List.fromList(w0),
          b0: base.headBias,
          baseModelRun: base.modelRun,
          featureSpecHash: base.modelSpecHash,
        ),
      );
      await _store.saveHead(head);
      final r = head.report;
      String pct(double? v) => v == null ? 'n/a' : '${(v * 100).round()}%';
      emit(
        state.copyWith(
          head: () => head,
          usePersonal: true,
          busy: () => null,
          message: () =>
              'Retrained on ${r.aiWindows} AI + ${r.humanWindows} human windows. '
              'Accuracy ${pct(r.accuracyBefore)} → ${pct(r.accuracyAfter)}'
              '${r.heldOutAccuracyAfter == null ? '' : '; on held-out recordings ${pct(r.heldOutAccuracyBefore)} → ${pct(r.heldOutAccuracyAfter)}'}.',
        ),
      );
      await _saveSettings();
    } on NotEnoughData catch (e) {
      emit(state.copyWith(busy: () => null, message: () => e.message));
    } catch (e) {
      emit(
        state.copyWith(
          busy: () => null,
          message: () => 'Retraining failed: $e',
        ),
      );
    }
  }

  String _bundle() {
    final head = state.head!;
    return encodeBundle(
      exportBundle(
        head,
        _samples,
        device:
            '${Platform.operatingSystem} ${Platform.operatingSystemVersion}',
      ),
    );
  }

  /// Sends the head + its labelled embeddings to the backend; falls back to a file if that fails.
  Future<void> sendToComputer({Rect? origin}) async {
    if (state.head == null) return;
    if (state.offlineMode) return exportFile(origin: origin);
    emit(state.copyWith(busy: () => 'Uploading…'));
    try {
      final res = await _api.uploadPersonalModel(_bundle());
      emit(
        state.copyWith(
          busy: () => null,
          message: () => 'Uploaded to the server as ${res['id']}.',
        ),
      );
    } catch (e) {
      emit(
        state.copyWith(
          busy: () => null,
          message: () => 'Upload failed ($e); saving a file instead.',
        ),
      );
      await exportFile(origin: origin);
    }
  }

  Future<void> exportFile({Rect? origin}) async {
    final head = state.head;
    if (head == null) return;
    try {
      final file = await _store.writeExport(
        _bundle(),
        'wmg-${head.baseModelRun}-${head.id}.wmgmodel',
      );
      final where = await _exporter.export(file, origin: origin);
      emit(
        state.copyWith(
          message: () =>
              where == null ? 'Export cancelled.' : 'Exported: $where',
        ),
      );
    } catch (e) {
      emit(state.copyWith(message: () => 'Export failed: $e'));
    }
  }

  Future<void> resetAll() async {
    await _store.clearAll();
    _samples = [];
    emit(
      state.copyWith(
        head: () => null,
        usePersonal: false,
        trainingMode: false,
        labelled: const {},
        counts: state.base == null
            ? const SampleCounts()
            : _countFor(state.base!.modelRun),
        message: () => 'Deleted labelled data and the personalised layer.',
      ),
    );
  }

  void clearMessage() => emit(state.copyWith(message: () => null));
}
