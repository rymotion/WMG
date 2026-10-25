part of 'personalization_cubit.dart';

final class SampleCounts extends Equatable {
  const SampleCounts({
    this.aiWindows = 0,
    this.humanWindows = 0,
    this.aiRecordings = 0,
    this.humanRecordings = 0,
    this.otherModelWindows = 0,
  });

  final int aiWindows;
  final int humanWindows;
  final int aiRecordings;
  final int humanRecordings;

  /// Labelled with a different base model; kept, but unusable for this model's layer.
  final int otherModelWindows;

  @override
  List<Object?> get props => [
    aiWindows,
    humanWindows,
    aiRecordings,
    humanRecordings,
    otherModelWindows,
  ];
}

final class PersonalizationState extends Equatable {
  const PersonalizationState({
    this.loaded = false,
    this.base,
    this.head,
    this.trainingMode = false,
    this.offlineMode = false,
    this.identifyMode = false,
    this.usePersonal = false,
    this.counts = const SampleCounts(),
    this.labelled = const {},
    this.busy,
    this.message,
  });

  final bool loaded;

  /// The bundled model (its run id and final-layer weights).
  final DetectorInfo? base;

  /// The on-device fine-tuned layer, if any.
  final PersonalHead? head;

  /// When on, each finished recording can be labelled human/AI for retraining.
  final bool trainingMode;
  final bool offlineMode;
  final bool identifyMode;
  final bool usePersonal;
  final SampleCounts counts;

  /// Recordings labelled in this app session: id -> label.
  final Map<String, int> labelled;
  final String? busy;
  final String? message;

  /// The head was fitted on this exact base model (embeddings differ between models).
  bool get headUsable =>
      head != null && base != null && head!.baseModelRun == base!.modelRun;

  /// The head scoring should use right now, or null for the base model.
  PersonalHead? get activeHead => usePersonal && headUsable ? head : null;

  PersonalizationState copyWith({
    bool? loaded,
    DetectorInfo? base,
    PersonalHead? Function()? head,
    bool? trainingMode,
    bool? offlineMode,
    bool? identifyMode,
    bool? usePersonal,
    SampleCounts? counts,
    Map<String, int>? labelled,
    String? Function()? busy,
    String? Function()? message,
  }) => PersonalizationState(
    loaded: loaded ?? this.loaded,
    base: base ?? this.base,
    head: head != null ? head() : this.head,
    trainingMode: trainingMode ?? this.trainingMode,
    offlineMode: offlineMode ?? this.offlineMode,
    identifyMode: identifyMode ?? this.identifyMode,
    usePersonal: usePersonal ?? this.usePersonal,
    counts: counts ?? this.counts,
    labelled: labelled ?? this.labelled,
    busy: busy != null ? busy() : this.busy,
    message: message != null ? message() : this.message,
  );

  @override
  List<Object?> get props => [
    loaded,
    base?.modelRun,
    head?.id,
    trainingMode,
    offlineMode,
    identifyMode,
    usePersonal,
    counts,
    labelled,
    busy,
    message,
  ];
}
