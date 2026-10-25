import 'package:equatable/equatable.dart';

import '../../../data/models/identification.dart';

/// One saved clip of the live session and, once ACRCloud is done, its result.
class ClipResult extends Equatable {
  const ClipResult({
    required this.index,
    required this.id,
    this.startS,
    this.durationS,
    this.identification,
  });

  final int index;
  final String id;
  final double? startS;
  final double? durationS;

  /// Null while identification is still running.
  final Identification? identification;

  ClipResult copyWith({
    double? startS,
    double? durationS,
    Identification? identification,
  }) => ClipResult(
    index: index,
    id: id,
    startS: startS ?? this.startS,
    durationS: durationS ?? this.durationS,
    identification: identification ?? this.identification,
  );

  @override
  List<Object?> get props => [index, id, startS, durationS, identification];
}

class ClipMetaData {
  final String prediction;
  final String id;
  final PredictionWindow predictionWindow;
  final Identification? identification;

  ClipMetaData({
    required this.prediction,
    required this.id,
    required this.predictionWindow,
    this.identification,
  });

  ClipMetaData copyWith({
    String? prediction,
    String? id,
    PredictionWindow? predictionWindow,
    Identification? identification,
  }) => ClipMetaData(
    prediction: prediction ?? this.prediction,
    id: id ?? this.id,
    predictionWindow: predictionWindow ?? this.predictionWindow,
    identification: identification ?? this.identification,
  );
}

class PredictionWindow {
  final String? window10;
  final String? window5;
  final String? delta;
  final String? bass;
  final String? mid;
  final String? treble;

  PredictionWindow({
    this.window10,
    this.window5,
    this.delta,
    this.bass,
    this.mid,
    this.treble,
  });

  PredictionWindow copyWith({
    String? window10,
    String? window5,
    String? delta,
    String? bass,
    String? mid,
    String? treble,
  }) => PredictionWindow(
    window10: window10 ?? this.window10,
    window5: window5 ?? this.window5,
    delta: delta ?? this.delta,
    bass: bass ?? this.bass,
    mid: mid ?? this.mid,
    treble: treble ?? this.treble,
  );
}
