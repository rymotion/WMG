import 'package:equatable/equatable.dart';

import '../../../data/models/identification.dart';

/// One saved clip of the live session and, once ACRCloud is done, its result.
class ClipResult extends Equatable {
  const ClipResult({required this.index, required this.id, this.startS, this.durationS, this.identification});

  final int index;
  final String id;
  final double? startS;
  final double? durationS;

  /// Null while identification is still running.
  final Identification? identification;

  ClipResult copyWith({double? startS, double? durationS, Identification? identification}) => ClipResult(
    index: index,
    id: id,
    startS: startS ?? this.startS,
    durationS: durationS ?? this.durationS,
    identification: identification ?? this.identification,
  );

  @override
  List<Object?> get props => [index, id, startS, durationS, identification];
}
