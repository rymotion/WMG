import 'package:flutter_test/flutter_test.dart';
import 'package:wmg_frontend/features/personalization/cubit/personalization_cubit.dart';

void main() {
  test('offline and identify mode count toward state equality, so toggles emit', () {
    const s = PersonalizationState();
    expect(s.copyWith(offlineMode: true), isNot(s));
    expect(s.copyWith(identifyMode: true), isNot(s));
    expect(s.copyWith(offlineMode: true).copyWith(offlineMode: false), s);
  });
}
