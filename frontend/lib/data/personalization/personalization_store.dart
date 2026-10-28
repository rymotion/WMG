import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

import 'personal_model.dart';

/// On-device storage under the app's support directory:
///   personalization/samples.jsonl   labelled windows (append-only)
///   personalization/head.json       the active fine-tuned layer
///   personalization/settings.json   toggles
class PersonalizationStore {
  PersonalizationStore({Directory? root}) : _root = root;

  Directory? _root;

  Future<Directory> _dir() async {
    final base = _root ??= Directory('${(await getApplicationSupportDirectory()).path}/personalization');
    if (!await base.exists()) await base.create(recursive: true);
    return base;
  }

  Future<File> _file(String name) async => File('${(await _dir()).path}/$name');

  Future<List<LabelledSample>> loadSamples() async {
    final f = await _file('samples.jsonl');
    if (!await f.exists()) return [];
    return [
      for (final line in await f.readAsLines())
        if (line.trim().isNotEmpty) LabelledSample.fromJson(jsonDecode(line) as Map<String, dynamic>),
    ];
  }

  Future<void> addSamples(List<LabelledSample> samples) async {
    final f = await _file('samples.jsonl');
    await f.writeAsString('${samples.map((s) => jsonEncode(s.toJson())).join('\n')}\n', mode: FileMode.append, flush: true);
  }

  Future<PersonalHead?> loadHead() async {
    final f = await _file('head.json');
    if (!await f.exists()) return null;
    return PersonalHead.fromJson(jsonDecode(await f.readAsString()) as Map<String, dynamic>);
  }

  Future<void> saveHead(PersonalHead? head) async {
    final f = await _file('head.json');
    if (head == null) {
      if (await f.exists()) await f.delete();
    } else {
      await f.writeAsString(jsonEncode(head.toJson()), flush: true);
    }
  }

  Future<Map<String, dynamic>> loadSettings() async {
    final f = await _file('settings.json');
    if (!await f.exists()) return {};
    return jsonDecode(await f.readAsString()) as Map<String, dynamic>;
  }

  Future<void> saveSettings(Map<String, dynamic> settings) async {
    await (await _file('settings.json')).writeAsString(jsonEncode(settings), flush: true);
  }

  /// Writes a `.wmgmodel` export into a temp file and returns it (for sharing).
  Future<File> writeExport(String json, String name) async {
    final f = File('${(await getTemporaryDirectory()).path}/$name');
    await f.writeAsString(json, flush: true);
    return f;
  }

  Future<void> clearAll() async {
    final d = await _dir();
    if (await d.exists()) await d.delete(recursive: true);
  }
}
