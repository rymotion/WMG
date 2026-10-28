import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

import '../../recognizer/models/spectrogram.dart';
import '../../../data/labels/recording_label.dart';
import '../models/saved_session.dart';
import 'wav_writer.dart';

/// Finished live sessions on device, one folder each under the app's support directory:
/// - `history/<id>/session.json`: the [SavedSession]
/// - `history/<id>/spectrogram.bin`: [Spectrogram.encode] bytes
/// - `history/<id>/audio.wav`: mic audio, written while recording
class SessionHistoryStore {
  SessionHistoryStore({Directory? root}) : _root = root;

  Directory? _root;

  /// A folder-safe id from the start time, e.g. `20261026-012233-123`.
  static String newId([DateTime? at]) {
    final t = at ?? DateTime.now();
    String two(int v) => v.toString().padLeft(2, '0');
    return '${t.year}${two(t.month)}${two(t.day)}-${two(t.hour)}${two(t.minute)}${two(t.second)}'
        '-${t.millisecond.toString().padLeft(3, '0')}';
  }

  Future<Directory> _base() async => _root ??= Directory(
    '${(await getApplicationSupportDirectory()).path}/history',
  );

  Future<Directory> _dir(String id) async =>
      Directory('${(await _base()).path}/$id');

  Future<File> _file(String id, String name) async =>
      File('${(await _dir(id)).path}/$name');

  /// Starts the session's `audio.wav`; the bloc appends mic PCM as it arrives.
  Future<WavWriter> openAudio(String id, {required int sampleRate}) async =>
      WavWriter.open(await _file(id, 'audio.wav'), sampleRate: sampleRate);

  /// Writes (or rewrites, as late results arrive) the session and its spectrogram.
  Future<void> save(SavedSession session, Spectrogram spectrogram) async {
    final dir = await _dir(session.id);
    await dir.create(recursive: true);
    if (!spectrogram.isEmpty) {
      await File(
        '${dir.path}/spectrogram.bin',
      ).writeAsBytes(spectrogram.encode(), flush: true);
    }
    // Keep labels set from the saved-sessions page while the recognizer is still re-saving
    // this session with late results: the recording's, and each clip's.
    final existing = File('${dir.path}/session.json');
    if (await existing.exists()) {
      try {
        final old = SavedSession.fromJson(jsonDecode(await existing.readAsString()) as Map<String, dynamic>);
        if (session.label == null && old.label != null) session = session.withLabel(old.label);
        for (final c in old.clips) {
          final now = session.clips.where((x) => x.index == c.index).firstOrNull;
          if (c.label != null && now != null && now.label == null) {
            session = session.withClipLabel(c.index, c.label);
          }
        }
      } catch (_) {}
    }
    // Write-then-rename so a crash never leaves a half-written session.json.
    final tmp = File('${dir.path}/session.json.tmp');
    await tmp.writeAsString(jsonEncode(session.toJson()), flush: true);
    await tmp.rename('${dir.path}/session.json');
  }

  /// Sets (or clears) a saved session's label, keeping everything else, including its audio.
  Future<SavedSession> setLabel(String id, RecordingLabel? label) async {
    final f = await _file(id, 'session.json');
    final s = SavedSession.fromJson(jsonDecode(await f.readAsString()) as Map<String, dynamic>).withLabel(label);
    final tmp = File('${f.path}.tmp');
    await tmp.writeAsString(jsonEncode(s.toJson()), flush: true);
    await tmp.rename(f.path);
    return s;
  }

  /// Sets (or, with null, clears) one clip's own label; the clip otherwise takes the recording's.
  Future<SavedSession> setClipLabel(String id, int clipIndex, RecordingLabel? label) =>
      setClipLabels(id, {clipIndex}, label);

  /// [setClipLabel] for several clips at once, in one write.
  Future<SavedSession> setClipLabels(String id, Set<int> clipIndexes, RecordingLabel? label) async {
    final f = await _file(id, 'session.json');
    final s = SavedSession.fromJson(
      jsonDecode(await f.readAsString()) as Map<String, dynamic>,
    ).withClipLabels(clipIndexes, label);
    final tmp = File('${f.path}.tmp');
    await tmp.writeAsString(jsonEncode(s.toJson()), flush: true);
    await tmp.rename(f.path);
    return s;
  }

  /// Every saved session, newest first. Unreadable or unfinished folders are skipped.
  Future<List<SavedSession>> list() async {
    final base = await _base();
    if (!await base.exists()) return [];
    final sessions = <SavedSession>[];
    await for (final entry in base.list()) {
      if (entry is! Directory) continue;
      final f = File('${entry.path}/session.json');
      if (!await f.exists()) continue;
      try {
        sessions.add(
          SavedSession.fromJson(
            jsonDecode(await f.readAsString()) as Map<String, dynamic>,
          ),
        );
      } on Object catch (_) {
        // Corrupt or from an incompatible version; leave it on disk, don't list it.
      }
    }
    sessions.sort((a, b) => b.startedAt.compareTo(a.startedAt));
    return sessions;
  }

  Future<Spectrogram> loadSpectrogram(SavedSession s) async {
    final f = await _file(s.id, 'spectrogram.bin');
    if (!s.hasSpectrogram || !await f.exists()) {
      return Spectrogram(nMels: s.nMels, dbFloor: s.dbFloor, hopS: s.hopS);
    }
    return Spectrogram.decode(
      await f.readAsBytes(),
      nMels: s.nMels,
      dbFloor: s.dbFloor,
      hopS: s.hopS,
    );
  }

  /// The session's recording, or null if it wasn't kept.
  Future<File?> audioFile(SavedSession s) async {
    final f = await _file(s.id, 'audio.wav');
    return s.hasAudio && await f.exists() ? f : null;
  }

  /// Removes a session and its files.
  Future<void> delete(String id) async {
    final dir = await _dir(id);
    if (await dir.exists()) await dir.delete(recursive: true);
  }
}
