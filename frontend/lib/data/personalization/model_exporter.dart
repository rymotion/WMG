import 'dart:io';
import 'dart:ui';

import 'package:file_selector/file_selector.dart';
import 'package:share_plus/share_plus.dart';

/// Gets a `.wmgmodel` file off the device without a server: the share sheet on iOS (AirDrop,
/// Save to Files, ...) and a Save dialog on macOS.
class ModelExporter {
  const ModelExporter();

  /// Returns a short description of where the file went, or null if the user cancelled.
  Future<String?> export(File file, {Rect? origin}) async {
    final name = file.uri.pathSegments.last;
    if (Platform.isMacOS) {
      final location = await getSaveLocation(
        suggestedName: name,
        acceptedTypeGroups: const [XTypeGroup(label: 'WMG model', extensions: ['wmgmodel'])],
      );
      if (location == null) return null;
      await file.copy(location.path);
      return location.path;
    }
    final result = await SharePlus.instance.share(ShareParams(files: [XFile(file.path)], sharePositionOrigin: origin));
    return result.status == ShareResultStatus.dismissed ? null : 'shared $name';
  }
}
