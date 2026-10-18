import 'dart:io' show Platform;

/// Runtime configuration. Override at build time:
///
///   flutter run --dart-define=API_BASE_URL=http://192.168.1.20:8010 --dart-define=LIVE_IDENTIFY=false
class AppConfig {
  const AppConfig({
    required this.apiBaseUrl,
    this.liveSampleRate = 16000,
    this.liveIdentify = true,
    this.liveMaxDuration = const Duration(minutes: 60),
    this.finishTimeout = const Duration(seconds: 20),
  });

  factory AppConfig.fromEnvironment() {
    const fromDefine = String.fromEnvironment('API_BASE_URL');
    const identify = bool.fromEnvironment('LIVE_IDENTIFY', defaultValue: true);
    if (fromDefine.isNotEmpty)
      return AppConfig(
        apiBaseUrl: Uri.parse(fromDefine),
        liveIdentify: identify,
      );
    // The Android emulator reaches the host machine through 10.0.2.2.
    final host = Platform.isAndroid ? '10.0.2.2' : 'localhost';
    return AppConfig(
      apiBaseUrl: Uri.parse('http://localhost:9000'),
      liveIdentify: identify,
    );
  }

  final Uri apiBaseUrl;

  /// PCM is sent as mono s16le at this rate. 16 kHz is the server's recommendation (~32 KB/s).
  final int liveSampleRate;

  /// `false` adds `identify=false`, so the server skips ACRCloud for the session.
  final bool liveIdentify;

  /// Matches the server's default `LIVE_MAX_MINUTES`; the client stops itself at this point.
  final Duration liveMaxDuration;

  /// After `{"type":"stop"}`, how long to wait for the server's final messages and close.
  final Duration finishTimeout;

  /// `ws(s)://<host>:<port>/live?sample_rate=16000[&identify=false]`
  Uri get liveAudioUri => apiBaseUrl.replace(
    scheme: apiBaseUrl.scheme == 'https' ? 'wss' : 'ws',
    path: '/live',
    queryParameters: {
      'sample_rate': '$liveSampleRate',
      if (!liveIdentify) 'identify': 'false',
    },
  );
}
