import 'dart:io' show Platform;

/// Runtime configuration. Override at build time:
///
///   flutter run --dart-define=API_BASE_URL=http://192.168.1.20:8010 --dart-define=LIVE_IDENTIFY=false
///     [--dart-define=LIVE_SAMPLE_RATE=16000]
class AppConfig {
  const AppConfig({
    required this.apiBaseUrl,
    this.liveSampleRate = 32000,
    this.liveIdentify = true,
    this.liveMaxDuration = const Duration(minutes: 60),
    this.finishTimeout = const Duration(seconds: 20),
  });

  factory AppConfig.fromEnvironment() {
    const fromDefine = String.fromEnvironment('API_BASE_URL');
    const identify = bool.fromEnvironment('LIVE_IDENTIFY', defaultValue: true);
    const sampleRate = int.fromEnvironment('LIVE_SAMPLE_RATE', defaultValue: 32000);
    if (fromDefine.isNotEmpty)
      return AppConfig(
        apiBaseUrl: Uri.parse(fromDefine),
        liveSampleRate: sampleRate,
        liveIdentify: identify,
      );
    // The Android emulator reaches the host machine through 10.0.2.2.
    final host = Platform.isAndroid ? '10.0.2.2' : 'localhost';
    return AppConfig(
      apiBaseUrl: Uri.parse('http://localhost:9000'),
      liveSampleRate: sampleRate,
      liveIdentify: identify,
    );
  }

  final Uri apiBaseUrl;

  /// PCM is sent as mono s16le at this rate (~64 KB/s at 32 kHz). The AI-music detector's
  /// features run at 32 kHz up to 16 kHz; a 16 kHz stream would leave its treble band empty.
  final int liveSampleRate;

  /// `false` adds `identify=false`, so the server skips ACRCloud for the session.
  final bool liveIdentify;

  /// Matches the server's default `LIVE_MAX_MINUTES`; the client stops itself at this point.
  final Duration liveMaxDuration;

  /// After `{"type":"stop"}`, how long to wait for the server's final messages and close.
  final Duration finishTimeout;

  /// `ws(s)://<host>:<port>/live?sample_rate=32000[&identify=false]`
  Uri get liveAudioUri => apiBaseUrl.replace(
    scheme: apiBaseUrl.scheme == 'https' ? 'wss' : 'ws',
    path: '/live',
    queryParameters: {
      'sample_rate': '$liveSampleRate',
      if (!liveIdentify) 'identify': 'false',
    },
  );
}
