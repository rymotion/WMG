import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import 'core/app_config.dart';
import 'data/api/live_audio_client.dart';
import 'data/api/spectro_api_client.dart';
import 'data/recorder/audio_recorder_service.dart';
import 'data/repository/identification_repository.dart';
import 'features/recognizer/view/recognizer_page.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  if (kDebugMode) Bloc.observer = const _LoggingBlocObserver();

  final config = AppConfig.fromEnvironment();
  final api = SpectroApiClient(baseUrl: config.apiBaseUrl);

  runApp(
    MultiRepositoryProvider(
      providers: [
        RepositoryProvider.value(value: config),
        RepositoryProvider(create: (_) => AudioRecorderService(), dispose: (recorder) => recorder.dispose()),
        RepositoryProvider(
          create: (_) => IdentificationRepository(
            api: api,
            live: LiveAudioClient(uri: config.liveAudioUri, sampleRate: config.liveSampleRate),
          ),
          dispose: (_) => api.close(),
        ),
      ],
      child: const App(),
    ),
  );
}

class App extends StatelessWidget {
  const App({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Audio Identifier',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      darkTheme: ThemeData(colorSchemeSeed: Colors.indigo, brightness: Brightness.dark, useMaterial3: true),
      home: const RecognizerPage(),
    );
  }
}

/// Logs every event and transition flowing through the BLoC event loop.
class _LoggingBlocObserver extends BlocObserver {
  const _LoggingBlocObserver();

  @override
  void onEvent(Bloc<dynamic, dynamic> bloc, Object? event) {
    super.onEvent(bloc, event);
    // Level/tick events fire ~20×/s; keep the log readable.
    final name = event.runtimeType.toString();
    if (name != '_PcmChunkSent' && name != '_RecordingTicked') debugPrint('[${bloc.runtimeType}] event: $name');
  }

  @override
  void onError(BlocBase<dynamic> bloc, Object error, StackTrace stackTrace) {
    debugPrint('[${bloc.runtimeType}] error: $error');
    super.onError(bloc, error, stackTrace);
  }
}
