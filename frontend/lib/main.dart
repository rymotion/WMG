import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import 'package:url_launcher/url_launcher.dart';

import 'core/app_config.dart';
import 'data/account/account_api.dart';
import 'data/account/fake_account_api.dart';
import 'data/api/live_audio_client.dart';
import 'data/auth/auth_session.dart';
import 'data/billing/store_billing.dart';
import 'data/api/spectro_api_client.dart';
import 'data/recorder/audio_recorder_service.dart';
import 'data/repository/identification_repository.dart';
import 'data/usage/network_meter.dart';
import 'features/account/cubit/account_cubit.dart';
import 'features/recognizer/view/recognizer_page.dart';
import 'features/results/data/session_history_store.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  if (kDebugMode) Bloc.observer = const _LoggingBlocObserver();

  final config = AppConfig.fromEnvironment();
  final meter = NetworkMeter();
  final api = SpectroApiClient(baseUrl: config.apiBaseUrl, httpClient: MeteredHttpClient(meter));
  // TODO(auth): swap in the chosen provider's AuthSession.
  final AuthSession auth = config.fakeAccountBackend ? DevAuthSession() : const UnavailableAuthSession();
  final AccountApi accounts = config.fakeAccountBackend
      ? FakeAccountApi(auth: auth)
      : HttpAccountApi(baseUrl: config.apiBaseUrl, auth: auth, httpClient: MeteredHttpClient(meter));

  runApp(
    MultiRepositoryProvider(
      providers: [
        RepositoryProvider.value(value: config),
        RepositoryProvider.value(value: api),
        RepositoryProvider.value(value: meter),
        RepositoryProvider(create: (_) => SessionHistoryStore()),
        RepositoryProvider(
          create: (_) => AudioRecorderService(),
          dispose: (recorder) => recorder.dispose(),
        ),
        RepositoryProvider(
          create: (_) => IdentificationRepository(
            api: api,
            live: LiveAudioClient(
              uri: config.liveAudioUri,
              sampleRate: config.liveSampleRate,
            ),
          ),
          dispose: (_) => api.close(),
        ),
      ],
      child: BlocProvider(
        create: (_) => AccountCubit(
          auth: auth,
          api: accounts,
          meter: meter,
          openUrl: (url) => launchUrl(url, mode: LaunchMode.externalApplication),
          // In the fake backend, store purchases still need App Store Connect products.
          store: config.billingEnabled ? InAppStoreBilling() : null,
        )..start(),
        child: const App(),
      ),
    ),
  );
}

class App extends StatefulWidget {
  const App({super.key});

  @override
  State<App> createState() => _AppState();
}

class _AppState extends State<App> {
  late final AppLifecycleListener _lifecycle;

  @override
  void initState() {
    super.initState();
    // Back from Stripe Checkout or the store's subscription settings: pick up the new plan.
    _lifecycle = AppLifecycleListener(onResume: () => context.read<AccountCubit>().refresh());
  }

  @override
  void dispose() {
    _lifecycle.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Audio Detector',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      darkTheme: ThemeData(
        colorSchemeSeed: Colors.indigo,
        brightness: Brightness.dark,
        useMaterial3: true,
      ),
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
    if (name != '_PcmChunkSent' && name != '_RecordingTicked')
      debugPrint('[${bloc.runtimeType}] event: $name');
  }

  @override
  void onError(BlocBase<dynamic> bloc, Object error, StackTrace stackTrace) {
    debugPrint('[${bloc.runtimeType}] error: $error');
    super.onError(bloc, error, stackTrace);
  }
}
