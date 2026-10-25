import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:wmg_frontend/core/app_config.dart';
import 'package:wmg_frontend/data/api/spectro_api_client.dart';
import 'package:wmg_frontend/data/personalization/personalization_store.dart';
import 'package:wmg_frontend/features/personalization/cubit/personalization_cubit.dart';

class ResultsPage extends StatelessWidget {
  const ResultsPage({super.key});

  @override
  Widget build(BuildContext context) {
    final config = context.read<AppConfig>();

    return MultiBlocProvider(
      providers: [
        BlocProvider(
          create: (context) => PersonalizationCubit(
            store: PersonalizationStore(),
            detector: null,
            api: context.read<SpectroApiClient>(),
          ),
        ),
      ],
      child: const ResultsView(),
    );
  }
}

class ResultsView extends StatelessWidget {
  const ResultsView({super.key});

  @override
  Widget build(BuildContext context) {
    return const Scaffold(body: Center(child: Text('Results View')));
  }
}
