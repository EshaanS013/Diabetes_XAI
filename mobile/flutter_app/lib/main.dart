import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'screens/home_screen.dart';
import 'services/api_client.dart';
import 'state/screening_state.dart';

void main() {
  runApp(const DiabetesXaiApp());
}

class DiabetesXaiApp extends StatelessWidget {
  const DiabetesXaiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        Provider(create: (_) => ApiClient(baseUrl: const String.fromEnvironment(
          'API_BASE_URL',
          defaultValue: 'http://10.0.2.2:8000', // Android emulator → host localhost
        ))),
        ChangeNotifierProvider(create: (_) => ScreeningState()),
      ],
      child: MaterialApp(
        title: 'Diabetes XAI',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(
            seedColor: const Color(0xFF0F6E56),
            brightness: Brightness.light,
          ),
          useMaterial3: true,
        ),
        home: const HomeScreen(),
      ),
    );
  }
}
