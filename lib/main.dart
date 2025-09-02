import 'package:flutter/material.dart';
import 'screens/model_management_screen.dart';
import 'screens/chat_screen.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gemma AI Assistant',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
            seedColor: Colors.blue, brightness: Brightness.dark),
        useMaterial3: true,
      ),
      initialRoute: '/models',
      onGenerateRoute: (settings) {
        switch (settings.name) {
          case '/models':
            return MaterialPageRoute(
              builder: (context) => const ModelManagementScreen(),
            );
          case '/chat':
            final args = settings.arguments as Map<String, dynamic>?;
            final modelId = args?['modelId'] as String?;
            return MaterialPageRoute(
              builder: (context) => ChatScreen(modelId: modelId),
            );
          default:
            return MaterialPageRoute(
              builder: (context) => const ModelManagementScreen(),
            );
        }
      },
    );
  }
}
