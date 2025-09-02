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
      routes: {
        '/models': (context) => const ModelManagementScreen(),
        '/chat': (context) => const ChatScreen(),
      },
    );
  }
}
