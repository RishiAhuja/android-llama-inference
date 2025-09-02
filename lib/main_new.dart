import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

// --- FFI Definitions ---

// Define a type for the opaque model pointer from C++
final class _Opaque extends Opaque {}

// C function signatures
typedef _LoadModelFunc = Pointer<_Opaque> Function(Pointer<Utf8> modelPath);
typedef _PredictFunc = Pointer<Utf8> Function(
    Pointer<_Opaque> context, Pointer<Utf8> prompt);
typedef _FreeStringFunc = Void Function(Pointer<Utf8> str);
typedef _FreeModelFunc = Void Function(Pointer<_Opaque> context);

// Dart function signatures
typedef _LoadModel = Pointer<_Opaque> Function(Pointer<Utf8> modelPath);
typedef _Predict = Pointer<Utf8> Function(
    Pointer<_Opaque> context, Pointer<Utf8> prompt);
typedef _FreeString = void Function(Pointer<Utf8> str);
typedef _FreeModel = void Function(Pointer<_Opaque> context);

// Helper class to load and bind the FFI functions
class LlamaFFI {
  late final DynamicLibrary _lib;
  late final _LoadModel loadModel;
  late final _Predict predict;
  late final _FreeString freeString;
  late final _FreeModel freeModel;

  LlamaFFI() {
    _lib = Platform.isAndroid
        ? DynamicLibrary.open("libnative-lib.so")
        : DynamicLibrary.process();

    loadModel = _lib
        .lookup<NativeFunction<_LoadModelFunc>>('load_model')
        .asFunction<_LoadModel>();
    predict = _lib
        .lookup<NativeFunction<_PredictFunc>>('predict')
        .asFunction<_Predict>();
    freeString = _lib
        .lookup<NativeFunction<_FreeStringFunc>>('free_string')
        .asFunction<_FreeString>();
    freeModel = _lib
        .lookup<NativeFunction<_FreeModelFunc>>('free_model')
        .asFunction<_FreeModel>();
  }
}

// High-level Llama model wrapper
class LlamaModel {
  final LlamaFFI _ffi = LlamaFFI();
  Pointer<_Opaque>? _modelPtr;
  bool _isLoaded = false;

  bool get isLoaded => _isLoaded;

  // Load model from file path
  bool loadModel(String modelPath) {
    if (_isLoaded) {
      dispose();
    }

    final pathPtr = modelPath.toNativeUtf8();
    try {
      _modelPtr = _ffi.loadModel(pathPtr);
      _isLoaded = _modelPtr != nullptr;
      return _isLoaded;
    } finally {
      malloc.free(pathPtr);
    }
  }

  // Generate text from prompt
  String predict(String prompt) {
    if (!_isLoaded || _modelPtr == null) {
      return "Model not loaded";
    }

    final promptPtr = prompt.toNativeUtf8();
    try {
      final resultPtr = _ffi.predict(_modelPtr!, promptPtr);
      if (resultPtr == nullptr) {
        return "Prediction failed";
      }

      final result = resultPtr.toDartString();
      _ffi.freeString(resultPtr);
      return result;
    } finally {
      malloc.free(promptPtr);
    }
  }

  // Clean up resources
  void dispose() {
    if (_modelPtr != null) {
      _ffi.freeModel(_modelPtr!);
      _modelPtr = null;
      _isLoaded = false;
    }
  }
}

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gemma On-Device',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const ChatScreen(),
    );
  }
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _promptController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final LlamaModel _model = LlamaModel();

  String _modelStatus = "Model not loaded";
  String _modelPath = "";
  List<Map<String, String>> _chatHistory = [];
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  @override
  void dispose() {
    _model.dispose();
    super.dispose();
  }

  // Copy model from assets to accessible location
  Future<String> _getModelPath() async {
    final docDir = await getApplicationDocumentsDirectory();
    final modelFile = File('${docDir.path}/gemma-3-1b-it-Q4_K_M.gguf');
    if (!await modelFile.exists()) {
      setState(() => _modelStatus = "Copying model from assets...");
      final byteData =
          await rootBundle.load('assets/models/gemma-3-1b-it-Q4_K_M.gguf');
      await modelFile.writeAsBytes(byteData.buffer.asUint8List());
    }
    return modelFile.path;
  }

  void _loadModel() async {
    setState(() {
      _isLoading = true;
      _modelStatus = "Loading model...";
    });
    
    try {
      _modelPath = await _getModelPath();
      final success = _model.loadModel(_modelPath);
      
      setState(() {
        if (success) {
          _modelStatus = "Model loaded successfully from $_modelPath";
        } else {
          _modelStatus = "Failed to load model";
        }
      });
    } catch (e) {
      setState(() {
        _modelStatus = "Failed to load model: $e";
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _sendMessage() async {
    if (_isLoading || !_model.isLoaded) return;

    final promptText = _promptController.text;
    if (promptText.isEmpty) return;

    setState(() {
      _isLoading = true;
      _chatHistory.add({'role': 'user', 'content': promptText});
      _promptController.clear();
    });
    _scrollToBottom();

    try {
      // Format prompt for Gemma
      final formattedPrompt =
          "<start_of_turn>user\n$promptText<end_of_turn>\n<start_of_turn>model\n";

      // Get prediction
      final response = _model.predict(formattedPrompt);

      setState(() {
        _chatHistory.add({'role': 'assistant', 'content': response.trim()});
        _isLoading = false;
      });
      _scrollToBottom();
    } catch (e) {
      setState(() {
        _chatHistory.add({'role': 'assistant', 'content': 'Error: $e'});
        _isLoading = false;
      });
      _scrollToBottom();
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.jumpTo(_scrollController.position.maxScrollExtent);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Gemma 3B On-Device'),
      ),
      body: Column(
        children: [
          // Status bar
          Container(
            padding: const EdgeInsets.all(8.0),
            color: Colors.grey.shade200,
            child: Text(
              _modelStatus,
              style: Theme.of(context).textTheme.bodySmall,
              textAlign: TextAlign.center,
            ),
          ),
          // Chat history
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              itemCount: _chatHistory.length,
              itemBuilder: (context, index) {
                final message = _chatHistory[index];
                final isUser = message['role'] == 'user';
                return Align(
                  alignment:
                      isUser ? Alignment.centerRight : Alignment.centerLeft,
                  child: Container(
                    margin:
                        const EdgeInsets.symmetric(vertical: 4, horizontal: 8),
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: isUser ? Colors.blue[100] : Colors.grey[300],
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Text(message['content']!),
                  ),
                );
              },
            ),
          ),
          // Loading indicator
          if (_isLoading)
            const Padding(
              padding: EdgeInsets.all(8.0),
              child: CircularProgressIndicator(),
            ),
          // Input area
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _promptController,
                    decoration: const InputDecoration(
                      hintText: 'Enter your prompt...',
                      border: OutlineInputBorder(),
                    ),
                    onSubmitted: (_) => _sendMessage(),
                  ),
                ),
                const SizedBox(width: 8),
                IconButton(
                  icon: const Icon(Icons.send),
                  onPressed: _sendMessage,
                  style: IconButton.styleFrom(
                    backgroundColor: Theme.of(context).colorScheme.primary,
                    foregroundColor: Theme.of(context).colorScheme.onPrimary,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
