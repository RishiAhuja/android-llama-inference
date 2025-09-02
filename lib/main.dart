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
        : DynamicLibrary.process(); // Or handle other platforms

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

// --- Main App ---

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gemma on Flutter (Manual)',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
            seedColor: Colors.blueGrey, brightness: Brightness.dark),
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
  final LlamaFFI _llama = LlamaFFI();
  Pointer<_Opaque> _modelContext = Pointer.fromAddress(0);
  final TextEditingController _promptController = TextEditingController();
  String _modelResponse = "";
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  // --- Model Loading ---
  Future<void> _loadModel() async {
    setState(() {
      _isLoading = true;
      _modelResponse = "Copying model from assets...";
    });

    try {
      // Get a file path to the model by copying it from assets
      final modelPath =
          await _getAssetFile('assets/models/gemma-3-1b-it-Q4_K_M.gguf');

      setState(() {
        _modelResponse = "Loading model, this may take a moment...";
      });

      // Load the model using FFI
      // Use a separate isolate to avoid blocking the UI thread (recommended for real apps)
      await Future.delayed(
          const Duration(milliseconds: 100)); // Allow UI to update
      final modelPathC = modelPath.toNativeUtf8();
      _modelContext = _llama.loadModel(modelPathC);
      calloc.free(modelPathC);

      if (_modelContext.address == 0) {
        _modelResponse = "Error: Failed to load model (null context).";
      } else {
        _modelResponse = "Model loaded successfully! Ask me anything.";
      }
    } catch (e) {
      _modelResponse = "Error loading model: $e";
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  // Helper to copy an asset to a temporary file
  Future<String> _getAssetFile(String asset) async {
    final byteData = await rootBundle.load(asset);
    final tempDir = await getTemporaryDirectory();
    final file = File('${tempDir.path}/${asset.split('/').last}');
    await file.writeAsBytes(byteData.buffer
        .asUint8List(byteData.offsetInBytes, byteData.lengthInBytes));
    return file.path;
  }

  // --- Inference Logic ---
  Future<void> _runInference() async {
    if (_modelContext.address == 0 ||
        _isLoading ||
        _promptController.text.isEmpty) {
      return;
    }

    FocusScope.of(context).unfocus();

    setState(() {
      _isLoading = true;
      _modelResponse = "";
    });

    try {
      final prompt = _promptController.text;
      _promptController.clear();

      final promptC = prompt.toNativeUtf8();
      final resultPtr = _llama.predict(_modelContext, promptC);
      calloc.free(promptC);

      _modelResponse = resultPtr.toDartString();

      // IMPORTANT: Free the string allocated in C++
      _llama.freeString(resultPtr);
    } catch (e) {
      _modelResponse = "Error during inference: $e";
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  void dispose() {
    if (_modelContext.address != 0) {
      _llama.freeModel(_modelContext);
    }
    _promptController.dispose();
    super.dispose();
  }

  // --- UI Build Method (Identical to previous version) ---
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Gemma 3B on Flutter (Manual)'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Expanded(
              child: Container(
                padding: const EdgeInsets.all(12.0),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(8.0),
                ),
                child: SingleChildScrollView(
                  child: SelectableText(
                    _modelResponse,
                    style: const TextStyle(fontSize: 16.0),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 16),
            if (_isLoading)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 16.0),
                child: Column(
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 8),
                    Text("Processing..."),
                  ],
                ),
              ),
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _promptController,
                    decoration: InputDecoration(
                      hintText: 'Enter your prompt...',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8.0),
                      ),
                    ),
                    onSubmitted: (_) => _runInference(),
                  ),
                ),
                const SizedBox(width: 8),
                IconButton(
                  icon: const Icon(Icons.send),
                  onPressed: _isLoading || _modelContext.address == 0
                      ? null
                      : _runInference,
                  style: IconButton.styleFrom(
                    backgroundColor: Theme.of(context).colorScheme.primary,
                    foregroundColor: Theme.of(context).colorScheme.onPrimary,
                    padding: const EdgeInsets.all(16),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
