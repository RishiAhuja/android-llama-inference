import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';
import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;

// --- FFI Definitions ---
final class _Opaque extends Opaque {}

typedef _LoadModelFunc = Pointer<_Opaque> Function(Pointer<Utf8> modelPath);
typedef _PredictFunc = Pointer<Utf8> Function(
    Pointer<_Opaque> context, Pointer<Utf8> prompt);
typedef _FreeStringFunc = Void Function(Pointer<Utf8> str);
typedef _FreeModelFunc = Void Function(Pointer<_Opaque> context);

typedef _LoadModel = Pointer<_Opaque> Function(Pointer<Utf8> modelPath);
typedef _Predict = Pointer<Utf8> Function(
    Pointer<_Opaque> context, Pointer<Utf8> prompt);
typedef _FreeString = void Function(Pointer<Utf8> str);
typedef _FreeModel = void Function(Pointer<_Opaque> context);

// --- Isolate Functions (Pure Functions Only) ---
Map<String, dynamic> _loadModelIsolate(String modelPath) {
  try {
    final DynamicLibrary lib = Platform.isAndroid
        ? DynamicLibrary.open("libnative-lib.so")
        : DynamicLibrary.process();

    final loadModel = lib
        .lookup<NativeFunction<_LoadModelFunc>>('load_model')
        .asFunction<_LoadModel>();

    final modelPathC = modelPath.toNativeUtf8();
    final context = loadModel(modelPathC);
    calloc.free(modelPathC);

    return {
      'success': context.address != 0,
      'address': context.address,
      'error': context.address == 0 ? 'Failed to load model' : null,
    };
  } catch (e) {
    return {
      'success': false,
      'address': 0,
      'error': e.toString(),
    };
  }
}

String _runInferenceIsolate(Map<String, dynamic> args) {
  try {
    final DynamicLibrary lib = Platform.isAndroid
        ? DynamicLibrary.open("libnative-lib.so")
        : DynamicLibrary.process();

    final predict = lib
        .lookup<NativeFunction<_PredictFunc>>('predict')
        .asFunction<_Predict>();
    final freeString = lib
        .lookup<NativeFunction<_FreeStringFunc>>('free_string')
        .asFunction<_FreeString>();

    final modelAddress = args['modelAddress'] as int;
    final prompt = args['prompt'] as String;

    final modelContext = Pointer<_Opaque>.fromAddress(modelAddress);
    final promptC = prompt.toNativeUtf8();
    final resultPtr = predict(modelContext, promptC);
    calloc.free(promptC);

    final result = resultPtr.toDartString();
    freeString(resultPtr);

    return result;
  } catch (e) {
    return 'Error during inference: $e';
  }
}

// --- Model Status and Data ---
enum ModelStatus {
  notDownloaded,
  downloading,
  downloaded,
  loading,
  loaded,
  error
}

class ModelInfo {
  final String name;
  final String url;
  final String filename;
  final int estimatedSizeMB;

  const ModelInfo({
    required this.name,
    required this.url,
    required this.filename,
    required this.estimatedSizeMB,
  });
}

class ChatMessage {
  final String text;
  final bool isUser;

  ChatMessage({required this.text, required this.isUser});
}

// --- Model Manager (State Management) ---
class ModelManager extends ChangeNotifier {
  static const ModelInfo gemma3B = ModelInfo(
    name: 'Gemma 3B Instruct',
    url:
        'https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf',
    filename: 'gemma-3-1b-it-Q4_K_M.gguf',
    estimatedSizeMB: 800,
  );

  ModelStatus _status = ModelStatus.notDownloaded;
  double _downloadProgress = 0.0;
  String? _errorMessage;
  int _modelAddress = 0;
  File? _modelFile;

  ModelStatus get status => _status;
  double get downloadProgress => _downloadProgress;
  String? get errorMessage => _errorMessage;
  bool get isModelLoaded => _status == ModelStatus.loaded && _modelAddress != 0;

  Future<void> initialize() async {
    try {
      final appDir = await getApplicationDocumentsDirectory();
      _modelFile = File('${appDir.path}/${gemma3B.filename}');

      if (await _modelFile!.exists()) {
        _setStatus(ModelStatus.downloaded);
      } else {
        _setStatus(ModelStatus.notDownloaded);
      }
    } catch (e) {
      _setError('Failed to initialize: $e');
    }
  }

  Future<void> downloadModel() async {
    if (_status == ModelStatus.downloading) return;

    _setStatus(ModelStatus.downloading);
    _clearError();

    try {
      final request = http.Request('GET', Uri.parse(gemma3B.url));
      final response = await request.send();

      if (response.statusCode == 200) {
        final totalBytes = response.contentLength ?? 0;
        var downloadedBytes = 0;
        final bytes = <int>[];

        await for (final chunk in response.stream) {
          bytes.addAll(chunk);
          downloadedBytes += chunk.length;

          if (totalBytes > 0) {
            _downloadProgress = downloadedBytes / totalBytes;
            notifyListeners();
          }
        }

        await _modelFile!.writeAsBytes(bytes);
        _setStatus(ModelStatus.downloaded);
      } else {
        throw Exception('Download failed: HTTP ${response.statusCode}');
      }
    } catch (e) {
      _setError('Download failed: $e');
    }
  }

  Future<void> loadModel() async {
    if (_status != ModelStatus.downloaded || _modelFile == null) return;

    _setStatus(ModelStatus.loading);
    _clearError();

    try {
      final result =
          await Isolate.run(() => _loadModelIsolate(_modelFile!.path));

      if (result['success'] == true) {
        _modelAddress = result['address'] as int;
        _setStatus(ModelStatus.loaded);
      } else {
        throw Exception(result['error'] ?? 'Unknown error');
      }
    } catch (e) {
      _setError('Model loading failed: $e');
    }
  }

  Future<String> runInference(String prompt) async {
    if (!isModelLoaded) {
      throw Exception('Model not loaded');
    }

    return await Isolate.run(() => _runInferenceIsolate({
          'modelAddress': _modelAddress,
          'prompt': prompt,
        }));
  }

  Future<void> deleteModel() async {
    if (_modelFile != null && await _modelFile!.exists()) {
      await _modelFile!.delete();
    }
    _modelAddress = 0;
    _setStatus(ModelStatus.notDownloaded);
  }

  void _setStatus(ModelStatus newStatus) {
    _status = newStatus;
    if (newStatus != ModelStatus.downloading) {
      _downloadProgress = 0.0;
    }
    notifyListeners();
  }

  void _setError(String error) {
    _errorMessage = error;
    _status = ModelStatus.error;
    notifyListeners();
  }

  void _clearError() {
    _errorMessage = null;
    notifyListeners();
  }

  @override
  void dispose() {
    if (_modelAddress != 0) {
      // Clean up in background
      final address = _modelAddress;
      Isolate.run(() {
        try {
          final DynamicLibrary lib = Platform.isAndroid
              ? DynamicLibrary.open("libnative-lib.so")
              : DynamicLibrary.process();
          final freeModel = lib
              .lookup<NativeFunction<_FreeModelFunc>>('free_model')
              .asFunction<_FreeModel>();
          final modelContext = Pointer<_Opaque>.fromAddress(address);
          freeModel(modelContext);
        } catch (e) {
          // Ignore cleanup errors
        }
      });
    }
    super.dispose();
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
      title: 'Gemma 3B on Flutter',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
            seedColor: Colors.blueGrey, brightness: Brightness.dark),
        useMaterial3: true,
      ),
      home: const ModelManagerScreen(),
    );
  }
}

// --- Model Manager Screen ---
class ModelManagerScreen extends StatefulWidget {
  const ModelManagerScreen({super.key});

  @override
  State<ModelManagerScreen> createState() => _ModelManagerScreenState();
}

class _ModelManagerScreenState extends State<ModelManagerScreen> {
  late ModelManager _modelManager;

  @override
  void initState() {
    super.initState();
    _modelManager = ModelManager();
    _modelManager.addListener(_onModelManagerChanged);
    _modelManager.initialize();
  }

  @override
  void dispose() {
    _modelManager.removeListener(_onModelManagerChanged);
    _modelManager.dispose();
    super.dispose();
  }

  void _onModelManagerChanged() {
    if (mounted) setState(() {});
  }

  void _navigateToChat() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => ChatScreen(modelManager: _modelManager),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Gemma 3B Model Manager'),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              Expanded(
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Card(
                        child: Padding(
                          padding: const EdgeInsets.all(16.0),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                ModelManager.gemma3B.name,
                                style:
                                    Theme.of(context).textTheme.headlineSmall,
                              ),
                              const SizedBox(height: 8),
                              Text(
                                'Size: ~${ModelManager.gemma3B.estimatedSizeMB}MB',
                                style: Theme.of(context).textTheme.bodyMedium,
                              ),
                              const SizedBox(height: 16),
                              _buildStatusWidget(),
                              const SizedBox(height: 16),
                              _buildActionButtons(),
                            ],
                          ),
                        ),
                      ),
                      if (_modelManager.errorMessage != null) ...[
                        const SizedBox(height: 16),
                        Card(
                          color: Theme.of(context).colorScheme.errorContainer,
                          child: Padding(
                            padding: const EdgeInsets.all(16.0),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Error',
                                  style: TextStyle(
                                    color: Theme.of(context)
                                        .colorScheme
                                        .onErrorContainer,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  _modelManager.errorMessage!,
                                  style: TextStyle(
                                    color: Theme.of(context)
                                        .colorScheme
                                        .onErrorContainer,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
              if (_modelManager.status == ModelStatus.loaded) ...[
                const SizedBox(height: 16),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: _navigateToChat,
                    icon: const Icon(Icons.chat),
                    label: const Text('Start Chat'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.all(16),
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatusWidget() {
    switch (_modelManager.status) {
      case ModelStatus.notDownloaded:
        return const Text('Model not downloaded');
      case ModelStatus.downloading:
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
                'Downloading... ${(_modelManager.downloadProgress * 100).toInt()}%'),
            const SizedBox(height: 8),
            LinearProgressIndicator(value: _modelManager.downloadProgress),
          ],
        );
      case ModelStatus.downloaded:
        return const Text('Model downloaded, ready to load');
      case ModelStatus.loading:
        return const Row(
          children: [
            SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(strokeWidth: 2)),
            SizedBox(width: 8),
            Text('Loading model...'),
          ],
        );
      case ModelStatus.loaded:
        return const Row(
          children: [
            Icon(Icons.check_circle, color: Colors.green),
            SizedBox(width: 8),
            Text('Model loaded and ready'),
          ],
        );
      case ModelStatus.error:
        return const Text('Error - see details below');
    }
  }

  Widget _buildActionButtons() {
    switch (_modelManager.status) {
      case ModelStatus.notDownloaded:
        return SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: () => _modelManager.downloadModel(),
            icon: const Icon(Icons.download),
            label: const Text('Download Model'),
          ),
        );
      case ModelStatus.downloading:
        return SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: null,
            icon: const SizedBox(
              width: 16,
              height: 16,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
            label: const Text('Downloading...'),
          ),
        );
      case ModelStatus.downloaded:
        return Row(
          children: [
            Expanded(
              child: ElevatedButton.icon(
                onPressed: () => _modelManager.loadModel(),
                icon: const Icon(Icons.memory),
                label: const Text('Load Model'),
              ),
            ),
            const SizedBox(width: 8),
            ElevatedButton.icon(
              onPressed: () => _modelManager.deleteModel(),
              icon: const Icon(Icons.delete),
              label: const Text('Delete'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Theme.of(context).colorScheme.error,
                foregroundColor: Theme.of(context).colorScheme.onError,
              ),
            ),
          ],
        );
      case ModelStatus.loading:
        return SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: null,
            icon: const SizedBox(
              width: 16,
              height: 16,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
            label: const Text('Loading...'),
          ),
        );
      case ModelStatus.loaded:
        return Row(
          children: [
            Expanded(
              child: ElevatedButton.icon(
                onPressed: () => _modelManager.deleteModel(),
                icon: const Icon(Icons.delete),
                label: const Text('Unload & Delete'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Theme.of(context).colorScheme.error,
                  foregroundColor: Theme.of(context).colorScheme.onError,
                ),
              ),
            ),
          ],
        );
      case ModelStatus.error:
        return SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: () => _modelManager.initialize(),
            icon: const Icon(Icons.refresh),
            label: const Text('Retry'),
          ),
        );
    }
  }
}

// --- Chat Screen ---
class ChatScreen extends StatefulWidget {
  final ModelManager modelManager;

  const ChatScreen({super.key, required this.modelManager});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _promptController = TextEditingController();
  final List<ChatMessage> _messages = [];
  bool _isInferencing = false;
  final ScrollController _scrollController = ScrollController();

  @override
  void dispose() {
    _promptController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _runInference() async {
    if (!widget.modelManager.isModelLoaded ||
        _isInferencing ||
        _promptController.text.isEmpty) {
      return;
    }

    FocusScope.of(context).unfocus();
    final prompt = _promptController.text;
    _promptController.clear();

    setState(() {
      _messages.add(ChatMessage(text: prompt, isUser: true));
      _isInferencing = true;
    });
    _scrollToBottom();

    try {
      final response = await widget.modelManager.runInference(prompt);
      setState(() {
        _messages.add(ChatMessage(text: response, isUser: false));
      });
    } catch (e) {
      setState(() {
        _messages.add(
            ChatMessage(text: "Error during inference: $e", isUser: false));
      });
    } finally {
      setState(() {
        _isInferencing = false;
      });
      _scrollToBottom();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Chat with Gemma 3B'),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              Expanded(
                child: _messages.isEmpty
                    ? const Center(
                        child: Text(
                          'Start a conversation with Gemma 3B!',
                          style: TextStyle(fontSize: 16),
                        ),
                      )
                    : ListView.builder(
                        controller: _scrollController,
                        itemCount: _messages.length,
                        itemBuilder: (context, index) {
                          final message = _messages[index];
                          return _MessageBubble(message: message);
                        },
                      ),
              ),
              if (_isInferencing) ...[
                const SizedBox(height: 16),
                const Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                    SizedBox(width: 12),
                    Text("Gemma is thinking..."),
                  ],
                ),
              ],
              const SizedBox(height: 16),
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
                      maxLines: null,
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton(
                    icon: const Icon(Icons.send),
                    onPressed: _isInferencing ? null : _runInference,
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
      ),
    );
  }
}

class _MessageBubble extends StatelessWidget {
  final ChatMessage message;
  const _MessageBubble({required this.message});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Align(
      alignment: message.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4.0),
        padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 10.0),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.8,
        ),
        decoration: BoxDecoration(
          color: message.isUser
              ? theme.colorScheme.primary
              : theme.colorScheme.secondaryContainer,
          borderRadius: BorderRadius.circular(12.0),
        ),
        child: SelectableText(
          message.text,
          style: TextStyle(
            color: message.isUser
                ? theme.colorScheme.onPrimary
                : theme.colorScheme.onSecondaryContainer,
          ),
        ),
      ),
    );
  }
}
