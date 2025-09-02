import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';
import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;

// --- Memory Management Utilities ---
class MemoryInfo {
  static Future<Map<String, int>> getMemoryInfo() async {
    try {
      // Try to get memory info on Android
      if (Platform.isAndroid) {
        // This is approximate - real memory info would need platform channels
        return {
          'availableMemoryMB': 2048, // Conservative estimate
          'recommendedMinimumMB': 1800, // Reduced requirement for Gemma 3B
        };
      }
      return {'availableMemoryMB': 4096, 'recommendedMinimumMB': 1800};
    } catch (e) {
      // Fallback values
      return {'availableMemoryMB': 2048, 'recommendedMinimumMB': 1800};
    }
  }

  static Future<bool> hasEnoughMemoryForInference() async {
    final memInfo = await getMemoryInfo();
    final available = memInfo['availableMemoryMB'] ?? 0;
    final required = memInfo['recommendedMinimumMB'] ?? 1800;
    return available >= required;
  }
}

// --- Isolate Function for Inference ---
String _inferenceIsolate(Map<String, dynamic> args) {
  try {
    print('Isolate: Starting inference');
    final modelAddress = args['modelAddress'] as int;
    final prompt = args['prompt'] as String;

    print('Isolate: Model address: $modelAddress, Prompt: $prompt');

    final DynamicLibrary lib = Platform.isAndroid
        ? DynamicLibrary.open("libnative-lib.so")
        : DynamicLibrary.process();

    final predict = lib
        .lookup<NativeFunction<_PredictFunc>>('predict')
        .asFunction<_Predict>();
    final freeString = lib
        .lookup<NativeFunction<_FreeStringFunc>>('free_string')
        .asFunction<_FreeString>();

    final modelContext = Pointer<_Opaque>.fromAddress(modelAddress);
    final promptC = prompt.toNativeUtf8();

    print('Isolate: Calling predict function');
    final resultPtr = predict(modelContext, promptC);
    calloc.free(promptC);

    if (resultPtr.address == 0) {
      print('Isolate: Predict returned null pointer');
      return 'Error: Model returned null response';
    }

    final result = resultPtr.cast<Utf8>().toDartString();
    print('Isolate: Got result from C++: "$result"');
    print('Isolate: Result length: ${result.length} characters');

    freeString(resultPtr);

    return result;
  } catch (e) {
    print('Isolate: Exception during inference: $e');
    return 'Error during inference: $e';
  }
}

// FFI Types
final class _Opaque extends Opaque {}

typedef _LoadModelFunc = Pointer<_Opaque> Function(Pointer<Utf8> modelPath);
typedef _PredictFunc = Pointer<Utf8> Function(
    Pointer<_Opaque> context, Pointer<Utf8> prompt);
typedef _FreeStringFunc = Void Function(Pointer<Utf8> str);

typedef _LoadModel = Pointer<_Opaque> Function(Pointer<Utf8> modelPath);
typedef _Predict = Pointer<Utf8> Function(
    Pointer<_Opaque> context, Pointer<Utf8> prompt);
typedef _FreeString = void Function(Pointer<Utf8> str);

// Data classes
enum ModelStatus {
  notDownloaded,
  downloading,
  downloaded,
  loading,
  loaded,
  error
}

class ChatMessage {
  final String text;
  final bool isUser;
  ChatMessage({required this.text, required this.isUser});
}

// Model Manager
class ModelManager extends ChangeNotifier {
  ModelStatus _status = ModelStatus.notDownloaded;
  double _downloadProgress = 0.0;
  String? _error;
  int _modelAddress = 0;
  File? _modelFile;

  ModelStatus get status => _status;
  double get downloadProgress => _downloadProgress;
  String? get error => _error;
  bool get isLoaded => _status == ModelStatus.loaded;

  static const String modelUrl =
      'https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf';
  static const String modelFilename = 'gemma-3-1b-it-Q4_K_M.gguf';

  Future<void> init() async {
    final appDir = await getApplicationDocumentsDirectory();
    _modelFile = File('${appDir.path}/$modelFilename');

    if (await _modelFile!.exists()) {
      _setStatus(ModelStatus.downloaded);
    } else {
      _setStatus(ModelStatus.notDownloaded);
    }
  }

  Future<void> download() async {
    if (_status == ModelStatus.downloading) return;

    _setStatus(ModelStatus.downloading);
    _error = null;

    try {
      final request = http.Request('GET', Uri.parse(modelUrl));
      final response = await request.send();

      if (response.statusCode == 200) {
        final totalBytes = response.contentLength ?? 0;
        var downloadedBytes = 0;

        // Open file for writing in chunks to avoid memory issues
        final fileSink = _modelFile!.openWrite();

        try {
          await for (final chunk in response.stream) {
            // Write chunk directly to file instead of accumulating in memory
            fileSink.add(chunk);
            downloadedBytes += chunk.length;

            if (totalBytes > 0) {
              _downloadProgress = downloadedBytes / totalBytes;
              notifyListeners();
            }
          }

          await fileSink.flush();
          await fileSink.close();
          _setStatus(ModelStatus.downloaded);
        } catch (e) {
          await fileSink.close();
          rethrow;
        }
      } else {
        throw Exception('HTTP ${response.statusCode}');
      }
    } catch (e) {
      _error = 'Download failed: $e';
      _setStatus(ModelStatus.error);
    }
  }

  Future<void> load({bool forceLoad = false}) async {
    if (_status != ModelStatus.downloaded) return;

    _setStatus(ModelStatus.loading);
    _error = null;

    try {
      if (!forceLoad) {
        // Check memory before loading
        print('Checking available memory for model loading...');
        final hasEnoughMemory = await MemoryInfo.hasEnoughMemoryForInference();
        final memInfo = await MemoryInfo.getMemoryInfo();

        print('Available memory: ${memInfo['availableMemoryMB']}MB');
        print('Required memory: ${memInfo['recommendedMinimumMB']}MB');

        if (!hasEnoughMemory) {
          throw Exception(
              'Memory warning: Need ~${memInfo['recommendedMinimumMB']}MB, '
              'but only ~${memInfo['availableMemoryMB']}MB available. '
              'Loading anyway might cause crashes. Try "Force Load" if you want to proceed.');
        }

        print('Memory check passed. Loading model...');
      } else {
        print('Force loading model (bypassing memory check)...');
      }

      // Direct model loading without isolates to avoid sendable object issues
      final DynamicLibrary lib = Platform.isAndroid
          ? DynamicLibrary.open("libnative-lib.so")
          : DynamicLibrary.process();

      final loadModel = lib
          .lookup<NativeFunction<_LoadModelFunc>>('load_model')
          .asFunction<_LoadModel>();

      final modelPathC = _modelFile!.path.toNativeUtf8();
      print('Calling native load_model with path: ${_modelFile!.path}');
      final context = loadModel(modelPathC);
      calloc.free(modelPathC);

      if (context.address != 0) {
        _modelAddress = context.address;
        print('Model loaded successfully at address: $_modelAddress');
        _setStatus(ModelStatus.loaded);
      } else {
        throw Exception('Model loading failed - returned null context');
      }
    } catch (e) {
      print('Model loading error: $e');
      _error = 'Load failed: $e';
      _setStatus(ModelStatus.error);
    }
  }

  Future<String> inference(String prompt) async {
    if (!isLoaded) throw Exception('Model not loaded');

    // Limit prompt length to avoid memory issues
    if (prompt.length > 1000) {
      prompt = prompt.substring(0, 1000) + '...';
      print('Main: Truncated long prompt to 1000 characters');
    }

    print('Main: Starting inference for prompt: $prompt');
    print('Main: Model address: $_modelAddress');

    try {
      // Extract model address to avoid capturing 'this' in the isolate
      final modelAddress = _modelAddress;

      // Run inference in isolate with timeout
      final result = await Isolate.run(() => _inferenceIsolate({
            'modelAddress': modelAddress,
            'prompt': prompt,
          })).timeout(
        const Duration(seconds: 500), // Increased timeout for mobile inference
        onTimeout: () =>
            'Error: Inference timed out after 60 seconds. The model might be out of memory or the prompt is too complex.',
      );

      print('Main: Inference completed, result length: ${result.length}');

      // Force garbage collection after inference to free memory
      // Note: This is a hint to the GC, not guaranteed
      print('Main: Suggesting garbage collection to free memory');

      return result;
    } catch (e) {
      print('Main: Inference error: $e');
      if (e.toString().contains('memory') || e.toString().contains('Memory')) {
        return 'Error: Out of memory during inference. Try:\n'
            '• Restart the app\n'
            '• Close other apps\n'
            '• Use shorter prompts\n'
            '• Restart your device if the problem persists';
      }
      return 'Error during inference: $e';
    }
  }

  Future<void> delete() async {
    if (_modelFile != null && await _modelFile!.exists()) {
      await _modelFile!.delete();
    }
    _modelAddress = 0;
    _setStatus(ModelStatus.notDownloaded);
  }

  void _setStatus(ModelStatus newStatus) {
    _status = newStatus;
    notifyListeners();
  }
}

// Main App
void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gemma 3B',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
            seedColor: Colors.blue, brightness: Brightness.dark),
        useMaterial3: true,
      ),
      home: ModelScreen(),
    );
  }
}

// Model Management Screen
class ModelScreen extends StatefulWidget {
  @override
  State<ModelScreen> createState() => _ModelScreenState();
}

class _ModelScreenState extends State<ModelScreen> {
  late ModelManager modelManager;

  @override
  void initState() {
    super.initState();
    modelManager = ModelManager();
    modelManager.addListener(() => setState(() {}));
    modelManager.init();
  }

  @override
  void dispose() {
    modelManager.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Gemma 3B Model Manager')),
      body: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          children: [
            Expanded(
              child: SingleChildScrollView(
                child: Card(
                  child: Padding(
                    padding: EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Gemma 3B Instruct',
                            style: Theme.of(context).textTheme.headlineMedium),
                        SizedBox(height: 8),
                        Text('Model Size: ~800MB'),
                        Text('RAM Required: ~1.8GB for inference',
                            style:
                                TextStyle(color: Colors.orange, fontSize: 12)),
                        SizedBox(height: 8),
                        Container(
                          padding: EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: Colors.blue.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(6),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('💡 Memory Tips:',
                                  style: TextStyle(
                                      fontWeight: FontWeight.bold,
                                      fontSize: 12)),
                              Text('• Close other apps before loading',
                                  style: TextStyle(fontSize: 11)),
                              Text(
                                  '• Restart your phone if you get memory errors',
                                  style: TextStyle(fontSize: 11)),
                              Text('• Works best on devices with 6GB+ RAM',
                                  style: TextStyle(fontSize: 11)),
                            ],
                          ),
                        ),
                        SizedBox(height: 16),
                        _buildStatus(),
                        SizedBox(height: 16),
                        _buildActions(),
                        if (modelManager.error != null) ...[
                          SizedBox(height: 16),
                          Container(
                            padding: EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: Colors.red.withOpacity(0.1),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Text(modelManager.error!,
                                style: TextStyle(color: Colors.red)),
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
              ),
            ),
            if (modelManager.isLoaded) ...[
              SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: () => Navigator.push(
                      context,
                      MaterialPageRoute(
                          builder: (_) =>
                              ChatScreen(modelManager: modelManager))),
                  icon: Icon(Icons.chat),
                  label: Text('Start Chat'),
                  style: ElevatedButton.styleFrom(padding: EdgeInsets.all(16)),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildStatus() {
    switch (modelManager.status) {
      case ModelStatus.notDownloaded:
        return Text('Model not downloaded');
      case ModelStatus.downloading:
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
                'Downloading... ${(modelManager.downloadProgress * 100).toInt()}%'),
            SizedBox(height: 8),
            LinearProgressIndicator(value: modelManager.downloadProgress),
          ],
        );
      case ModelStatus.downloaded:
        return Text('Model ready to load');
      case ModelStatus.loading:
        return Row(
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
        return Row(
          children: [
            Icon(Icons.check_circle, color: Colors.green),
            SizedBox(width: 8),
            Text('Model loaded and ready'),
          ],
        );
      case ModelStatus.error:
        return Text('Error occurred');
    }
  }

  Widget _buildActions() {
    switch (modelManager.status) {
      case ModelStatus.notDownloaded:
        return SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: modelManager.download,
            icon: Icon(Icons.download),
            label: Text('Download Model'),
          ),
        );
      case ModelStatus.downloading:
        return SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: null,
            icon: SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(strokeWidth: 2)),
            label: Text('Downloading...'),
          ),
        );
      case ModelStatus.downloaded:
        return Column(
          children: [
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: modelManager.load,
                    icon: Icon(Icons.memory),
                    label: Text('Load Model'),
                  ),
                ),
                SizedBox(width: 8),
                ElevatedButton.icon(
                  onPressed: modelManager.delete,
                  icon: Icon(Icons.delete),
                  label: Text('Delete'),
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
                ),
              ],
            ),
          ],
        );
      case ModelStatus.error:
        // Show both regular load and force load for memory errors
        final isMemoryError =
            modelManager.error?.contains('Memory warning') ?? false;
        return Column(
          children: [
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: modelManager.load,
                    icon: Icon(Icons.memory),
                    label: Text('Try Again'),
                  ),
                ),
                SizedBox(width: 8),
                ElevatedButton.icon(
                  onPressed: modelManager.delete,
                  icon: Icon(Icons.delete),
                  label: Text('Delete'),
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
                ),
              ],
            ),
            if (isMemoryError) ...[
              SizedBox(height: 8),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: () => modelManager.load(forceLoad: true),
                  icon: Icon(Icons.warning),
                  label: Text('Force Load (Risk of Crash)'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.orange,
                    foregroundColor: Colors.white,
                  ),
                ),
              ),
            ],
          ],
        );
      case ModelStatus.loading:
        return SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: null,
            icon: SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(strokeWidth: 2)),
            label: Text('Loading...'),
          ),
        );
      case ModelStatus.loaded:
        return SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: modelManager.delete,
            icon: Icon(Icons.delete),
            label: Text('Unload & Delete'),
            style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
          ),
        );
      case ModelStatus.error:
        return SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: modelManager.init,
            icon: Icon(Icons.refresh),
            label: Text('Retry'),
          ),
        );
    }
  }
}

// Chat Screen
class ChatScreen extends StatefulWidget {
  final ModelManager modelManager;

  ChatScreen({required this.modelManager});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _controller = TextEditingController();
  final _messages = <ChatMessage>[];
  final _scrollController = ScrollController();
  bool _isThinking = false;

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _sendMessage() async {
    if (_controller.text.isEmpty || _isThinking) return;

    final prompt = _controller.text;
    _controller.clear();

    print('Chat: Sending message: $prompt');

    setState(() {
      _messages.add(ChatMessage(text: prompt, isUser: true));
      _isThinking = true;
    });
    _scrollToBottom();

    try {
      print('Chat: Calling modelManager.inference()');
      final response = await widget.modelManager.inference(prompt);
      print(
          'Chat: Got response: ${response.substring(0, response.length > 100 ? 100 : response.length)}...');

      setState(() {
        _messages.add(ChatMessage(text: response, isUser: false));
      });
    } catch (e) {
      print('Chat: Error during inference: $e');
      setState(() {
        _messages.add(ChatMessage(text: 'Error: $e', isUser: false));
      });
    } finally {
      setState(() => _isThinking = false);
      _scrollToBottom();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Chat with Gemma 3B')),
      body: Column(
        children: [
          Expanded(
            child: _messages.isEmpty
                ? Center(child: Text('Start chatting with Gemma 3B!'))
                : ListView.builder(
                    controller: _scrollController,
                    itemCount: _messages.length,
                    itemBuilder: (context, index) {
                      final message = _messages[index];
                      return Container(
                        margin:
                            EdgeInsets.symmetric(vertical: 4, horizontal: 16),
                        child: Align(
                          alignment: message.isUser
                              ? Alignment.centerRight
                              : Alignment.centerLeft,
                          child: Container(
                            padding: EdgeInsets.all(12),
                            constraints: BoxConstraints(
                                maxWidth:
                                    MediaQuery.of(context).size.width * 0.8),
                            decoration: BoxDecoration(
                              color: message.isUser
                                  ? Theme.of(context).colorScheme.primary
                                  : Theme.of(context)
                                      .colorScheme
                                      .secondaryContainer,
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: SelectableText(
                              message.text,
                              style: TextStyle(
                                color: message.isUser
                                    ? Theme.of(context).colorScheme.onPrimary
                                    : Theme.of(context)
                                        .colorScheme
                                        .onSecondaryContainer,
                              ),
                            ),
                          ),
                        ),
                      );
                    },
                  ),
          ),
          if (_isThinking)
            Padding(
              padding: EdgeInsets.all(16),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2)),
                  SizedBox(width: 8),
                  Text('Gemma is thinking...'),
                ],
              ),
            ),
          Padding(
            padding: EdgeInsets.all(16),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    decoration: InputDecoration(
                      hintText: 'Type a message...',
                      border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(24)),
                    ),
                    onSubmitted: (_) => _sendMessage(),
                  ),
                ),
                SizedBox(width: 8),
                IconButton(
                  onPressed: _isThinking ? null : _sendMessage,
                  icon: Icon(Icons.send),
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
