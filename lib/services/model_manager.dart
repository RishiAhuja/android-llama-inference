import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;

enum ModelStatus {
  notDownloaded,
  downloading,
  downloaded,
  error,
}

class ModelManager {
  static const String modelUrl = 'https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf';
  static const String modelFileName = 'gemma-3-1b-it-Q4_K_M.gguf';
  
  ModelStatus _status = ModelStatus.notDownloaded;
  double _downloadProgress = 0.0;
  String _errorMessage = '';
  File? _modelFile;

  ModelStatus get status => _status;
  double get downloadProgress => _downloadProgress;
  String get errorMessage => _errorMessage;
  File? get modelFile => _modelFile;

  Future<void> checkModelStatus() async {
    try {
      final appDir = await getApplicationDocumentsDirectory();
      _modelFile = File('${appDir.path}/$modelFileName');
      
      if (await _modelFile!.exists()) {
        _status = ModelStatus.downloaded;
      } else {
        _status = ModelStatus.notDownloaded;
      }
    } catch (e) {
      _status = ModelStatus.error;
      _errorMessage = 'Error checking model: $e';
    }
  }

  Future<void> downloadModel({Function(double)? onProgress, Function(String)? onError}) async {
    try {
      _status = ModelStatus.downloading;
      _downloadProgress = 0.0;
      
      final appDir = await getApplicationDocumentsDirectory();
      _modelFile = File('${appDir.path}/$modelFileName');
      
      final request = http.Request('GET', Uri.parse(modelUrl));
      final response = await request.send();
      
      if (response.statusCode != 200) {
        throw Exception('HTTP ${response.statusCode}');
      }

      final totalBytes = response.contentLength ?? 0;
      var downloadedBytes = 0;
      final bytes = <int>[];

      await for (final chunk in response.stream) {
        bytes.addAll(chunk);
        downloadedBytes += chunk.length;
        
        if (totalBytes > 0) {
          _downloadProgress = downloadedBytes / totalBytes;
          onProgress?.call(_downloadProgress);
        }
      }

      await _modelFile!.writeAsBytes(bytes);
      _status = ModelStatus.downloaded;
      
    } catch (e) {
      _status = ModelStatus.error;
      _errorMessage = 'Download failed: $e';
      onError?.call(_errorMessage);
    }
  }

  Future<void> deleteModel() async {
    try {
      if (_modelFile != null && await _modelFile!.exists()) {
        await _modelFile!.delete();
      }
      _status = ModelStatus.notDownloaded;
    } catch (e) {
      _status = ModelStatus.error;
      _errorMessage = 'Error deleting model: $e';
    }
  }
}
