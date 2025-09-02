import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import '../models/model_config.dart';

enum ModelStatus {
  notDownloaded,
  downloading,
  downloaded,
  error,
}

class ModelManager {
  ModelStatus _status = ModelStatus.notDownloaded;
  double _downloadProgress = 0.0;
  String _errorMessage = '';
  File? _modelFile;
  ModelConfig? _currentModel;

  ModelStatus get status => _status;
  double get downloadProgress => _downloadProgress;
  String get errorMessage => _errorMessage;
  File? get modelFile => _modelFile;
  ModelConfig? get currentModel => _currentModel;

  Future<void> checkModelStatus([String? modelId]) async {
    try {
      final appDir = await getApplicationDocumentsDirectory();
      
      // Check if a specific model is requested
      if (modelId != null) {
        _currentModel = AvailableModels.getModelById(modelId);
      } else {
        // Check for any existing model
        for (final model in AvailableModels.models) {
          final file = File('${appDir.path}/${model.fileName}');
          if (await file.exists()) {
            _currentModel = model;
            _modelFile = file;
            _status = ModelStatus.downloaded;
            return;
          }
        }
        // If no model found, use default
        _currentModel = AvailableModels.defaultModel;
      }

      if (_currentModel != null) {
        _modelFile = File('${appDir.path}/${_currentModel!.fileName}');
        if (await _modelFile!.exists()) {
          _status = ModelStatus.downloaded;
        } else {
          _status = ModelStatus.notDownloaded;
        }
      }
    } catch (e) {
      _status = ModelStatus.error;
      _errorMessage = 'Error checking model: $e';
    }
  }

  Future<void> downloadModel(String modelId,
      {Function(double)? onProgress, Function(String)? onError}) async {
    try {
      final model = AvailableModels.getModelById(modelId);
      if (model == null) {
        throw Exception('Model not found: $modelId');
      }

      _currentModel = model;
      _status = ModelStatus.downloading;
      _downloadProgress = 0.0;

      final appDir = await getApplicationDocumentsDirectory();
      _modelFile = File('${appDir.path}/${model.fileName}');

      final request = http.Request('GET', Uri.parse(model.url));
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

  Future<void> deleteModel([String? modelId]) async {
    try {
      final appDir = await getApplicationDocumentsDirectory();
      
      if (modelId != null) {
        final model = AvailableModels.getModelById(modelId);
        if (model != null) {
          final file = File('${appDir.path}/${model.fileName}');
          if (await file.exists()) {
            await file.delete();
          }
          // If this was the current model, reset status
          if (_currentModel?.id == modelId) {
            _status = ModelStatus.notDownloaded;
            _modelFile = null;
          }
        }
      } else {
        // Delete current model
        if (_modelFile != null && await _modelFile!.exists()) {
          await _modelFile!.delete();
        }
        _status = ModelStatus.notDownloaded;
        _modelFile = null;
      }
    } catch (e) {
      _status = ModelStatus.error;
      _errorMessage = 'Delete failed: $e';
    }
  }

  Future<void> setCurrentModel(String modelId) async {
    _currentModel = AvailableModels.getModelById(modelId);
    if (_currentModel != null) {
      await checkModelStatus(modelId);
    }
  }

  Future<List<String>> getDownloadedModels() async {
    try {
      final appDir = await getApplicationDocumentsDirectory();
      final downloadedModels = <String>[];
      
      for (final model in AvailableModels.models) {
        final file = File('${appDir.path}/${model.fileName}');
        if (await file.exists()) {
          downloadedModels.add(model.fileName); // Return fileName for compatibility
        }
      }
      
      return downloadedModels;
    } catch (e) {
      return [];
    }
  }
}
