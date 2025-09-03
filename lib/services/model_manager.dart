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
  bool _downloadCancelled = false;

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
      _downloadCancelled = false; // Reset cancellation flag

      final appDir = await getApplicationDocumentsDirectory();
      _modelFile = File('${appDir.path}/${model.fileName}');

      // Check if partial download exists and try to resume
      var startByte = 0;
      if (await _modelFile!.exists()) {
        startByte = await _modelFile!.length();
        print('Resuming download from byte $startByte');
      }

      // Create the file and open it for writing (append mode if resuming)
      final sink = _modelFile!.openWrite(mode: startByte > 0 ? FileMode.append : FileMode.write);

      try {
        final request = http.Request('GET', Uri.parse(model.url));
        
        // Add range header for resume capability
        if (startByte > 0) {
          request.headers['Range'] = 'bytes=$startByte-';
        }
        
        // Add user-agent for better compatibility
        request.headers['User-Agent'] = 'FlutterApp/1.0';
        
        // Set timeout for the request
        final client = http.Client();
        final response = await client.send(request).timeout(
          const Duration(minutes: 30), // 30-minute timeout
          onTimeout: () {
            client.close();
            throw Exception('Download timeout - please check your internet connection');
          },
        );

        // Accept both 200 (full download) and 206 (partial content/resume)
        if (response.statusCode != 200 && response.statusCode != 206) {
          throw Exception('HTTP ${response.statusCode}: ${response.reasonPhrase}');
        }

        final totalBytes = (response.contentLength ?? 0) + startByte;
        var downloadedBytes = startByte;
        
        print('Download started: ${downloadedBytes}/${totalBytes} bytes');

        // Stream download with chunked writing to disk
        await for (final chunk in response.stream) {
          // Check if download is cancelled
          if (_downloadCancelled) {
            sink.close();
            await _modelFile!.delete();
            _status = ModelStatus.notDownloaded;
            print('Download cancelled');
            return;
          }

          // Write chunk directly to file (no memory accumulation)
          sink.add(chunk);
          downloadedBytes += chunk.length;

          if (totalBytes > 0) {
            _downloadProgress = downloadedBytes / totalBytes;
            onProgress?.call(_downloadProgress);
          }

          // Periodically flush to disk to avoid large write buffers
          if (downloadedBytes % (1024 * 1024) == 0) { // Flush every 1MB
            await sink.flush();
          }
        }

        // Ensure all data is written to disk
        await sink.flush();
        await sink.close();
        
        // Verify file size matches expected size
        final finalSize = await _modelFile!.length();
        if (totalBytes > 0 && finalSize != totalBytes) {
          throw Exception('Download incomplete: ${finalSize}/${totalBytes} bytes');
        }
        
        print('Download completed successfully: $finalSize bytes');
        _status = ModelStatus.downloaded;
      } catch (e) {
        // Clean up the file sink and remove partial file on error
        await sink.close();
        if (await _modelFile!.exists()) {
          await _modelFile!.delete();
        }
        rethrow;
      }
    } catch (e) {
      _status = ModelStatus.error;
      _errorMessage = 'Download failed: $e';
      onError?.call(_errorMessage);
    }
  }

  void cancelDownload() {
    _downloadCancelled = true;
    _status = ModelStatus.notDownloaded;
    _downloadProgress = 0.0;
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
