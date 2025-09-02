import 'dart:ffi';
import 'package:ffi/ffi.dart';
import '../services/llama_ffi.dart';

class LlamaService {
  final LlamaFFI _ffi = LlamaFFI();
  Pointer<LlamaOpaque>? _context;
  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;

  Future<bool> loadModel(String modelPath) async {
    try {
      final pathC = modelPath.toNativeUtf8();
      _context = _ffi.loadModel(pathC);
      calloc.free(pathC);
      
      _isInitialized = _context != null && _context!.address != 0;
      return _isInitialized;
    } catch (e) {
      _isInitialized = false;
      return false;
    }
  }

  Future<String> generateResponse(String prompt) async {
    if (!_isInitialized || _context == null) {
      return 'Error: Model not loaded';
    }

    try {
      final promptC = prompt.toNativeUtf8();
      final resultPtr = _ffi.predict(_context!, promptC);
      calloc.free(promptC);

      final result = resultPtr.toDartString();
      _ffi.freeString(resultPtr);
      
      return result;
    } catch (e) {
      return 'Error generating response: $e';
    }
  }

  void dispose() {
    if (_isInitialized && _context != null) {
      _ffi.freeModel(_context!);
      _context = null;
      _isInitialized = false;
    }
  }
}
