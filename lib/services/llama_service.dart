import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';
import '../services/llama_ffi.dart';

class LlamaService {
  final LlamaFFI _ffi = LlamaFFI();
  Pointer<LlamaOpaque>? _context;
  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;

  Future<bool> loadModel(String modelPath, {bool useGpu = true}) async {
    try {
      final pathC = modelPath.toNativeUtf8();
      
      // Use GPU-enabled loading if supported
      _context = _ffi.loadModelWithGpu(pathC, useGpu);
      calloc.free(pathC);

      _isInitialized = _context != null && _context!.address != 0;
      
      if (_isInitialized) {
        print('Model loaded successfully with GPU: $useGpu');
      } else {
        print('Failed to load model with GPU: $useGpu');
      }
      
      return _isInitialized;
    } catch (e) {
      print('Error loading model: $e');
      _isInitialized = false;
      return false;
    }
  }

  // Fallback method for backward compatibility
  Future<bool> loadModelCpuOnly(String modelPath) async {
    return loadModel(modelPath, useGpu: false);
  }

  void resetConversation() {
    if (_isInitialized && _context != null) {
      _ffi.resetConversation(_context!);
    }
  }

  Future<String> generateResponse(String prompt) async {
    if (!_isInitialized || _context == null) {
      return 'Error: Model not loaded';
    }

    try {
      // Run inference on a background isolate using compute
      final result = await compute(_runInferenceCompute, {
        'contextAddress': _context!.address,
        'prompt': prompt,
      });

      return result.isEmpty ? 'No response generated' : result;
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

// Top-level function for isolate execution with compute
String _runInferenceCompute(Map<String, dynamic> args) {
  try {
    final int contextAddress = args['contextAddress'];
    final String prompt = args['prompt'];

    // Load the native library in the isolate
    final DynamicLibrary lib = Platform.isAndroid
        ? DynamicLibrary.open("libnative-lib.so")
        : DynamicLibrary.process();

    // Use simple function signatures without defining types
    final predict = lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<Void> context, Pointer<Utf8> prompt),
        Pointer<Utf8> Function(Pointer<Void> context, Pointer<Utf8> prompt)
    >('predict');

    final freeString = lib.lookupFunction<
        Void Function(Pointer<Utf8> str),
        void Function(Pointer<Utf8> str)
    >('free_string');

    // Convert context address back to pointer
    final contextPtr = Pointer<Void>.fromAddress(contextAddress);
    
    // Convert prompt to native string
    final promptC = prompt.toNativeUtf8();
    
    // Call the native predict function
    final resultPtr = predict(contextPtr, promptC);
    
    // Free the prompt string
    calloc.free(promptC);
    
    // Convert result to Dart string
    final result = resultPtr.toDartString();
    
    // Free the result string
    freeString(resultPtr);
    
    return result;
  } catch (e) {
    return 'Error in isolate: $e';
  }
}
