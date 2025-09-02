import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

// --- FFI Type Definitions ---
final class LlamaOpaque extends Opaque {}

typedef LoadModelNative = Pointer<LlamaOpaque> Function(
    Pointer<Utf8> modelPath);
typedef PredictNative = Pointer<Utf8> Function(
    Pointer<LlamaOpaque> context, Pointer<Utf8> prompt);
typedef FreeStringNative = Void Function(Pointer<Utf8> str);
typedef FreeModelNative = Void Function(Pointer<LlamaOpaque> context);

typedef LoadModelDart = Pointer<LlamaOpaque> Function(Pointer<Utf8> modelPath);
typedef PredictDart = Pointer<Utf8> Function(
    Pointer<LlamaOpaque> context, Pointer<Utf8> prompt);
typedef FreeStringDart = void Function(Pointer<Utf8> str);
typedef FreeModelDart = void Function(Pointer<LlamaOpaque> context);

class LlamaFFI {
  late final DynamicLibrary _lib;
  late final LoadModelDart loadModel;
  late final PredictDart predict;
  late final FreeStringDart freeString;
  late final FreeModelDart freeModel;

  LlamaFFI() {
    _lib = Platform.isAndroid
        ? DynamicLibrary.open("libnative-lib.so")
        : DynamicLibrary.process();

    loadModel = _lib
        .lookup<NativeFunction<LoadModelNative>>('load_model')
        .asFunction<LoadModelDart>();

    predict = _lib
        .lookup<NativeFunction<PredictNative>>('predict')
        .asFunction<PredictDart>();

    freeString = _lib
        .lookup<NativeFunction<FreeStringNative>>('free_string')
        .asFunction<FreeStringDart>();

    freeModel = _lib
        .lookup<NativeFunction<FreeModelNative>>('free_model')
        .asFunction<FreeModelDart>();
  }
}
