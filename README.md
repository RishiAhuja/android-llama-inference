# On-Device Language Model Inference with Flutter and llama.cpp

A Flutter implementation for running quantized GGUF language models on Android devices using llama.cpp with Vulkan GPU acceleration.

## Technical Overview

This project implements a mobile application that performs local language model inference without network dependencies. The architecture integrates Flutter's Dart runtime with llama.cpp's C++ inference engine through FFI (Foreign Function Interface), enabling deployment of sub-1B parameter models on resource-constrained mobile hardware.

## Implementation Details

### Core Architecture

**Frontend Layer (Dart/Flutter)**
- Isolate-based asynchronous inference to prevent UI blocking
- FFI bindings for native library communication
- Streaming model download with chunked file writing
- Memory-aware model lifecycle management

**Native Layer (C++/llama.cpp)**
- GGUF format model loading with mmap support
- Vulkan compute backend for GPU acceleration
- Batch processing for prompt tokenization
- Context management with KV cache optimization

**Build System**
- CMake configuration with Vulkan backend compilation
- Android NDK integration with ARM64 NEON optimizations
- Cross-compilation for aarch64-linux-android target

### Memory Management

The implementation addresses mobile memory constraints through:

- **Streaming Downloads**: HTTP response streaming with direct-to-disk writing
- **Model Loading**: Memory-mapped file access to reduce RAM usage
- **Context Pooling**: Reusable inference contexts with batch allocation
- **Garbage Collection**: Explicit cleanup of native resources

### GPU Acceleration

Vulkan integration provides compute acceleration through:

```cpp
// Model parameters for GPU offloading
mparams.n_gpu_layers = 99;  // Offload all compatible layers
cparams.n_threads = 2;      // Reduced CPU threads when using GPU
```

GPU layer offloading automatically detects device capabilities and falls back to CPU processing for unsupported operations.

## Supported Model Formats

All models must be in GGUF format with Q4_K_M quantization for optimal mobile performance:

| Model | Parameters | Size (MB) | Context | Use Case |
|-------|------------|-----------|---------|----------|
| gemma-3-1b-it | 1B | 800 | 1024 | General instruction following |
| gemma-3-270m-it | 270M | 200 | 1024 | Minimal resource usage |
| llama-3.2-1b | 1B | 700 | 1024 | Edge-optimized inference |
| qwen2.5-0.5b | 500M | 350 | 1024 | Multilingual support |
| lille-130m | 130M | 120 | 1024 | Ultra-low resource environments |

## Build Configuration

### Prerequisites

- Flutter SDK 3.0+
- Android NDK 26.1.10909125
- CMake 3.22.1+
- Git with submodule support

### Compilation Flags

```cmake
# ARM64 optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16")

# Vulkan backend
set(LLAMA_VULKAN ON CACHE BOOL "Enable Vulkan support")

# Memory optimization
set(LLAMA_NATIVE OFF CACHE BOOL "Disable native optimizations")
```

### Build Process

```bash
git submodule update --init --recursive
flutter pub get
flutter build apk --release
```

## Runtime Configuration

### Inference Parameters

```cpp
// Context configuration
cparams.n_ctx = 1024;           // Token context window
cparams.n_batch = 512;          // Batch size for parallel processing
cparams.n_ubatch = 512;         // Micro-batch size
cparams.n_threads = 4;          // CPU thread count (auto-adjusted for GPU)
```

### Sampling Configuration

```cpp
// Sampling chain
llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
```

### Memory Requirements

- **Minimum RAM**: 3GB (2GB for model + 1GB system overhead)
- **Recommended RAM**: 4GB+ for stable operation
- **Storage**: 1-2GB per model (varies by quantization)

### Memory Usage Patterns

```
Model Loading Phase:
├── File mapping: 800MB-1.2GB
├── Context allocation: 100-200MB
└── Batch buffers: 50-100MB

Inference Phase:
├── KV cache: 50-150MB (grows with context)
├── Attention computation: 100-300MB (temporary)
└── Token generation: 10-50MB
```

## API Reference

### FFI Interface

```dart
typedef LoadModelWithGpuNative = Pointer<LlamaOpaque> Function(
    Pointer<Utf8> modelPath, Bool useGpu);

typedef PredictNative = Pointer<Utf8> Function(
    Pointer<LlamaOpaque> context, Pointer<Utf8> prompt);
```

### Service Layer

```dart
class LlamaService {
  Future<bool> loadModel(String modelPath, {bool useGpu = true});
  Future<String> generateResponse(String prompt);
  void resetConversation();
  void dispose();
}
```

### Model Management

```dart
class ModelManager {
  Future<void> downloadModel(String modelId, {
    Function(double)? onProgress,
    Function(String)? onError,
  });
  Future<void> deleteModel(String modelId);
  Future<List<String>> getDownloadedModels();
}
```

## Error Handling

### Common Failure Modes

**Memory allocation failures:**
```
I/LlamaJNI: Failed to allocate context memory
E/LlamaJNI: Model loading failed: insufficient memory
```

**GPU initialization failures:**
```
W/LlamaJNI: Vulkan device not found, falling back to CPU
I/LlamaJNI: GPU acceleration disabled
```

**Model format errors:**
```
E/LlamaJNI: Invalid GGUF header
E/LlamaJNI: Unsupported quantization format
```

### Recovery Mechanisms

- Automatic GPU fallback to CPU on device incompatibility
- Progressive memory pressure handling with context reduction
- Model download resume on network interruption
- Graceful degradation for unsupported architectures

## Debugging and Profiling

### Native Layer Debugging

```bash
adb logcat | grep LlamaJNI
```

### Memory Profiling

```bash
adb shell dumpsys meminfo com.example.gemma_app
```

### Performance Analysis

```cpp
// Timing inference
auto start = std::chrono::high_resolution_clock::now();
llama_decode(ctx, batch);
auto end = std::chrono::high_resolution_clock::now();
LOGI("Decode time: %ldms", duration_cast<milliseconds>(end - start).count());
```

## Security Considerations

- Models execute in application sandbox
- No network access during inference
- Local file system access only to app directories
- Native code compiled with stack protection enabled

## Platform Limitations

- **Android API 23+**: Required for NDK features
- **ARM64 only**: No x86/x86_64 support
- **Vulkan 1.0+**: For GPU acceleration
- **64-bit only**: 32-bit architectures unsupported

## Optimization Techniques

### Quantization Strategy
- Q4_K_M: Optimal balance of size and quality
- Mixed precision: FP16 for attention, INT4 for weights
- Dynamic quantization: Runtime precision adjustment

### Context Management
- Sliding window attention for long conversations
- KV cache compression for memory efficiency
- Batch processing for parallel token generation

### Threading Model
- CPU threads: 2-4 cores depending on GPU usage
- GPU work queues: Asynchronous compute dispatch
- Dart isolates: Background inference without UI blocking

## License and Dependencies

- **llama.cpp**: MIT License
- **Flutter**: BSD 3-Clause License
- **Vulkan SDK**: Apache 2.0 License
- **Model weights**: Individual model licenses apply
