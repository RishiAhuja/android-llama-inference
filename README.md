# Gemma AI Assistant - Flutter Mobile App

A Flutter application that runs Google's Gemma and other small language models on-device using llama.cpp with GPU acceleration support via Vulkan.

## Overview

This application provides a native mobile chat interface for running small instruction-tuned language models (under 1B parameters) directly on Android devices. The app features efficient on-device inference, model management, and GPU acceleration capabilities optimized for mobile hardware.

## Features

### Core Functionality
- **On-Device Inference**: Run language models completely offline without internet connectivity
- **GPU Acceleration**: Vulkan-based GPU acceleration for improved inference speed
- **Multiple Model Support**: Download and switch between various small language models
- **Streaming Inference**: Non-blocking UI with real-time response generation
- **Memory Management**: Intelligent memory usage with chunked model downloading
- **Response Time Tracking**: Display inference performance metrics

### Supported Models
- Gemma 3 1B Instruct (Recommended)
- Gemma 3 270M Instruct (Ultra-lightweight)
- Llama 3.2 1B Instruct (Edge-optimized)
- Qwen2.5 0.5B Instruct (Multilingual)
- Lille 130M Instruct (Minimal footprint)
- TinyLlama 1.1B Chat (Compact chat model)
- TinyLlama 2 1B MiniGuanaco (Instruction-tuned)
- TinyVicuna 1B (Dialogue-optimized)

### User Interface
- **Model Management Screen**: Browse, download, and manage language models
- **Chat Interface**: Clean, modern chat UI with typing indicators
- **GPU Toggle**: Switch between GPU and CPU-only inference modes
- **Download Progress**: Real-time download progress with cancellation support
- **Confirmation Dialogs**: Safe model deletion with user confirmation

## Architecture

### Technology Stack
- **Frontend**: Flutter (Dart)
- **Backend**: llama.cpp (C++)
- **FFI**: Dart Foreign Function Interface for native integration
- **Build System**: CMake for native library compilation
- **GPU Backend**: Vulkan for Android GPU acceleration

### Key Components

#### Dart/Flutter Layer
- `LlamaService`: High-level service for model operations
- `ModelManager`: Handles model downloading, loading, and lifecycle
- `LlamaFFI`: Foreign Function Interface bindings to native library
- `ChatScreen`: Main chat interface with async inference
- `ModelManagementScreen`: Model browsing and management UI
- `GpuSettings`: Global GPU acceleration configuration

#### Native C++ Layer
- `native-lib.cpp`: JNI bridge and llama.cpp integration
- Model loading with GPU/CPU configuration
- Efficient batch processing for prompt inference
- Memory-optimized context management
- Chat template formatting for different model types

#### Build Configuration
- `CMakeLists.txt`: Native library build configuration with Vulkan support
- Gradle integration for Android builds
- ARM64 optimization flags for mobile performance

## Installation and Setup

### Prerequisites
- Flutter SDK (3.0 or higher)
- Android Studio with NDK
- Android device with API level 23+ (Android 6.0+)
- Minimum 3GB RAM recommended for model loading
- ARM64 device architecture (arm64-v8a)

### Build Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd gemma_app
   ```

2. **Initialize llama.cpp submodule**
   ```bash
   git submodule update --init --recursive
   ```

3. **Install Flutter dependencies**
   ```bash
   flutter pub get
   ```

4. **Build and run**
   ```bash
   flutter run
   ```

### Android Build Configuration

The app includes optimized build settings for mobile devices:

- **ARM64 Optimization**: FP16 NEON instruction support
- **Vulkan GPU Support**: Hardware-accelerated inference
- **Memory Mapping**: Efficient model file access
- **Batch Processing**: Parallel token processing for speed

## Usage Guide

### Getting Started

1. **Launch the App**: Start with the Model Management screen
2. **Download a Model**: Select and download your preferred language model
3. **Configure GPU**: Toggle GPU acceleration based on your device capabilities
4. **Start Chatting**: Navigate to the chat interface and begin conversations

### Model Selection Guidelines

- **Gemma 3 1B**: Best balance of quality and performance (Recommended)
- **Gemma 3 270M**: Fastest inference, basic capabilities
- **Qwen2.5 0.5B**: Good multilingual support
- **Lille 130M**: Minimal resource usage for low-end devices

### Performance Optimization

#### GPU Acceleration
- Enable for devices with modern GPUs (Adreno 640+, Mali-G76+)
- Disable for older devices or if experiencing stability issues
- Monitor battery usage as GPU acceleration may increase power consumption

#### Memory Management
- Models are downloaded in chunks to prevent out-of-memory errors
- Only one model is kept in memory at a time
- Automatic memory checks before model loading

#### Inference Settings
- Context size: 1024 tokens (configurable)
- Batch size: 512 tokens for efficient processing
- Thread count: Auto-configured based on CPU/GPU mode

## Development

### Project Structure

```
lib/
├── main.dart                    # App entry point and routing
├── models/
│   ├── chat_message.dart        # Chat message data model
│   └── model_config.dart        # Model configuration and GPU settings
├── screens/
│   ├── chat_screen.dart         # Main chat interface
│   └── model_management_screen.dart # Model management UI
└── services/
    ├── llama_ffi.dart          # FFI bindings to native library
    ├── llama_service.dart      # High-level model service
    └── model_manager.dart      # Model download and lifecycle

android/app/src/main/cpp/
├── CMakeLists.txt              # Native build configuration
├── native-lib.cpp              # C++ implementation
└── llama.cpp/                  # llama.cpp submodule
```

### Adding New Models

1. **Update Model Configuration**
   ```dart
   ModelConfig(
     id: 'new-model-id',
     name: 'New Model Name',
     description: 'Model description',
     url: 'https://huggingface.co/model/file.gguf',
     fileName: 'model.gguf',
     sizeInMB: 500,
     capabilities: ['Chat', 'Code'],
     isRecommended: false,
   )
   ```

2. **Test Compatibility**
   - Ensure model uses GGUF format
   - Verify model size fits device constraints
   - Test inference quality and speed

### Debugging

#### Common Issues
- **Out of Memory**: Reduce model size or enable streaming download
- **Slow Inference**: Enable GPU acceleration or use smaller model
- **Build Errors**: Ensure NDK version compatibility and submodule initialization

#### Logging
- Native layer logs via Android logcat with tag "LlamaJNI"
- Dart layer logging through Flutter's debug console
- Performance metrics displayed in chat interface

## Performance Benchmarks

### Inference Speed (tokens/second)
- **GPU Mode**: 15-25 tokens/sec (varies by device)
- **CPU Mode**: 5-15 tokens/sec (varies by model size and device)

### Memory Usage
- **Model Loading**: 800MB-1.2GB (depends on model size)
- **Runtime Overhead**: 100-200MB
- **Peak Usage**: During model loading and inference

### Battery Impact
- **CPU Mode**: Moderate battery usage
- **GPU Mode**: Higher battery usage but faster inference
- **Idle**: Minimal battery impact when not inferencing

## Contributing

### Development Guidelines
- Follow Flutter/Dart style conventions
- Test on multiple Android devices and API levels
- Ensure memory efficiency for mobile constraints
- Maintain compatibility with llama.cpp updates

### Testing
- Test model downloading and loading
- Verify inference quality across different models
- Check memory usage patterns
- Validate GPU/CPU switching functionality

## License

This project integrates multiple open-source components:
- Flutter framework (BSD License)
- llama.cpp library (MIT License)
- Model weights subject to their respective licenses

## Acknowledgments

- **llama.cpp**: Core inference engine
- **Google**: Gemma model architecture
- **Meta**: Llama model architecture
- **Alibaba**: Qwen model contributions
- **Flutter Team**: Mobile framework

## Troubleshooting

### Common Solutions

**App crashes on model loading**
- Check available device memory
- Try smaller model or enable GPU acceleration
- Restart app to clear memory

**Slow download speeds**
- Check internet connection
- Try downloading during off-peak hours
- Use WiFi instead of mobile data

**GPU acceleration not working**
- Verify device has Vulkan support
- Update device drivers if possible
- Fall back to CPU mode

**Build failures**
- Clean build directory: `flutter clean`
- Rebuild native components
- Check NDK version compatibility

For additional support and updates, refer to the project repository and documentation.
