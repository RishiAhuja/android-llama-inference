class ModelConfig {
  final String id;
  final String name;
  final String description;
  final String url;
  final String fileName;
  final int sizeInMB;
  final List<String> capabilities;
  final bool isRecommended;

  const ModelConfig({
    required this.id,
    required this.name,
    required this.description,
    required this.url,
    required this.fileName,
    required this.sizeInMB,
    required this.capabilities,
    this.isRecommended = false,
  });
}

class AvailableModels {
  static const List<ModelConfig> models = [
    ModelConfig(
      id: 'gemma-3-1b-it',
      name: 'Gemma 3 1B Instruct',
      description:
          'Google\'s Gemma 3 1B instruction-tuned model. Good balance of quality and speed.',
      url:
          'https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf',
      fileName: 'gemma-3-1b-it-Q4_K_M.gguf',
      sizeInMB: 800,
      capabilities: ['Chat', 'Question Answering', 'Text Generation'],
      isRecommended: true,
    ),
    ModelConfig(
      id: 'gemma-3-270m-it',
      name: 'Gemma 3 270M Instruct',
      description:
          'Ultra-lightweight Google Gemma model. Very fast but basic capabilities.',
      url:
          'https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf?download=true',
      fileName: 'gemma-3-270m-it-Q4_K_M.gguf',
      sizeInMB: 200,
      capabilities: ['Chat', 'Basic Text Generation'],
    ),
    ModelConfig(
      id: 'llama-3.2-1b',
      name: 'Llama 3.2 1B Instruct',
      description:
          'Meta\'s Llama 3.2 1B model. Optimized for edge devices and mobile.',
      url:
          'https://huggingface.co/ggml-org/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf',
      fileName: 'Llama-3.2-1B-Instruct-Q4_K_M.gguf',
      sizeInMB: 700,
      capabilities: ['Chat', 'Question Answering', 'Text Generation'],
    ),
    ModelConfig(
      id: 'qwen2.5-0.5b',
      name: 'Qwen2.5 0.5B Instruct',
      description:
          'Alibaba\'s ultra-efficient Qwen model. Great for mobile with multilingual support.',
      url:
          'https://huggingface.co/ggml-org/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf',
      fileName: 'Qwen2.5-0.5B-Instruct-Q4_K_M.gguf',
      sizeInMB: 350,
      capabilities: ['Chat', 'Multilingual', 'Text Generation'],
    ),
    ModelConfig(
      id: 'lille-130m-instruct',
      name: 'Lille 130M Instruct',
      description:
          'A tiny 130M parameter instruct model. Extremely lightweight, suitable for ultra-low-resource devices.',
      url:
          'https://huggingface.co/Nikity/lille-130m-instruct-GGUF/resolve/main/lille-130m-instruct-Q4_K_M.gguf',
      fileName: 'lille-130m-instruct-Q4_K_M.gguf',
      sizeInMB: 120,
      capabilities: ['Basic Chat', 'Instruction Following'],
    ),
    ModelConfig(
      id: 'tinyllama-1.1b-chat',
      name: 'TinyLlama 1.1B Chat',
      description:
          'A compact chat-optimized model with multiple quantization levels. Balanced quality for its small size.',
      url:
          'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf',
      fileName: 'TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf',
      sizeInMB: 670,
      capabilities: ['Chat', 'Question Answering', 'Text Generation'],
    ),
    ModelConfig(
      id: 'tinyllama-2-1b-miniguanaco',
      name: 'TinyLlama 2 1B MiniGuanaco',
      description:
          'Instruction-tuned TinyLlama variant, very compact with Q4 quantization for speed on edge devices.',
      url:
          'https://huggingface.co/TheBloke/TinyLlama-2-1B-miniguanaco-GGUF/resolve/main/TinyLlama-2-1B-miniguanaco.Q4_K_M.gguf',
      fileName: 'TinyLlama-2-1B-miniguanaco.Q4_K_M.gguf',
      sizeInMB: 720,
      capabilities: ['Chat', 'Instruction Following', 'Lightweight QA'],
    ),
    ModelConfig(
      id: 'tiny-vicuna-1b',
      name: 'TinyVicuna 1B',
      description:
          'Vicuna-finetuned lightweight model, efficient at small scale. Works well for dialogue tasks.',
      url:
          'https://huggingface.co/afrideva/Tiny-Vicuna-1B-GGUF/resolve/main/Tiny-Vicuna-1B.Q4_K_M.gguf',
      fileName: 'Tiny-Vicuna-1B.Q4_K_M.gguf',
      sizeInMB: 690,
      capabilities: ['Chat', 'Conversation', 'Text Generation'],
    ),

  ];

  static ModelConfig? getModelById(String id) {
    try {
      return models.firstWhere((model) => model.id == id);
    } catch (e) {
      return null;
    }
  }

  static ModelConfig get defaultModel => models.first;
}
