#include <jni.h>
#include <string>
#include <vector>
#include <cstring>
#include <algorithm>
#include <random>
#include <cmath>
#include <android/log.h>
#include "llama.h"

// Log helper
#define LOG_TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Enhanced struct to hold model and context with proper memory management
struct llama_context_wrapper {
    llama_model* model = nullptr;
    llama_context* context = nullptr;
    llama_sampler* sampler = nullptr;
    llama_memory_t memory = nullptr;
    llama_batch batch = {0};  // Reusable batch for efficiency
    std::vector<llama_seq_id> seq_ids;  // Buffer for sequence IDs
    std::vector<llama_token> conversation_tokens;
    int n_past = 0;  // Track position in conversation
    bool conversation_started = false;
    
    ~llama_context_wrapper() {
        cleanup();
    }
    
    void cleanup() {
        if (batch.token) {
            llama_batch_free(batch);
            batch = {0};
        }
        if (sampler) {
            llama_sampler_free(sampler);
            sampler = nullptr;
        }
        if (context) {
            llama_free(context);
            context = nullptr;
        }
        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
    }
};

// Helper function to create and configure sampler (ultra-fast for mobile)
llama_sampler* create_sampler() {
    auto sparams = llama_sampler_chain_default_params();
    auto* sampler = llama_sampler_chain_init(sparams);
    
    // Balanced sampling for good quality (sampling is not the bottleneck):
    // 1. Top-K filtering
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    
    // 2. Top-P nucleus sampling  
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    
    // 3. Temperature scaling
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    
    // 4. Final distribution sampling
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(12345));
    
    return sampler;
}

// Helper function to format chat messages using proper Gemma template
std::string format_chat_message(llama_model* model, const std::string& user_message) {
    // Try using the model's built-in chat template first
    llama_chat_message messages[1] = {
        {"user", user_message.c_str()}
    };
    
    const char* chat_template = llama_model_chat_template(model, nullptr);
    
    if (chat_template != nullptr) {
        // Apply chat template
        std::vector<char> formatted(user_message.length() * 6); // More generous buffer
        int32_t result = llama_chat_apply_template(
            chat_template,
            messages,
            1,
            true, // add_assistant_start
            formatted.data(),
            formatted.size()
        );
        
        if (result > 0 && result <= (int32_t)formatted.size()) {
            std::string template_result(formatted.data(), result);
            LOGI("Using model chat template, result: %.100s...", template_result.c_str());
            return template_result;
        }
        
        if (result > (int32_t)formatted.size()) {
            // Need larger buffer
            formatted.resize(result + 1);
            result = llama_chat_apply_template(
                chat_template,
                messages,
                1,
                true,
                formatted.data(),
                formatted.size()
            );
            if (result > 0) {
                std::string template_result(formatted.data(), result);
                LOGI("Using model chat template (large buffer), result: %.100s...", template_result.c_str());
                return template_result;
            }
        }
    }
    
    // Fallback to manual Gemma format
    LOGI("Chat template failed or not available, using manual Gemma format");
    return "<start_of_turn>user\n" + user_message + "<end_of_turn>\n<start_of_turn>model\n";
}

// Helper function to convert C++ string to C char*
char* string_to_char_ptr(const std::string& s) {
    char* pc = new char[s.size() + 1];
    std::strcpy(pc, s.c_str());
    return pc;
}

// Helper function to clear and reset batch for reuse
void clear_batch(llama_batch& batch) {
    batch.n_tokens = 0;
}

// Helper function to add a token to the batch efficiently
bool add_token_to_batch(llama_batch& batch, llama_token token, llama_pos pos, 
                       std::vector<llama_seq_id>& seq_ids, bool get_logits = false) {
    if (batch.n_tokens >= 512) {  // Max batch size
        return false;
    }
    
    const int idx = batch.n_tokens;
    batch.token[idx] = token;
    batch.pos[idx] = pos;
    batch.n_seq_id[idx] = 1;  // Number of sequences this token belongs to
    
    // Ensure seq_ids buffer has enough space
    if (seq_ids.size() <= (size_t)idx) {
        seq_ids.resize(idx + 1);
    }
    seq_ids[idx] = 0;  // Use sequence 0
    batch.seq_id[idx] = &seq_ids[idx];  // Point to our sequence ID
    
    batch.logits[idx] = get_logits ? 1 : 0;
    batch.n_tokens++;
    
    return true;
}

// Helper function to process tokens in efficient batches
int process_tokens_in_batches(llama_context* ctx, llama_batch& batch, 
                             const std::vector<llama_token>& tokens, 
                             std::vector<llama_seq_id>& seq_ids,
                             int start_pos, bool get_logits_for_last = true) {
    // Process ALL tokens in a single batch for maximum efficiency
    clear_batch(batch);
    
    LOGI("Processing %zu tokens in single efficient batch", tokens.size());
    
    // Add all tokens to the batch at once
    for (size_t i = 0; i < tokens.size(); i++) {
        const bool is_last_token = (i == tokens.size() - 1);
        const bool get_logits = get_logits_for_last && is_last_token;
        
        if (!add_token_to_batch(batch, tokens[i], start_pos + i, seq_ids, get_logits)) {
            LOGE("Failed to add token %zu to batch", i);
            return -1;
        }
    }
    
    // Process entire batch in ONE call to llama_decode
    if (llama_decode(ctx, batch) != 0) {
        LOGE("Failed to decode batch of %zu tokens", tokens.size());
        return -1;
    }
    
    LOGI("Successfully processed all %zu tokens in single batch", tokens.size());
    return static_cast<int>(tokens.size());
}

extern "C" {
    // ---- FFI Functions Exposed to Dart ----

    __attribute__((visibility("default"))) __attribute__((used))
    void* load_model_with_gpu(const char* model_path, bool use_gpu) {
        LOGI("Loading model from: %s (GPU: %s)", model_path, use_gpu ? "enabled" : "disabled");
        
        // Initialize backend once
        llama_backend_init();

        auto* wrapper = new llama_context_wrapper();

        // Configure model parameters
        llama_model_params mparams = llama_model_default_params();
        mparams.use_mmap = true;  // Use memory mapping for efficiency
        mparams.use_mlock = false; // Don't lock memory on mobile
        
        // GPU acceleration settings
        if (use_gpu) {
            mparams.n_gpu_layers = 10; // Offload some layers to GPU (will auto-limit based on VRAM)
            LOGI("GPU acceleration enabled: offloading layers to GPU");
        } else {
            mparams.n_gpu_layers = 0; // CPU-only mode
            LOGI("CPU-only mode enabled");
        }
        
        // Load model
        wrapper->model = llama_model_load_from_file(model_path, mparams);
        if (wrapper->model == nullptr) {
            LOGE("Failed to load model");
            delete wrapper;
            return nullptr;
        }

        // Configure context parameters (properly optimized for mobile performance)
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 1024;      // Reasonable context size
        cparams.n_batch = 512;     // Large batch size for efficient parallel processing
        cparams.n_ubatch = 512;
        
        if (use_gpu) {
            // GPU-optimized settings
            cparams.n_threads = 2;     // Fewer CPU threads when using GPU
            cparams.n_threads_batch = 2;
            LOGI("Using GPU-optimized thread configuration");
        } else {
            // CPU-optimized settings  
            cparams.n_threads = 4;     // Use multiple CPU cores for matrix operations
            cparams.n_threads_batch = 4; // Use multiple cores for batch processing
            LOGI("Using CPU-optimized thread configuration");
        }
        
        cparams.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
        cparams.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
        cparams.attention_type = LLAMA_ATTENTION_TYPE_UNSPECIFIED;
        cparams.defrag_thold = -1.0f;
        
        // Create context
        wrapper->context = llama_init_from_model(wrapper->model, cparams);
        if (wrapper->context == nullptr) {
            LOGE("Failed to create context");
            llama_model_free(wrapper->model);
            delete wrapper;
            return nullptr;
        }

        // Get memory handle for efficient KV cache management
        wrapper->memory = llama_get_memory(wrapper->context);
        
        // Create and configure sampler
        wrapper->sampler = create_sampler();
        if (wrapper->sampler == nullptr) {
            LOGE("Failed to create sampler");
            wrapper->cleanup();
            delete wrapper;
            return nullptr;
        }

        // Initialize reusable batch (proper size for efficient parallel processing)
        wrapper->batch = llama_batch_init(512, 0, 1);  // Match n_batch size
        if (wrapper->batch.token == nullptr) {
            LOGE("Failed to create batch");
            wrapper->cleanup();
            delete wrapper;
            return nullptr;
        }
        
        // Initialize sequence IDs buffer (match batch size)
        wrapper->seq_ids.resize(512, 0);  // Match batch size

        LOGI("Model loaded successfully");
        return wrapper;
    }

    __attribute__((visibility("default"))) __attribute__((used))
    void* load_model(const char* model_path) {
        return load_model_with_gpu(model_path, true); // Default to GPU enabled
    }

    __attribute__((visibility("default"))) __attribute__((used))
    const char* predict(void* context_ptr, const char* prompt) {
        auto* wrapper = static_cast<llama_context_wrapper*>(context_ptr);
        if (wrapper == nullptr || wrapper->context == nullptr || wrapper->model == nullptr) {
            return string_to_char_ptr("Model not loaded");
        }

        LOGI("Starting prediction for prompt: %.100s...", prompt);

        // Get vocab from model for tokenization
        const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
        if (vocab == nullptr) {
            return string_to_char_ptr("Failed to get vocab");
        }

        // Format prompt using proper chat template
        std::string formatted_prompt = format_chat_message(wrapper->model, std::string(prompt));
        LOGI("Formatted prompt: %.200s...", formatted_prompt.c_str());
        
        // Tokenize the formatted prompt
        std::vector<llama_token> prompt_tokens;
        prompt_tokens.resize(llama_n_ctx(wrapper->context));
        
        int n_prompt_tokens = llama_tokenize(
            vocab, 
            formatted_prompt.c_str(), 
            formatted_prompt.length(), 
            prompt_tokens.data(), 
            prompt_tokens.size(), 
            true,  // add_special
            false  // parse_special
        );
        
        if (n_prompt_tokens < 0) {
            LOGE("Failed to tokenize prompt");
            return string_to_char_ptr("Failed to tokenize prompt");
        }
        prompt_tokens.resize(n_prompt_tokens);
        LOGI("Tokenized prompt: %d tokens", n_prompt_tokens);

        // Clear memory for new conversation if this is a fresh start
        if (!wrapper->conversation_started) {
            llama_memory_clear(wrapper->memory, true);
            wrapper->conversation_tokens.clear();
            wrapper->n_past = 0;
            wrapper->conversation_started = true;
            LOGI("Started new conversation");
        }
        
        // Add prompt tokens to conversation
        wrapper->conversation_tokens.insert(
            wrapper->conversation_tokens.end(), 
            prompt_tokens.begin(), 
            prompt_tokens.end()
        );

        // Process prompt tokens efficiently in batches
        LOGI("Processing %d prompt tokens in batches", n_prompt_tokens);
        LOGI("Starting ultra-fast processing..."); // Immediate feedback
        int processed = process_tokens_in_batches(
            wrapper->context, 
            wrapper->batch, 
            prompt_tokens, 
            wrapper->seq_ids,  // Pass sequence IDs buffer
            wrapper->n_past, 
            true  // get logits for last token
        );
        
        if (processed != n_prompt_tokens) {
            LOGE("Failed to process prompt tokens: processed %d/%d", processed, n_prompt_tokens);
            return string_to_char_ptr("Failed to process prompt");
        }
        
        wrapper->n_past += n_prompt_tokens;
        LOGI("Processed prompt efficiently, n_past = %d", wrapper->n_past);

        // Generation parameters - optimized for mobile speed
        const int n_predict = 20;  // Ultra-short for mobile speed
        const llama_token eos_token = llama_vocab_eos(vocab);
        const llama_token eot_token = llama_vocab_eot(vocab);
        
        std::string response = "";
        std::string accumulated_text = "";  // Buffer to check for end patterns
        
        LOGI("Starting efficient generation loop, max tokens: %d", n_predict);
        
        // Efficient generation loop with single reusable batch
        for (int i = 0; i < n_predict; i++) {
            // Sample next token using the sampler chain
            llama_token new_token = llama_sampler_sample(wrapper->sampler, wrapper->context, -1);
            
            // Check for end of sequence tokens first
            if (new_token == eos_token || new_token == eot_token) {
                LOGI("Hit EOS/EOT token (%d), stopping generation", new_token);
                break;
            }
            
            // Accept the token (updates sampler state)
            llama_sampler_accept(wrapper->sampler, new_token);
            
            // Convert token to text
            char piece[256];
            int n_chars = llama_token_to_piece(
                vocab, 
                new_token, 
                piece, 
                sizeof(piece), 
                0,     // lstrip
                false  // special
            );
            
            if (n_chars > 0) {
                piece[n_chars] = '\0';
                std::string token_text(piece);
                
                // Add to accumulated text for pattern checking
                accumulated_text += token_text;
                response += token_text;
                
                // Check for various end patterns (more comprehensive)
                if (accumulated_text.find("<end_of_turn>") != std::string::npos ||
                    accumulated_text.find("</s>") != std::string::npos ||
                    accumulated_text.find("<|end|>") != std::string::npos ||
                    accumulated_text.find("<start_of_turn>user") != std::string::npos) {
                    LOGI("Hit end pattern in text: '%.30s', stopping generation", accumulated_text.c_str());
                    
                    // Remove the end pattern from response
                    size_t end_pos = response.find("<end_of_turn>");
                    if (end_pos != std::string::npos) {
                        response = response.substr(0, end_pos);
                    }
                    end_pos = response.find("<start_of_turn>");
                    if (end_pos != std::string::npos) {
                        response = response.substr(0, end_pos);
                    }
                    break;
                }
                
                // Keep only last 50 chars in accumulated_text for efficiency
                if (accumulated_text.length() > 50) {
                    accumulated_text = accumulated_text.substr(accumulated_text.length() - 50);
                }
            }

            // Add new token to conversation
            wrapper->conversation_tokens.push_back(new_token);
            
            // EFFICIENT: Reuse existing batch instead of creating new ones
            clear_batch(wrapper->batch);
            if (!add_token_to_batch(wrapper->batch, new_token, wrapper->n_past, wrapper->seq_ids, true)) {
                LOGE("Failed to add token to batch at position %d", i);
                break;
            }
            
            // Decode single token efficiently
            if (llama_decode(wrapper->context, wrapper->batch) != 0) {
                LOGE("Failed to decode token at position %d", i);
                break;
            }
            
            wrapper->n_past++;
            
            // Check for context overflow
            if (wrapper->n_past >= llama_n_ctx(wrapper->context) - 10) {
                LOGI("Approaching context limit, stopping generation");
                break;
            }
            
            // Log progress every 5 tokens for better mobile UX feedback
            if ((i + 1) % 5 == 0) {
                LOGI("Generated %d/%d tokens, current: '%.20s...'", i + 1, n_predict, response.c_str());
            }
        }

        LOGI("Generated response: %.200s...", response.c_str());
        return string_to_char_ptr(response);
    }
    
    __attribute__((visibility("default"))) __attribute__((used))
    void free_string(char* str) {
        delete[] str;
    }

    __attribute__((visibility("default"))) __attribute__((used))
    void free_model(void* context_ptr) {
        if (context_ptr != nullptr) {
            auto* wrapper = static_cast<llama_context_wrapper*>(context_ptr);
            LOGI("Freeing model resources");
            delete wrapper; // Destructor will handle cleanup
        }
        llama_backend_free();
    }

    __attribute__((visibility("default"))) __attribute__((used))
    void reset_conversation(void* context_ptr) {
        auto* wrapper = static_cast<llama_context_wrapper*>(context_ptr);
        if (wrapper != nullptr && wrapper->context != nullptr && wrapper->memory != nullptr) {
            LOGI("Resetting conversation");
            
            // Clear memory (both data and metadata)
            llama_memory_clear(wrapper->memory, true);
            
            // Reset sampler state
            if (wrapper->sampler) {
                llama_sampler_reset(wrapper->sampler);
            }
            
            // Reset wrapper state
            wrapper->conversation_tokens.clear();
            wrapper->n_past = 0;
            wrapper->conversation_started = false;
            
            LOGI("Conversation reset complete");
        }
    }
}