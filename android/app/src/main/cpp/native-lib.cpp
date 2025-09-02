#include <jni.h>
#include <string>
#include <vector>
#include <cstring>
#include "llama.h"

// Simple struct to hold model and context
struct llama_context_wrapper {
    llama_model* model = nullptr;
    llama_context* context = nullptr;
    llama_batch batch = {0};
};

// Helper function to convert C++ string to C char*
char* string_to_char_ptr(const std::string& s) {
    char* pc = new char[s.size() + 1];
    std::strcpy(pc, s.c_str());
    return pc;
}

extern "C" {
    // ---- FFI Functions Exposed to Dart ----

    __attribute__((visibility("default"))) __attribute__((used))
    void* load_model(const char* model_path) {
        // Initialize backend once
        llama_backend_init();

        auto* wrapper = new llama_context_wrapper();

        // Use modern API for model loading
        llama_model_params mparams = llama_model_default_params();
        wrapper->model = llama_model_load_from_file(model_path, mparams);
        if (wrapper->model == nullptr) {
            delete wrapper;
            return nullptr;
        }

        // Use modern API for context creation
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 2048;
        cparams.n_batch = 512;
        
        wrapper->context = llama_init_from_model(wrapper->model, cparams);
        if (wrapper->context == nullptr) {
            llama_model_free(wrapper->model);
            delete wrapper;
            return nullptr;
        }

        // Initialize batch
        wrapper->batch = llama_batch_init(512, 0, 1);

        return wrapper;
    }

    __attribute__((visibility("default"))) __attribute__((used))
    const char* predict(void* context_ptr, const char* prompt) {
        auto* wrapper = static_cast<llama_context_wrapper*>(context_ptr);
        if (wrapper == nullptr || wrapper->context == nullptr || wrapper->model == nullptr) {
            return string_to_char_ptr("Model not loaded");
        }

        // Reset the batch for new inference
        wrapper->batch.n_tokens = 0;
        
        // Removed the non-existent kv_cache_clear function. 
        // Resetting the batch and starting at pos 0 handles this implicitly in older versions.

        // Tokenize the prompt using modern API
        std::vector<llama_token> tokens_list;
        tokens_list.resize(llama_n_ctx(wrapper->context));
        
        // FIX: Using the correct function name from your codebase: llama_model_get_vocab
        int n_tokens = llama_tokenize(llama_model_get_vocab(wrapper->model), prompt, strlen(prompt), tokens_list.data(), tokens_list.size(), true, false);
        if (n_tokens < 0) {
            return string_to_char_ptr("Failed to tokenize prompt");
        }
        
        // Add prompt tokens to the batch manually
        for (int i = 0; i < n_tokens; i++) {
            // FIX: Removed invalid check for 'n_tokens_alloc' which does not exist.
            wrapper->batch.token[wrapper->batch.n_tokens] = tokens_list[i];
            wrapper->batch.pos[wrapper->batch.n_tokens] = i;
            wrapper->batch.n_seq_id[wrapper->batch.n_tokens] = 1;
            wrapper->batch.seq_id[wrapper->batch.n_tokens][0] = 0;
            wrapper->batch.logits[wrapper->batch.n_tokens] = false;
            wrapper->batch.n_tokens++;
        }
        
        // The last token gets the logits
        if (wrapper->batch.n_tokens > 0) {
            wrapper->batch.logits[wrapper->batch.n_tokens - 1] = true;
        }

        // Evaluate the prompt
        if (llama_decode(wrapper->context, wrapper->batch) != 0) {
            return string_to_char_ptr("Failed to evaluate prompt");
        }

        std::string response = "";
        int n_cur = n_tokens;
        int n_len = 100; // Generate up to 100 tokens

        // Generation loop
        for (int i = 0; i < n_len; i++) {
            // Get logits for the last token
            auto* logits = llama_get_logits_ith(wrapper->context, wrapper->batch.n_tokens - 1);
            // FIX: Pass the model's vocabulary to llama_n_vocab.
            auto n_vocab = llama_n_vocab(llama_model_get_vocab(wrapper->model));

            // Simple greedy sampling
            llama_token new_token_id = 0;
            float max_logit = logits[0];
            for (int j = 1; j < n_vocab; j++) {
                if (logits[j] > max_logit) {
                    max_logit = logits[j];
                    new_token_id = j;
                }
            }

            // FIX: Pass the model's vocabulary to llama_token_eos.
            if (new_token_id == llama_token_eos(llama_model_get_vocab(wrapper->model))) {
                break;
            }

            // Convert token to text
            char piece[256];
            // FIX: Pass the model's vocabulary to llama_token_to_piece.
            int n_chars = llama_token_to_piece(llama_model_get_vocab(wrapper->model), new_token_id, piece, sizeof(piece), 0, false);
            if (n_chars > 0) {
                piece[n_chars] = '\0';
                response += std::string(piece);
            }

            // Prepare for next iteration
            wrapper->batch.n_tokens = 0;
            wrapper->batch.token[0] = new_token_id;
            wrapper->batch.pos[0] = n_cur;
            wrapper->batch.n_seq_id[0] = 1;
            wrapper->batch.seq_id[0][0] = 0;
            wrapper->batch.logits[0] = true;
            wrapper->batch.n_tokens = 1;

            if (llama_decode(wrapper->context, wrapper->batch) != 0) {
                break;
            }

            n_cur++;
        }

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
            if (wrapper->context != nullptr) {
                llama_free(wrapper->context);
            }
            if (wrapper->model != nullptr) {
                llama_model_free(wrapper->model);
            }
            llama_batch_free(wrapper->batch);
            delete wrapper;
        }
        llama_backend_free();
    }
}

