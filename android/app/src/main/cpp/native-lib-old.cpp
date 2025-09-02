#include <jni.h>
#include <string>
#include <vector>
#include <cstring> // For std::strcpy
#include "llama.h"

// Simple struct to hold model and context
struct llama_context_wrapper {
    llama_model* model = nullptr;
    llama_context* context = nullptr;
    llama_batch batch;
};

// Helper function to convert C++ string to C char*
// The caller is responsible for freeing the memory.
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
        cparams.n_ctx = 512; // Context size
        cparams.n_threads = 4;
        cparams.n_threads_batch = 4;

        wrapper->context = llama_init_from_model(wrapper->model, cparams);
        if (wrapper->context == nullptr) {
            llama_model_free(wrapper->model);
            delete wrapper;
            return nullptr;
        }
        
        // Allocate the batch
        wrapper->batch = llama_batch_init(cparams.n_ctx, 0, 1);

        return wrapper;
    }

    __attribute__((visibility("default"))) __attribute__((used))
    const char* predict(void* context_ptr, const char* prompt) {
        auto* wrapper = static_cast<llama_context_wrapper*>(context_ptr);
        if (wrapper == nullptr || wrapper->context == nullptr || wrapper->model == nullptr) {
            return string_to_char_ptr("Model not loaded");
        }

        llama_batch_clear(wrapper->batch);
        llama_kv_cache_clear(wrapper->context);

        // Get vocab for tokenization
        const llama_vocab* vocab = llama_get_vocab(wrapper->model);

        // Tokenize the prompt
        std::vector<llama_token> tokens_list;
        tokens_list.resize(llama_n_ctx(wrapper->context));
        
        // Use modern tokenize function signature
        int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens_list.data(), tokens_list.size(), true, false);
        if (n_tokens < 0) {
            return string_to_char_ptr("Failed to tokenize prompt");
        }
        
        // Add prompt tokens to the batch
        for (int i = 0; i < n_tokens; i++) {
            llama_batch_add(wrapper->batch, tokens_list[i], i, { 0 }, false);
        }
        // The last token gets the logits
        wrapper->batch.logits[wrapper->batch.n_tokens - 1] = true;

        // Evaluate the prompt
        if (llama_decode(wrapper->context, wrapper->batch) != 0) {
            return string_to_char_ptr("Failed to evaluate prompt");
        }

        std::string result = "";
        int n_cur = wrapper->batch.n_tokens;
        const int max_tokens = 256;

        // Main generation loop
        while (n_cur <= llama_n_ctx(wrapper->context) && n_cur < n_tokens + max_tokens) {
            const llama_vocab* vocab = llama_get_vocab(wrapper->model);
            auto n_vocab = llama_vocab_n_tokens(vocab);
            auto* logits = llama_get_logits_ith(wrapper->context, wrapper->batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
            }
            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            // Use modern greedy sampling - just select the highest probability token
            llama_token new_token_id = 0;
            float max_logit = candidates[0].logit;
            for (size_t i = 1; i < candidates.size(); i++) {
                if (candidates[i].logit > max_logit) {
                    max_logit = candidates[i].logit;
                    new_token_id = candidates[i].id;
                }
            }

            // Check for end of sequence token
            if (new_token_id == llama_vocab_eos(vocab)) {
                break;
            }

            // Append the token to the result string using modern API
            char piece[64];
            llama_token_to_piece(vocab, new_token_id, piece, sizeof(piece), 0, false);
            result += piece;

            // Prepare for next iteration
            llama_batch_clear(wrapper->batch);
            llama_batch_add(wrapper->batch, new_token_id, n_cur, { 0 }, true);

            if (llama_decode(wrapper->context, wrapper->batch) != 0) {
                break;
            }
            n_cur++;
        }

        return string_to_char_ptr(result);
    }
    
    __attribute__((visibility("default"))) __attribute__((used))
    void free_string(char* str) {
        delete[] str;
    }

    __attribute__((visibility("default"))) __attribute__((used))
    void free_model(void* context_ptr) {
        auto* wrapper = static_cast<llama_context_wrapper*>(context_ptr);
        if (wrapper != nullptr) {
            llama_batch_free(wrapper->batch);
            if (wrapper->context) {
                llama_free(wrapper->context);
            }
            if (wrapper->model) {
                // Use modern API to free model
                llama_model_free(wrapper->model);
            }
            delete wrapper;
        }
    }
}

