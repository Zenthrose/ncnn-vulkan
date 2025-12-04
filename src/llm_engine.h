#ifndef LLM_ENGINE_H
#define LLM_ENGINE_H

#include "gguf.h"
#include "tokenizer.h"
#include "mat.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace ncnn {

struct GenerationConfig {
    int max_tokens = 100;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    bool do_sample = true;
    int repetition_penalty = 1.0f;
    std::vector<int> stop_tokens;
};

class LLMEngine {
public:
    LLMEngine();
    ~LLMEngine();

    bool load_model(const std::string& model_path);
    std::vector<int> generate(const std::string& prompt, const GenerationConfig& config = GenerationConfig());
    std::string generate_text(const std::string& prompt, const GenerationConfig& config = GenerationConfig());

    const Tokenizer& get_tokenizer() const { return tokenizer; }

private:
    GGUFLoader loader;
    Tokenizer tokenizer;
    std::unordered_map<std::string, Mat> weights;
    std::string architecture;

    // Model parameters
    int n_layers;
    int n_head;
    int n_kv_head;
    int hidden_size;
    int vocab_size;
    int max_seq_len;

    // KV cache
    std::vector<Mat> key_cache;
    std::vector<Mat> value_cache;

    bool load_weights();
    bool detect_architecture();
    Mat forward(const std::vector<int>& tokens);
    Mat forward_layer(int layer_idx, const Mat& x, int start_pos);
    int sample_token(const float* logits, const GenerationConfig& config, const std::vector<int>& history);

    // Architecture-specific implementations
    Mat forward_llama(const std::vector<int>& tokens);
    Mat forward_gpt2(const std::vector<int>& tokens);
    Mat forward_phi3(const std::vector<int>& tokens);
    Mat forward_mistral(const std::vector<int>& tokens);
    Mat forward_qwen(const std::vector<int>& tokens);
};

} // namespace ncnn

#endif // LLM_ENGINE_H