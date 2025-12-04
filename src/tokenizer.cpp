#include "tokenizer.h"
#include <algorithm>
#include <sstream>
#include <regex>
#include <cctype>

namespace ncnn {

Tokenizer::Tokenizer()
    : bos_token_id(-1), eos_token_id(-1), unk_token_id(-1), pad_token_id(-1)
{
    initialize_byte_encoder();
}

Tokenizer::~Tokenizer()
{
}

void Tokenizer::initialize_byte_encoder()
{
    for (int i = 0; i < 256; ++i) {
        char c = static_cast<char>(i);
        std::string byte_str(1, c);
        byte_encoder.insert(byte_str);
    }
}

bool Tokenizer::load_from_gguf(const std::unordered_map<std::string, std::string>& kv_strings,
                              const std::unordered_map<std::string, std::vector<std::string>>& kv_string_arrays)
{
    // Check tokenizer model type
    auto model_it = kv_strings.find("tokenizer.ggml.model");
    if (model_it != kv_strings.end()) {
        model_type = model_it->second;
    } else {
        // Fallback to basic token ID input
        return true;
    }

    // Load vocab
    auto vocab_it = kv_string_arrays.find("tokenizer.ggml.tokens");
    if (vocab_it != kv_string_arrays.end()) {
        id_to_token = vocab_it->second;
        for (size_t i = 0; i < id_to_token.size(); ++i) {
            vocab[id_to_token[i]] = static_cast<int>(i);
        }
    }

    // Load merges for BPE
    auto merges_it = kv_string_arrays.find("tokenizer.ggml.merges");
    if (merges_it != kv_string_arrays.end()) {
        for (const auto& merge : merges_it->second) {
            size_t space_pos = merge.find(' ');
            if (space_pos != std::string::npos) {
                std::string first = merge.substr(0, space_pos);
                std::string second = merge.substr(space_pos + 1);
                bpe_merges.emplace_back(first, second);
            }
        }
    }

    // Load special tokens
    auto bos_it = kv_strings.find("tokenizer.ggml.bos_token_id");
    if (bos_it != kv_strings.end()) {
        bos_token_id = std::stoi(bos_it->second);
    }

    auto eos_it = kv_strings.find("tokenizer.ggml.eos_token_id");
    if (eos_it != kv_strings.end()) {
        eos_token_id = std::stoi(eos_it->second);
    }

    auto unk_it = kv_strings.find("tokenizer.ggml.unknown_token_id");
    if (unk_it != kv_strings.end()) {
        unk_token_id = std::stoi(unk_it->second);
    }

    auto pad_it = kv_strings.find("tokenizer.ggml.padding_token_id");
    if (pad_it != kv_strings.end()) {
        pad_token_id = std::stoi(pad_it->second);
    }

    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text) const
{
    if (model_type == "gpt2" || !bpe_merges.empty()) {
        // Use BPE encoding
        auto bpe_tokens = bpe_encode(text);
        std::vector<int> tokens;
        for (const auto& token : bpe_tokens) {
            auto it = vocab.find(token);
            if (it != vocab.end()) {
                tokens.push_back(it->second);
            } else if (unk_token_id >= 0) {
                tokens.push_back(unk_token_id);
            }
        }
        return tokens;
    } else {
        // Fallback: assume space-separated token IDs
        std::vector<int> tokens;
        std::stringstream ss(text);
        int id;
        while (ss >> id) {
            tokens.push_back(id);
        }
        return tokens;
    }
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const
{
    if (model_type == "gpt2" || !bpe_merges.empty()) {
        // Use BPE decoding
        std::vector<std::string> token_strings;
        for (int token : tokens) {
            if (token >= 0 && token < static_cast<int>(id_to_token.size())) {
                token_strings.push_back(id_to_token[token]);
            }
        }
        return bpe_decode(token_strings);
    } else {
        // Fallback: just convert to string
        std::stringstream ss;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) ss << " ";
            ss << tokens[i];
        }
        return ss.str();
    }
}

std::string Tokenizer::decode(int token) const
{
    return decode(std::vector<int>{token});
}

std::vector<std::string> Tokenizer::bpe_encode(const std::string& text) const
{
    // Simplified BPE encoding - this is a basic implementation
    // In practice, you'd want a more robust BPE tokenizer
    std::vector<std::string> words;
    std::stringstream ss(text);
    std::string word;
    while (ss >> word) {
        // Convert to bytes
        std::vector<std::string> chars;
        for (char c : word) {
            chars.push_back(std::string(1, c));
        }
        words.insert(words.end(), chars.begin(), chars.end());
    }
    return words;
}

std::string Tokenizer::bpe_decode(const std::vector<std::string>& tokens) const
{
    std::string result;
    for (const auto& token : tokens) {
        result += token;
    }
    return result;
}

} // namespace ncnn