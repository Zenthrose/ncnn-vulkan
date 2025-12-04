#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace ncnn {

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    bool load_from_gguf(const std::unordered_map<std::string, std::string>& kv_strings,
                        const std::unordered_map<std::string, std::vector<std::string>>& kv_string_arrays);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    std::string decode(int token) const;

    int bos_token() const { return bos_token_id; }
    int eos_token() const { return eos_token_id; }
    int unk_token() const { return unk_token_id; }
    int pad_token() const { return pad_token_id; }

private:
    std::string model_type;
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> merges; // for BPE
    int bos_token_id;
    int eos_token_id;
    int unk_token_id;
    int pad_token_id;

    // BPE helpers
    std::vector<std::pair<std::string, std::string>> bpe_merges;
    std::unordered_set<std::string> byte_encoder;

    void initialize_byte_encoder();
    std::vector<std::string> bpe_encode(const std::string& text) const;
    std::string bpe_decode(const std::vector<std::string>& tokens) const;
};

} // namespace ncnn

#endif // TOKENIZER_H