#ifndef GGUF_H
#define GGUF_H

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace ncnn {
class Mat;
}

namespace ncnn {

enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
};

enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

struct gguf_tensor {
    std::string name;
    ggml_type type;
    std::vector<uint64_t> ne;  // tensor dimensions
    uint64_t offset;           // offset in file (relative to file start)
    size_t size;               // size in bytes
};

class GGUFLoader {
public:
    GGUFLoader() : file_data(nullptr), file_size(0) {}
    ~GGUFLoader();

    bool load(const char* file_path);

    const gguf_tensor* get_tensor(const std::string& name) const {
        auto it = tensor_map.find(name);
        return it != tensor_map.end() ? &it->second : nullptr;
    }

    const std::unordered_map<std::string, gguf_tensor>& get_tensor_map() const {
        return tensor_map;
    }

    size_t get_file_size() const { return file_size; }
    const char* get_file_data() const { return file_data; }

    const std::unordered_map<std::string, std::string>& get_kv_strings() const { return kv_strings; }
    const std::unordered_map<std::string, std::vector<std::string>>& get_kv_string_arrays() const { return kv_string_arrays; }
    const std::unordered_map<std::string, int64_t>& get_kv_ints() const { return kv_ints; }
    const std::unordered_map<std::string, float>& get_kv_floats() const { return kv_floats; }
    const std::unordered_map<std::string, std::vector<float>>& get_kv_float_arrays() const { return kv_float_arrays; }
    const std::unordered_map<std::string, std::vector<int32_t>>& get_kv_int32_arrays() const { return kv_int32_arrays; }

private:
    char* file_data;
    size_t file_size;
    std::unordered_map<std::string, gguf_tensor> tensor_map;
    std::unordered_map<std::string, std::string> kv_strings;
    std::unordered_map<std::string, std::vector<std::string>> kv_string_arrays;
    std::unordered_map<std::string, int64_t> kv_ints;
    std::unordered_map<std::string, float> kv_floats;
    std::unordered_map<std::string, std::vector<float>> kv_float_arrays;
    std::unordered_map<std::string, std::vector<int32_t>> kv_int32_arrays;
};

ncnn::Mat dequant_gguf_tensor(const gguf_tensor& t, const char* file_data);

} // namespace ncnn

#endif // GGUF_H
