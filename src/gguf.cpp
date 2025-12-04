#include "gguf.h"
#include "mat.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstddef>
#include <unordered_map>
#include <vector>
#include <cstring> // for memcpy

namespace ncnn {

static uint32_t read_u32(const char*& ptr) {
    uint32_t val = 0;
    val |= (unsigned char)ptr[0];
    val |= (unsigned char)ptr[1] << 8;
    val |= (unsigned char)ptr[2] << 16;
    val |= (unsigned char)ptr[3] << 24;
    ptr += 4;
    return val;
}

static uint64_t read_u64(const char*& ptr) {
    uint64_t val = 0;
    val |= (uint64_t)(unsigned char)ptr[0];
    val |= (uint64_t)(unsigned char)ptr[1] << 8;
    val |= (uint64_t)(unsigned char)ptr[2] << 16;
    val |= (uint64_t)(unsigned char)ptr[3] << 24;
    val |= (uint64_t)(unsigned char)ptr[4] << 32;
    val |= (uint64_t)(unsigned char)ptr[5] << 40;
    val |= (uint64_t)(unsigned char)ptr[6] << 48;
    val |= (uint64_t)(unsigned char)ptr[7] << 56;
    ptr += 8;
    return val;
}

static uint16_t read_u16(const char*& ptr) {
    uint16_t val = 0;
    val |= (unsigned char)ptr[0];
    val |= (unsigned char)ptr[1] << 8;
    ptr += 2;
    return val;
}

static size_t ggml_blck_size(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q8_K:
            return 32;
        default:
            return 1;
    }
}

static size_t ggml_type_size(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_Q4_0: return sizeof(uint32_t) + 16;
        case GGML_TYPE_Q4_1: return 2*sizeof(uint32_t) + 16;
        case GGML_TYPE_Q5_0: return 2*sizeof(uint32_t) + 20;
        case GGML_TYPE_Q5_1: return 3*sizeof(uint32_t) + 20;
        case GGML_TYPE_Q8_0: return sizeof(uint32_t) + 32;
        case GGML_TYPE_Q4_K: return 84;
        case GGML_TYPE_Q5_K: return 92;
        case GGML_TYPE_Q6_K: return 48 + 24;
        case GGML_TYPE_Q8_K: return 2*sizeof(uint32_t) + 32;
        default: return 0;
    }
}

size_t gguf_tensor_size(const std::vector<uint64_t>& ne, ggml_type type) {
    uint64_t elements = 1;
    for (auto d : ne) elements *= d;
    if (type < GGML_TYPE_Q4_0) return elements * ggml_type_size(type);
    uint64_t blocks = (elements + ggml_blck_size(type) - 1) / ggml_blck_size(type);
    return blocks * ggml_type_size(type);
}

bool GGUFLoader::load(const char* file_path) {
    fprintf(stderr, "Loading GGUF: %s\n", file_path);

    FILE* fp = fopen(file_path, "rb");
    if (!fp) return false;

    fseek(fp, 0, SEEK_END);
    file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    file_data = (char*)malloc(file_size);
    if (!file_data || fread(file_data, 1, file_size, fp) != file_size) {
        fclose(fp);
        return false;
    }
    fclose(fp);

    const char* ptr = file_data;
    const char* end = file_data + file_size;

    uint32_t magic = read_u32(ptr);
    uint32_t version = read_u32(ptr);
    if (magic != 0x46554747 || (version != 2 && version != 3)) return false;

    uint64_t tensor_count = read_u64(ptr);
    uint64_t kv_count     = read_u64(ptr);

    // Parse KV pairs
    for (uint64_t i = 0; i < kv_count; ++i) {
        uint64_t key_len = read_u64(ptr);
        std::string key_str(ptr, key_len);
        ptr += key_len;

        uint32_t type = read_u32(ptr);

        switch (type) {
            case GGUF_TYPE_UINT8:   this->kv_ints[key_str] = (int64_t)*(uint8_t*)ptr; ptr += 1; break;
            case GGUF_TYPE_INT8:    this->kv_ints[key_str] = (int64_t)*(int8_t*)ptr; ptr += 1; break;
            case GGUF_TYPE_UINT16:  this->kv_ints[key_str] = (int64_t)read_u16(ptr); break;
            case GGUF_TYPE_INT16:   this->kv_ints[key_str] = (int64_t)*(int16_t*)ptr; ptr += 2; break;
            case GGUF_TYPE_UINT32:  this->kv_ints[key_str] = (int64_t)read_u32(ptr); break;
            case GGUF_TYPE_INT32:   this->kv_ints[key_str] = (int64_t)*(int32_t*)ptr; ptr += 4; break;
            case GGUF_TYPE_FLOAT32: this->kv_floats[key_str] = *(float*)ptr; ptr += 4; break;
            case GGUF_TYPE_BOOL:    this->kv_ints[key_str] = (int64_t)*(bool*)ptr; ptr += 1; break;
            case GGUF_TYPE_STRING: {
                uint64_t slen = read_u64(ptr);
                this->kv_strings[key_str] = std::string(ptr, slen);
                ptr += slen;
                break;
            }
            case GGUF_TYPE_ARRAY: {
                uint32_t arr_type = read_u32(ptr);
                uint64_t arr_len  = read_u64(ptr);

                switch (arr_type) {
                    case GGUF_TYPE_UINT8:
                    case GGUF_TYPE_INT8:
                    case GGUF_TYPE_UINT16:
                    case GGUF_TYPE_INT16:
                    case GGUF_TYPE_UINT32:
                    case GGUF_TYPE_INT32:
                    case GGUF_TYPE_UINT64:
                    case GGUF_TYPE_INT64: {
                        std::vector<int32_t> vec;
                        size_t element_size;
                        if (arr_type == GGUF_TYPE_UINT64 || arr_type == GGUF_TYPE_INT64) element_size = 8;
                        else if (arr_type == GGUF_TYPE_UINT32 || arr_type == GGUF_TYPE_INT32) element_size = 4;
                        else if (arr_type == GGUF_TYPE_UINT16 || arr_type == GGUF_TYPE_INT16) element_size = 2;
                        else element_size = 1;
                        for (uint64_t j = 0; j < arr_len; j++) {
                            int64_t val = 0;
                            if (element_size == 8) val = (int64_t)read_u64(ptr);
                            else if (element_size == 4) val = (int32_t)read_u32(ptr);
                            else if (element_size == 2) val = (int16_t)read_u16(ptr);
                            else { val = (int8_t)ptr[0]; ptr += 1; }
                            vec.push_back((int32_t)val);
                        }
                        this->kv_int32_arrays[key_str] = vec;
                        break;
                    }
                    case GGUF_TYPE_FLOAT32: {
                        std::vector<float> vec;
                        for (uint64_t j = 0; j < arr_len; j++) {
                            vec.push_back(*(float*)ptr);
                            ptr += 4;
                        }
                        this->kv_float_arrays[key_str] = vec;
                        break;
                    }
                    case GGUF_TYPE_STRING: {
                        std::vector<std::string> vec;
                        for (uint64_t j = 0; j < arr_len; j++) {
                            uint64_t slen = read_u64(ptr);
                            vec.push_back(std::string(ptr, slen));
                            ptr += slen;
                        }
                        this->kv_string_arrays[key_str] = vec;
                        break;
                    }
                    default: {
                        // skip unsupported array types
                        size_t element_size = 0;
                        switch (arr_type) {
                            case GGUF_TYPE_UINT8:   element_size = 1; break;
                            case GGUF_TYPE_INT8:    element_size = 1; break;
                            case GGUF_TYPE_UINT16:  element_size = 2; break;
                            case GGUF_TYPE_INT16:   element_size = 2; break;
                            case GGUF_TYPE_UINT32:  element_size = 4; break;
                            case GGUF_TYPE_INT32:   element_size = 4; break;
                            case GGUF_TYPE_FLOAT32: element_size = 4; break;
                            case GGUF_TYPE_BOOL:    element_size = 1; break;
                            case GGUF_TYPE_UINT64:  element_size = 8; break;
                            case GGUF_TYPE_INT64:   element_size = 8; break;
                            case GGUF_TYPE_FLOAT64: element_size = 8; break;
                            case GGUF_TYPE_STRING: {
                                for (uint64_t j = 0; j < arr_len; j++) {
                                    uint64_t slen = read_u64(ptr);
                                    ptr += slen;
                                }
                                element_size = 0; // already advanced
                                break;
                            }
                            default:
                                return false;
                        }

                        if (element_size > 0) {
                            size_t bytes = arr_len * element_size;
                            if (ptr + bytes > end) return false;
                            ptr += bytes;
                        }
                        break;
                    }
                }
                break;
            }
            case GGUF_TYPE_UINT64:  this->kv_ints[key_str] = (int64_t)read_u64(ptr); break;
            case GGUF_TYPE_INT64:   this->kv_ints[key_str] = (int64_t)read_u64(ptr); break;
            case GGUF_TYPE_FLOAT64: this->kv_floats[key_str] = (float)*(double*)(ptr); ptr += 8; break;
            default:
                return false;
        }
    }

    // uint32_t alignment = 32; // unused for now

    for (uint64_t i = 0; i < tensor_count; ++i) {
        std::string name_str;
        // GGUF spec: name is gguf_string_t: uint64 len + bytes
        uint64_t name_len = read_u64(ptr);
        name_str = std::string(ptr, name_len);
        ptr += name_len;

        uint32_t n_dims = read_u32(ptr);
        std::vector<uint64_t> ne(n_dims);
        for (uint32_t j = 0; j < n_dims; ++j) ne[j] = read_u64(ptr);

        ggml_type type = (ggml_type)read_u32(ptr);
        uint64_t tensor_offset = read_u64(ptr);

        gguf_tensor t;
        t.name = name_str;
        t.type = type;
        t.ne = ne;
        t.offset = tensor_offset;
        t.size = gguf_tensor_size(ne, type);



        tensor_map[name_str] = t;
    }

    return true;
}

GGUFLoader::~GGUFLoader() {
    if (file_data) free(file_data);
    file_data = nullptr;
    file_size = 0;
    tensor_map.clear();
    kv_strings.clear();
    kv_string_arrays.clear();
    kv_ints.clear();
    kv_floats.clear();
    kv_float_arrays.clear();
    kv_int32_arrays.clear();
}

ncnn::Mat dequant_gguf_tensor(const gguf_tensor& t, const char* file_data) {
    const char* data = file_data + t.offset;
    ncnn::Mat mat;
    uint64_t elements = 1;
    for (auto d : t.ne) elements *= d;
    mat.create(elements);
    float* dst = mat;
    if (t.type == GGML_TYPE_F32) {
        memcpy(dst, data, elements * 4);
    } else if (t.type == GGML_TYPE_F16) {
        const uint16_t* src = (const uint16_t*)data;
        for (uint64_t i = 0; i < elements; i++) {
            uint32_t h = src[i];
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 31;
            uint32_t mant = h & 1023;
            uint32_t f;
            if (exp == 0) {
                f = (mant << 13) | (sign << 31);
            } else if (exp == 31) {
                f = (255 << 23) | (mant << 13) | (sign << 31);
            } else {
                f = ((exp + 112) << 23) | (mant << 13) | (sign << 31);
            }
             memcpy(&dst[i], &f, sizeof(float));
        }
    } else if (t.type == GGML_TYPE_Q4_0) {
        const char* ptr = data;
        uint64_t blocks = (elements + 31) / 32;
        for (uint64_t b = 0; b < blocks; b++) {
            float d = *(const float*)ptr; ptr += 4;
            const uint8_t* q = (const uint8_t*)ptr; ptr += 16;
            for (int i = 0; i < 32 && b*32 + i < elements; i++) {
                int val = (q[i/2] >> (4*(i%2))) & 0xf;
                dst[b*32 + i] = d * (val - 8.0f);
            }
        }
    } else {
        fprintf(stderr, "Unsupported type %d\n", t.type);
    }
    if (t.ne.size() == 2) {
        mat = mat.reshape(t.ne[0], t.ne[1]);
    } else if (t.ne.size() == 1) {
        mat = mat.reshape(t.ne[0]);
    } else {
        // For higher dims, flatten or handle accordingly
        mat = mat.reshape(t.ne[0], t.ne[1]);
    }
    return mat;
}

} // namespace ncnn
