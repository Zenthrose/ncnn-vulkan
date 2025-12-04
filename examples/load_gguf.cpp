#include "gguf.h"
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s [gguf file]\n", argv[0]);
        return -1;
    }

    ncnn::GGUFLoader loader;
    if (!loader.load(argv[1])) {
        fprintf(stderr, "Failed to load GGUF file\n");
        return -1;
    }

    const auto& tensor_map = loader.get_tensor_map();  // <-- correct accessor
    std::cout << "Successfully loaded " << tensor_map.size() << " tensors:\n";

    for (const auto& kv : tensor_map) {
        const ncnn::gguf_tensor& t = kv.second;
        std::cout << "  " << t.name << "  dims:";
        for (uint64_t d : t.ne) std::cout << " " << d;
        std::cout << "  type=" << t.type << "  size=" << t.size << " bytes\n";
    }

    return 0;
}
