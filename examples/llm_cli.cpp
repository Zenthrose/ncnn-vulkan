#include "llm_engine.h"
#include <iostream>
#include <string>
#include <getopt.h>

int main(int argc, char** argv)
{
    std::string model_path;
    std::string prompt;
    int max_tokens = 100;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;

    static struct option long_options[] = {
        {"model", required_argument, 0, 'm'},
        {"prompt", required_argument, 0, 'p'},
        {"max-tokens", required_argument, 0, 't'},
        {"temperature", required_argument, 0, 'T'},
        {"top-p", required_argument, 0, 'P'},
        {"top-k", required_argument, 0, 'k'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:p:t:T:P:k:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'm':
                model_path = optarg;
                break;
            case 'p':
                prompt = optarg;
                break;
            case 't':
                max_tokens = atoi(optarg);
                break;
            case 'T':
                temperature = atof(optarg);
                break;
            case 'P':
                top_p = atof(optarg);
                break;
            case 'k':
                top_k = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Usage: %s --model <model.gguf> --prompt <text> [options]\n", argv[0]);
                return -1;
        }
    }

    if (model_path.empty() || prompt.empty()) {
        fprintf(stderr, "Usage: %s --model <model.gguf> --prompt <text> [options]\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --max-tokens <n>    Maximum tokens to generate (default: 100)\n");
        fprintf(stderr, "  --temperature <f>   Sampling temperature (default: 1.0)\n");
        fprintf(stderr, "  --top-p <f>         Top-p sampling (default: 1.0)\n");
        fprintf(stderr, "  --top-k <n>         Top-k sampling (default: 0)\n");
        return -1;
    }

    ncnn::LLMEngine engine;
    if (!engine.load_model(model_path)) {
        fprintf(stderr, "Failed to load model: %s\n", model_path.c_str());
        return -1;
    }

    ncnn::GenerationConfig config;
    config.max_tokens = max_tokens;
    config.temperature = temperature;
    config.top_p = top_p;
    config.top_k = top_k;

    std::cout << "Generating response..." << std::endl;
    std::string response = engine.generate_text(prompt, config);
    std::cout << "Response: " << response << std::endl;

    return 0;
}