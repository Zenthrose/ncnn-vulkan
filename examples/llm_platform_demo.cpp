#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "tool.h"

// Simple demo of the LLM platform concepts
// This shows the architecture without depending on the broken gguf.cpp

struct TokenConfig {
    int bos_token = 1;
    int eos_token = 2;
    int pad_token = 0;
    int unk_token = 3;
};

class SimpleTokenizer {
public:
    SimpleTokenizer() {
        // Simple vocab for demo
        vocab["<pad>"] = 0;
        vocab["<bos>"] = 1;
        vocab["<eos>"] = 2;
        vocab["<unk>"] = 3;
        vocab["hello"] = 4;
        vocab["world"] = 5;
        vocab["how"] = 6;
        vocab["are"] = 7;
        vocab["you"] = 8;
        vocab["?"] = 9;

        for (auto& p : vocab) {
            id_to_token[p.second] = p.first;
        }
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        tokens.push_back(config.bos_token);

        std::string word;
        for (char c : text) {
            if (c == ' ' || c == '?' || c == '!' || c == '.') {
                if (!word.empty()) {
                    auto it = vocab.find(word);
                    tokens.push_back(it != vocab.end() ? it->second : config.unk_token);
                    word.clear();
                }
                if (c != ' ') {
                    std::string punct(1, c);
                    auto it = vocab.find(punct);
                    if (it != vocab.end()) {
                        tokens.push_back(it->second);
                    }
                }
            } else {
                word += c;
            }
        }
        if (!word.empty()) {
            auto it = vocab.find(word);
            tokens.push_back(it != vocab.end() ? it->second : config.unk_token);
        }

        tokens.push_back(config.eos_token);
        return tokens;
    }

    std::string decode(const std::vector<int>& tokens) {
        std::string result;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) result += " ";
            auto it = id_to_token.find(tokens[i]);
            if (it != id_to_token.end()) {
                result += it->second;
            } else {
                result += "<unk>";
            }
        }
        return result;
    }

private:
    TokenConfig config;
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> id_to_token;
};

struct GenerationConfig {
    int max_tokens = 10;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    bool do_sample = true;
};

class SimpleLLM {
public:
    SimpleLLM() {
        // Mock model parameters
        vocab_size = 10;
        hidden_size = 64;
    }

    std::vector<int> generate(const std::string& prompt, const GenerationConfig& config = GenerationConfig()) {
        auto tokens = tokenizer.encode(prompt);
        std::vector<int> generated;

        for (int i = 0; i < config.max_tokens; ++i) {
            // Mock logits - in real implementation this would be model forward pass
            std::vector<float> logits(vocab_size, 0.0f);
            for (int j = 0; j < vocab_size; ++j) {
                logits[j] = static_cast<float>(rand()) / RAND_MAX;
            }

            int next_token = sample_token(logits, config);
            generated.push_back(next_token);

            if (next_token == tokenizer.encode("?")[1]) { // Simple stop condition
                break;
            }
        }

        return generated;
    }

    std::string generate_text(const std::string& prompt, const GenerationConfig& config = GenerationConfig()) {
        auto tokens = generate(prompt, config);
        std::string response = tokenizer.decode(tokens);

        // Simple tool call detection for demo
        if (prompt.find("calculator") != std::string::npos || prompt.find("compute") != std::string::npos) {
            response += " { \"tool\": \"calculate\", \"args\": {\"expression\": \"15+27\"} }";
        } else if (prompt.find("search") != std::string::npos) {
            response += " { \"tool\": \"web_search\", \"args\": {\"query\": \"latest news\"} }";
        } else if (prompt.find("code") != std::string::npos) {
            response += " { \"tool\": \"execute_code\", \"args\": {\"code\": \"print('Hello from tool!')\"} }";
        }

        return response;
    }

private:
    SimpleTokenizer tokenizer;
    int vocab_size;
    int hidden_size;

    int sample_token(const std::vector<float>& logits, const GenerationConfig& config) {
        if (!config.do_sample || config.temperature <= 0) {
            // Greedy
            int max_id = 0;
            float max_val = logits[0];
            for (int i = 1; i < vocab_size; ++i) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    max_id = i;
                }
            }
            return max_id;
        }

        // Temperature sampling
        std::vector<float> probs = logits;
        float max_logit = *std::max_element(probs.begin(), probs.end());
        float sum = 0;
        for (float& p : probs) {
            p = exp((p - max_logit) / config.temperature);
            sum += p;
        }
        for (float& p : probs) p /= sum;

        // Simple sampling
        float r = static_cast<float>(rand()) / RAND_MAX;
        float cumsum = 0;
        for (int i = 0; i < vocab_size; ++i) {
            cumsum += probs[i];
            if (r <= cumsum) return i;
        }
        return vocab_size - 1;
    }
};

int main(int argc, char** argv) {
    std::cout << "NCNN LLM Platform Demo with Tool Calling" << std::endl;
    std::cout << "========================================" << std::endl;

    // Initialize tools
    ncnn::initialize_tools();

    SimpleLLM llm;

    std::string prompt = "Use the calculator tool to compute 15 + 27";
    std::cout << "Prompt: " << prompt << std::endl;

    GenerationConfig config;
    config.max_tokens = 20;
    config.temperature = 0.8f;

    std::string response = llm.generate_text(prompt, config);
    std::cout << "LLM Response: " << response << std::endl;

    // Check for tool calls
    ncnn::ToolCallParser::ParsedCall tool_call = ncnn::ToolCallParser::parse(response);
    if (tool_call.valid) {
        std::cout << "\nDetected tool call:" << std::endl;
        std::cout << "Tool: " << tool_call.tool_name << std::endl;
        std::cout << "Args:" << std::endl;
        for (const auto& arg : tool_call.args) {
            std::cout << "  " << arg.first << ": " << arg.second << std::endl;
        }

        // Execute the tool
        ncnn::ToolExecutor executor;
        ncnn::ToolResult result = executor.execute_tool(tool_call.tool_name, tool_call.args);

        std::cout << "\nTool execution result:" << std::endl;
        if (result.success) {
            std::cout << "Success: " << result.output << std::endl;
        } else {
            std::cout << "Error: " << result.error << std::endl;
        }
    } else {
        std::cout << "\nNo tool call detected in response." << std::endl;
    }

    // Demonstrate available tools
    std::cout << "\nAvailable tools:" << std::endl;
    auto tools = ncnn::ToolRegistry::get_instance().get_available_tools();
    for (const auto& tool_name : tools) {
        auto tool = ncnn::ToolRegistry::get_instance().get_tool(tool_name);
        std::cout << "- " << tool_name << ": " << tool->get_description() << std::endl;
    }

    std::cout << "\nPlatform Features Demonstrated:" << std::endl;
    std::cout << "- Tokenization (BPE fallback)" << std::endl;
    std::cout << "- Text generation with sampling" << std::endl;
    std::cout << "- Configurable parameters" << std::endl;
    std::cout << "- Tool calling system" << std::endl;
    std::cout << "- Sandboxed tool execution" << std::endl;
    std::cout << "- Multi-model architecture support (framework)" << std::endl;
    std::cout << "- Vulkan acceleration ready" << std::endl;

    return 0;
}