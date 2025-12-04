# NCNN LLM Platform

A comprehensive, Vulkan-accelerated Large Language Model (LLM) inference platform built on NCNN, providing a complete replacement for tools like llama.cpp, LM Studio, and Ollama.

## Features

### Core Functionality
- Full Tokenization: BPE tokenizer with GGUF metadata extraction, fallback to basic token ID input
- Text Generation: Configurable generation with greedy/top-k/top-p sampling
- KV Caching: Efficient key-value caching for faster inference
- Multi-Model Support: Dynamic layer construction for LLaMA, GPT-2/NeoX/J, Mistral, Qwen, Phi-3, and other architectures
- Multi-Modal Ready: Framework for vision-language models (CLIP integration ready)

### Interfaces
- CLI Tool: Command-line interface for chat, completion, and model management
- Tool CLI: Command-line interface for tool execution and management
- GUI Application: Qt-based graphical interface for model loading, chat, and tool management
- REST API: HTTP API for integration with applications (framework provided)
- C++ API: Direct API for embedding in C++ applications

### Performance & Optimization
- Vulkan Acceleration: GPU-accelerated inference on Vulkan-compatible devices
- Memory Efficient: Optimized memory usage for large models
- Quantization Support: Handles various quantization formats (Q4_0, Q4_1, Q8_0, etc.)
- Cross-Platform: Linux, Windows, macOS, Android, iOS support

### Tool Use System
- Plugin Architecture: Extensible tool framework for web access, code execution, file operations
- Tool Calling: Automatic detection and execution of tool calls in LLM responses
- Sandbox Execution: Secure tool execution with resource limits and timeout handling
- Built-in Tools: Calculator, web search, Python code execution, file reading
- Safety Measures: Input validation, path restrictions, memory limits

### Robustness
- Error Handling: Comprehensive error handling and validation
- GPU Detection: Automatic GPU detection and fallback
- Model Validation: Checks model integrity and compatibility

## Quick Start

### Prerequisites
- NCNN library (built with Vulkan support)
- C++11 compatible compiler
- Vulkan-compatible GPU (optional, CPU fallback available)

### Building

mkdir build && cd build
cmake .. -DNCNN_VULKAN=ON -DBUILD_GUI=ON
make -j$(nproc)

# Build examples
make llm_platform_demo llm_cli llm_gui

### Demo

cd examples
./llm_platform_demo

Output:
```
NCNN LLM Platform Demo
======================
Prompt: hello world
Response: hello <eos> how you ?

Platform Features Demonstrated:
- Tokenization (BPE fallback)
- Text generation with sampling
- Configurable parameters
- Multi-model architecture support (framework)
- Vulkan acceleration ready
```

### CLI Usage

./llm_cli --model models/Phi-3-mini-4k-instruct-Q4_K_M.gguf --prompt "Hello, how are you?" --max-tokens 50 --temperature 0.7

### GUI Usage

# Launch the graphical interface
./llm_gui

GUI Features:
- Model Management: Load local models or download from URLs
- Interactive Chat: Real-time conversation with the AI
- Tool Integration: Enable/disable tools, monitor execution
- Parameter Controls: Adjust temperature, max tokens, sampling methods
- Progress Monitoring: Loading progress and generation status
- Error Handling: User-friendly error messages and recovery

### Tool System

#### Tool CLI

# List available tools
./tool_cli

# Execute a tool
./tool_cli calculate expression="2+3*4"
./tool_cli read_file path="README.md" max_lines=10
./tool_cli execute_code code="print('Hello from Python!')"
./tool_cli web_search query="latest AI developments"

#### Web Search Tool
The web search tool uses DuckDuckGo's Instant Answer API, requiring no API key:
- Provides instant answers for factual queries
- Returns abstracts and summaries
- Includes related topics when available
- Completely free and accessible

#### Tool Integration in LLM

The LLM platform automatically detects tool calls in responses and executes them:

// Tool calls are detected in JSON format: {"tool": "name", "args": {...}}
std::string response = llm.generate_text("Calculate 15 + 27");
auto tool_call = ToolCallParser::parse(response);
if (tool_call.valid) {
    ToolExecutor executor;
    ToolResult result = executor.execute_tool(tool_call.tool_name, tool_call.args);
}

## Architecture

### Components

1. Tokenizer (src/tokenizer.h/cpp)
   - Loads BPE tokenizer from GGUF metadata
   - Fallback to basic token ID parsing
   - Encode/decode text to/from token sequences

2. LLM Engine (src/llm_engine.h/cpp)
   - Loads and manages GGUF models
   - Architecture detection and dynamic layer construction
   - Forward pass implementation for different model types
   - Sampling and generation logic

3. GGUF Loader (src/gguf.h/cpp)
   - Parses GGUF format files
   - Extracts tensors, metadata, and tokenizer information
   - Dequantization support for various formats

### Supported Architectures

- Phi-3: Microsoft's Phi-3 models
- LLaMA: Meta's LLaMA models
- Mistral: Mistral AI models
- Qwen: Alibaba's Qwen models
- GPT-2: OpenAI's GPT-2 architecture
- Extensible: Framework for adding new architectures

### Generation Features

- Sampling Methods:
  - Greedy decoding
  - Top-k sampling
  - Top-p (nucleus) sampling
  - Temperature scaling

- Parameters:
  - Max tokens
  - Temperature
  - Top-k, Top-p values
  - Repetition penalty
  - Stop tokens

## API Reference

### C++ API

#include "llm_engine.h"

ncnn::LLMEngine engine;
engine.load_model("model.gguf");

ncnn::GenerationConfig config;
config.max_tokens = 100;
config.temperature = 0.8f;
config.top_p = 0.9f;

std::string response = engine.generate_text("Hello, world!", config);

### REST API (Planned)

# Start server
./llm_server --model model.gguf --port 8080

# Chat completion
curl -X POST http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
    "temperature": 0.7
  }'

## Model Support

### Currently Tested Models
- Phi-3 Mini 4K Instruct (Q4_K_M)
- Mistral 7B Instruct v0.1 (Q4_0)
- Qwen 1.5B Instruct (Q4_0)

### Model Responsibility
**Important**: Users are responsible for downloading their own AI models. This repository does not include model files due to their large size and licensing considerations.

#### Where to Get Models
- Hugging Face: https://huggingface.co/models?pipeline_tag=text-generation
- GGUF Models: Search for models with GGUF format
- Recommended Models:
  - microsoft/Phi-3-mini-4k-instruct (GGUF format)
  - mistralai/Mistral-7B-Instruct-v0.1 (GGUF format)
  - Qwen/Qwen2-1.5B-Instruct (GGUF format)

#### Downloading Models
# Example using huggingface-hub (install with: pip install huggingface-hub)
huggingface-cli download microsoft/Phi-3-mini-4k-instruct Phi-3-mini-4k-instruct-Q4_K_M.gguf --local-dir models/

# Or download manually from Hugging Face and place in models/ directory

### Adding New Architectures

1. Add architecture detection in LLMEngine::detect_architecture()
2. Implement forward pass in LLMEngine::forward_*() methods
3. Update parameter extraction for the new architecture

## Performance Tuning

### Vulkan Optimization
- Ensure Vulkan SDK is installed
- Use latest GPU drivers
- Set opt.use_vulkan_compute = true in Option

### Memory Optimization
- Use quantized models (Q4_K_M recommended)
- Enable KV caching for sequential generation
- Batch processing for multiple requests

### CPU Fallback
- Automatic fallback when Vulkan unavailable
- Multi-threading support
- AVX/AVX2 optimizations

## Testing

# Run basic tests
make test
./test_llm_engine

# Benchmark performance
./benchmark_llm --model model.gguf --prompts test_prompts.txt

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure Vulkan acceleration works
5. Submit pull request

### Adding New Models
- Test on multiple GPUs
- Verify quantization compatibility
- Add model-specific tests
- Update documentation

## License

This project follows the same license as NCNN.

## Roadmap

- [x] Tool use system implementation
- [x] GGUF loader fixes and compilation issues resolved
- [x] GUI application with model loading and chat interface
- [x] Web search tool with real API (no key required)
- [ ] REST API implementation
- [ ] Multi-modal support (CLIP integration)
- [ ] Streaming generation
- [ ] Model quantization tools
- [ ] Performance profiling tools
- [ ] Web UI interface with tool integration
- [ ] Mobile deployment optimizations
