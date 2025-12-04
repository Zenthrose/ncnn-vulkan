#ifndef TOOL_H
#define TOOL_H

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>

namespace ncnn {

// Tool result structure
struct ToolResult {
    bool success = false;
    std::string output;
    std::string error;
    int exit_code = 0;
};

// Tool interface
class Tool {
public:
    virtual ~Tool() = default;
    virtual std::string get_name() const = 0;
    virtual std::string get_description() const = 0;
    virtual ToolResult execute(const std::unordered_map<std::string, std::string>& args) = 0;
    virtual std::vector<std::string> get_required_args() const = 0;
    virtual std::vector<std::string> get_optional_args() const = 0;
};

// Tool registry
class ToolRegistry {
public:
    static ToolRegistry& get_instance() {
        static ToolRegistry instance;
        return instance;
    }

    void register_tool(std::unique_ptr<Tool> tool) {
        tools_[tool->get_name()] = std::move(tool);
    }

    Tool* get_tool(const std::string& name) {
        auto it = tools_.find(name);
        return it != tools_.end() ? it->second.get() : nullptr;
    }

    std::vector<std::string> get_available_tools() const {
        std::vector<std::string> names;
        for (const auto& pair : tools_) {
            names.push_back(pair.first);
        }
        return names;
    }

private:
    ToolRegistry() = default;
    std::unordered_map<std::string, std::unique_ptr<Tool>> tools_;
};

// Built-in tools

// Web search tool
class WebSearchTool : public Tool {
public:
    std::string get_name() const override { return "web_search"; }
    std::string get_description() const override {
        return "Search the web for information. Args: query (string)";
    }
    ToolResult execute(const std::unordered_map<std::string, std::string>& args) override;
    std::vector<std::string> get_required_args() const override { return {"query"}; }
    std::vector<std::string> get_optional_args() const override { return {}; }
};

// Code execution tool (Python)
class CodeExecutionTool : public Tool {
public:
    std::string get_name() const override { return "execute_code"; }
    std::string get_description() const override {
        return "Execute Python code. Args: code (string), timeout (int, optional)";
    }
    ToolResult execute(const std::unordered_map<std::string, std::string>& args) override;
    std::vector<std::string> get_required_args() const override { return {"code"}; }
    std::vector<std::string> get_optional_args() const override { return {"timeout"}; }
};

// File system access tool (read-only)
class FileAccessTool : public Tool {
public:
    std::string get_name() const override { return "read_file"; }
    std::string get_description() const override {
        return "Read a file from the filesystem. Args: path (string), max_lines (int, optional)";
    }
    ToolResult execute(const std::unordered_map<std::string, std::string>& args) override;
    std::vector<std::string> get_required_args() const override { return {"path"}; }
    std::vector<std::string> get_optional_args() const override { return {"max_lines"}; }
};

// Calculator tool
class CalculatorTool : public Tool {
public:
    std::string get_name() const override { return "calculate"; }
    std::string get_description() const override {
        return "Perform mathematical calculations. Args: expression (string)";
    }
    ToolResult execute(const std::unordered_map<std::string, std::string>& args) override;
    std::vector<std::string> get_required_args() const override { return {"expression"}; }
    std::vector<std::string> get_optional_args() const override { return {}; }
};

// Tool executor with sandboxing
class ToolExecutor {
public:
    ToolExecutor();
    ~ToolExecutor();

    ToolResult execute_tool(const std::string& tool_name,
                           const std::unordered_map<std::string, std::string>& args);

    // Security settings
    void set_timeout(int seconds) { timeout_seconds_ = seconds; }
    void set_memory_limit(size_t mb) { memory_limit_mb_ = mb; }
    void set_allowed_paths(const std::vector<std::string>& paths) { allowed_paths_ = paths; }

private:
    int timeout_seconds_ = 30;
    size_t memory_limit_mb_ = 100;
    std::vector<std::string> allowed_paths_;
};

// Tool call parser
class ToolCallParser {
public:
    struct ParsedCall {
        std::string tool_name;
        std::unordered_map<std::string, std::string> args;
        bool valid = false;
    };

    static ParsedCall parse(const std::string& text);
};

// Initialize built-in tools
void initialize_tools();

} // namespace ncnn

#endif // TOOL_H