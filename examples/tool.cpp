#include "tool.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <regex>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <signal.h>
#include <algorithm>

namespace ncnn {

// WebSearchTool implementation
ToolResult WebSearchTool::execute(const std::unordered_map<std::string, std::string>& args) {
    ToolResult result;
    auto it = args.find("query");
    if (it == args.end()) {
        result.error = "Missing required argument: query";
        return result;
    }

    std::string query = it->second;

    // Simple mock web search - in real implementation, use web API
    result.success = true;
    result.output = "Mock web search results for: " + query + "\n";
    result.output += "1. Result 1\n";
    result.output += "2. Result 2\n";
    result.output += "3. Result 3\n";

    return result;
}

// CodeExecutionTool implementation
ToolResult CodeExecutionTool::execute(const std::unordered_map<std::string, std::string>& args) {
    ToolResult result;
    auto it = args.find("code");
    if (it == args.end()) {
        result.error = "Missing required argument: code";
        return result;
    }

    std::string code = it->second;
    int timeout = 10; // default timeout
    auto timeout_it = args.find("timeout");
    if (timeout_it != args.end()) {
        timeout = std::stoi(timeout_it->second);
    }

    // Create temporary file for code
    char temp_file[] = "/tmp/ncnn_code_XXXXXX.py";
    int fd = mkstemps(temp_file, 3);
    if (fd == -1) {
        result.error = "Failed to create temporary file";
        return result;
    }

    write(fd, code.c_str(), code.size());
    close(fd);

    // Execute Python code with timeout
    pid_t pid = fork();
    if (pid == 0) {
        // Child process
        execlp("python3", "python3", temp_file, NULL);
        _exit(1);
    } else if (pid > 0) {
        // Parent process
        int status;
        alarm(timeout);
        waitpid(pid, &status, 0);
        alarm(0);

        if (WIFEXITED(status)) {
            result.exit_code = WEXITSTATUS(status);
            result.success = (result.exit_code == 0);
        } else {
            result.error = "Code execution failed or timed out";
        }
    } else {
        result.error = "Failed to fork process";
    }

    // Clean up
    unlink(temp_file);

    return result;
}

// FileAccessTool implementation
ToolResult FileAccessTool::execute(const std::unordered_map<std::string, std::string>& args) {
    ToolResult result;
    auto it = args.find("path");
    if (it == args.end()) {
        result.error = "Missing required argument: path";
        return result;
    }

    std::string path = it->second;
    int max_lines = 100; // default
    auto lines_it = args.find("max_lines");
    if (lines_it != args.end()) {
        max_lines = std::stoi(lines_it->second);
    }

    // Security check - only allow reading from current directory and subdirs
    if (path.find("..") != std::string::npos || path[0] == '/') {
        result.error = "Access denied: path not allowed";
        return result;
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        result.error = "Failed to open file: " + path;
        return result;
    }

    std::string line;
    int count = 0;
    while (std::getline(file, line) && count < max_lines) {
        result.output += line + "\n";
        count++;
    }

    result.success = true;
    return result;
}

// CalculatorTool implementation
ToolResult CalculatorTool::execute(const std::unordered_map<std::string, std::string>& args) {
    ToolResult result;
    auto it = args.find("expression");
    if (it == args.end()) {
        result.error = "Missing required argument: expression";
        return result;
    }

    std::string expr = it->second;

    // Simple calculator - evaluate basic expressions
    // This is a very basic implementation for demo purposes
    try {
        // Remove spaces
        expr.erase(std::remove(expr.begin(), expr.end(), ' '), expr.end());

        // Basic arithmetic operations
        std::regex pattern(R"(([0-9]+(?:\.[0-9]+)?)([\+\-\*\/])([0-9]+(?:\.[0-9]+)?))");
        std::smatch match;
        if (std::regex_match(expr, match, pattern)) {
            double a = std::stod(match[1]);
            char op = match[2].str()[0];
            double b = std::stod(match[3]);

            double res = 0;
            switch (op) {
                case '+': res = a + b; break;
                case '-': res = a - b; break;
                case '*': res = a * b; break;
                case '/':
                    if (b == 0) throw std::runtime_error("Division by zero");
                    res = a / b;
                    break;
                default:
                    throw std::runtime_error("Unsupported operation");
            }

            result.output = std::to_string(res);
            result.success = true;
        } else {
            result.error = "Invalid expression format";
        }
    } catch (const std::exception& e) {
        result.error = std::string("Calculation error: ") + e.what();
    }

    return result;
}

// ToolExecutor implementation
ToolExecutor::ToolExecutor() {
    // Set default security limits
    timeout_seconds_ = 30;
    memory_limit_mb_ = 100;
    allowed_paths_ = {"./", "../"}; // Allow current and parent directory
}

ToolExecutor::~ToolExecutor() {
}

ToolResult ToolExecutor::execute_tool(const std::string& tool_name,
                                     const std::unordered_map<std::string, std::string>& args) {
    Tool* tool = ToolRegistry::get_instance().get_tool(tool_name);
    if (!tool) {
        ToolResult result;
        result.error = "Tool not found: " + tool_name;
        return result;
    }

    // Validate arguments
    auto required = tool->get_required_args();
    for (const auto& req : required) {
        if (args.find(req) == args.end()) {
            ToolResult result;
            result.error = "Missing required argument: " + req;
            return result;
        }
    }

    // For file access tool, check path security
    if (tool_name == "read_file") {
        auto path_it = args.find("path");
        if (path_it != args.end()) {
            std::string path = path_it->second;
            bool allowed = false;
            for (const auto& allowed_path : allowed_paths_) {
                if (path.find(allowed_path) == 0) {
                    allowed = true;
                    break;
                }
            }
            if (!allowed) {
                ToolResult result;
                result.error = "Access denied: path not in allowed list";
                return result;
            }
        }
    }

    // Execute the tool
    return tool->execute(args);
}

// ToolCallParser implementation
ToolCallParser::ParsedCall ToolCallParser::parse(const std::string& text) {
    ParsedCall call;

    // Look for tool call pattern: { "tool": "name", "args": { ... } }
    std::regex pattern("\\{\\s*\"tool\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"args\"\\s*:\\s*\\{([^\\}]*)\\}\\s*\\}");
    std::smatch match;
    if (std::regex_search(text, match, pattern)) {
        call.tool_name = match[1];
        std::string args_str = match[2];

        // Parse arguments: "key": "value"
        std::regex arg_pattern("\"([^\"]+)\"\\s*:\\s*\"([^\"]+)\"");
        std::sregex_iterator iter(args_str.begin(), args_str.end(), arg_pattern);
        std::sregex_iterator end;

        for (; iter != end; ++iter) {
            std::string key = (*iter)[1];
            std::string value = (*iter)[2];
            call.args[key] = value;
        }

        call.valid = true;
    }

    return call;
}

// Initialize built-in tools
static void initialize_builtin_tools() {
    ncnn::ToolRegistry& registry = ncnn::ToolRegistry::get_instance();
    registry.register_tool(std::unique_ptr<ncnn::Tool>(new ncnn::WebSearchTool()));
    registry.register_tool(std::unique_ptr<ncnn::Tool>(new ncnn::CodeExecutionTool()));
    registry.register_tool(std::unique_ptr<ncnn::Tool>(new ncnn::FileAccessTool()));
    registry.register_tool(std::unique_ptr<ncnn::Tool>(new ncnn::CalculatorTool()));
}

// Call this function to initialize tools
void initialize_tools() {
    initialize_builtin_tools();
}

} // namespace ncnn