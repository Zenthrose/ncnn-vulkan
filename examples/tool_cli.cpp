#include "tool.h"
#include <iostream>
#include <string>
#include <vector>
#include <getopt.h>

int main(int argc, char** argv) {
    // Initialize tools
    ncnn::initialize_tools();

    if (argc < 2) {
        std::cout << "NCNN Tool CLI" << std::endl;
        std::cout << "Usage: " << argv[0] << " <tool_name> [args...]" << std::endl;
        std::cout << "Available tools:" << std::endl;

        auto tools = ncnn::ToolRegistry::get_instance().get_available_tools();
        for (const auto& tool_name : tools) {
            auto tool = ncnn::ToolRegistry::get_instance().get_tool(tool_name);
            std::cout << "  " << tool_name << ": " << tool->get_description() << std::endl;
            std::cout << "    Required args: ";
            for (const auto& arg : tool->get_required_args()) {
                std::cout << arg << " ";
            }
            std::cout << std::endl;
            std::cout << "    Optional args: ";
            for (const auto& arg : tool->get_optional_args()) {
                std::cout << arg << " ";
            }
            std::cout << std::endl;
        }
        return 0;
    }

    std::string tool_name = argv[1];
    std::unordered_map<std::string, std::string> args;

    // Parse arguments as key=value pairs
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        size_t eq_pos = arg.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = arg.substr(0, eq_pos);
            std::string value = arg.substr(eq_pos + 1);
            args[key] = value;
        }
    }

    // Execute tool
    ncnn::ToolExecutor executor;
    ncnn::ToolResult result = executor.execute_tool(tool_name, args);

    if (result.success) {
        std::cout << "Success:" << std::endl;
        std::cout << result.output << std::endl;
    } else {
        std::cout << "Error:" << std::endl;
        std::cout << result.error << std::endl;
    }

    return result.success ? 0 : 1;
}