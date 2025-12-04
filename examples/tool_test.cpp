#include "tool.h"
#include <iostream>

int main() {
    // Initialize tools
    ncnn::initialize_tools();

    // Test calculator tool
    ncnn::ToolExecutor executor;
    std::unordered_map<std::string, std::string> args;
    args["expression"] = "15+27";

    ncnn::ToolResult result = executor.execute_tool("calculate", args);
    std::cout << "Calculator result: " << (result.success ? result.output : result.error) << std::endl;

    // Test tool call parser
    std::string test_call = "Some text { \"tool\": \"calculate\", \"args\": {\"expression\": \"10*5\"} } more text";
    auto parsed = ncnn::ToolCallParser::parse(test_call);
    if (parsed.valid) {
        std::cout << "Parsed tool: " << parsed.tool_name << std::endl;
        result = executor.execute_tool(parsed.tool_name, parsed.args);
        std::cout << "Parsed tool result: " << (result.success ? result.output : result.error) << std::endl;
    }

    return 0;
}