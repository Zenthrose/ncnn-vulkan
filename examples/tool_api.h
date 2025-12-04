#ifndef TOOL_API_H
#define TOOL_API_H

#include "tool.h"
#include <string>
#include <unordered_map>
#include <functional>

namespace ncnn {

// Simple HTTP request/response structures
struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    std::unordered_map<std::string, std::string> headers;
    std::unordered_map<std::string, std::string> query_params;
};

struct HttpResponse {
    int status_code = 200;
    std::string status_message = "OK";
    std::string body;
    std::unordered_map<std::string, std::string> headers;
};

// Tool API server (simplified - would need actual HTTP server integration)
class ToolAPIServer {
public:
    ToolAPIServer();
    ~ToolAPIServer();

    // Handle API requests
    HttpResponse handle_request(const HttpRequest& request);

private:
    // API endpoints
    HttpResponse list_tools(const HttpRequest& request);
    HttpResponse execute_tool(const HttpRequest& request);
    HttpResponse get_tool_info(const HttpRequest& request);

    // JSON helpers
    std::string to_json(const std::unordered_map<std::string, std::string>& map);
    std::string to_json(const std::vector<std::string>& vec);
    std::string tool_result_to_json(const ToolResult& result);
    std::unordered_map<std::string, std::string> parse_json_args(const std::string& json);
};

} // namespace ncnn

#endif // TOOL_API_H