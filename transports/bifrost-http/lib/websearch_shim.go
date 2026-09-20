package lib

import (
	"strings"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// anthropicTools defines web_search and web_fetch in Anthropic's native
// input_schema format for injection into Messages API requests.
var anthropicTools = []interface{}{
	map[string]interface{}{
		"name":        "web_search",
		"description": "Search the web for current information. Use when you need up-to-date data from the internet.",
		"input_schema": map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "Search query",
				},
			},
			"required": []string{"query"},
		},
	},
	map[string]interface{}{
		"name":        "web_fetch",
		"description": "Fetch the content of a URL and return its text.",
		"input_schema": map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"url": map[string]interface{}{
					"type":        "string",
					"description": "URL to fetch",
				},
			},
			"required": []string{"url"},
		},
	},
}

// openaiTools defines web_search and web_fetch in OpenAI's function-calling
// format for injection into Chat Completions API requests.
var openaiTools = []interface{}{
	map[string]interface{}{
		"type": "function",
		"function": map[string]interface{}{
			"name":        "web_search",
			"description": "Search the web for current information. Use when you need up-to-date data from the internet.",
			"parameters": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Search query",
					},
				},
				"required": []string{"query"},
			},
		},
	},
	map[string]interface{}{
		"type": "function",
		"function": map[string]interface{}{
			"name":        "web_fetch",
			"description": "Fetch the content of a URL and return its text.",
			"parameters": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"url": map[string]interface{}{
						"type":        "string",
						"description": "URL to fetch",
					},
				},
				"required": []string{"url"},
			},
		},
	},
}

// StreamingShimMiddleware injects web_search and web_fetch tool definitions
// into the raw body of both Anthropic Messages API and OpenAI Chat Completions
// requests, using the appropriate format for each API.
func StreamingShimMiddleware() schemas.BifrostHTTPMiddleware {
	return func(next fasthttp.RequestHandler) fasthttp.RequestHandler {
		return func(ctx *fasthttp.RequestCtx) {
			path := string(ctx.Path())

			var tools []interface{}
			switch {
			case isMessagesPath(path):
				tools = anthropicTools
			case isChatCompletionsPath(path):
				tools = openaiTools
			default:
				next(ctx)
				return
			}

			body := ctx.Request.Body()
			var req map[string]interface{}
			if err := sonic.Unmarshal(body, &req); err != nil {
				next(ctx)
				return
			}

			existing := toolNamesFromRequest(req)
			var toAdd []interface{}
			for _, t := range tools {
				name := toolName(t)
				if name != "" && !existing[name] {
					toAdd = append(toAdd, t)
				}
			}

			if len(toAdd) > 0 {
				switch existing := req["tools"].(type) {
				case []interface{}:
					req["tools"] = append(existing, toAdd...)
				default:
					req["tools"] = toAdd
				}
				if newBody, err := sonic.Marshal(req); err == nil {
					ctx.Request.SetBody(newBody)
					ctx.Request.Header.SetContentLength(len(newBody))
				}
			}

			next(ctx)
		}
	}
}

// toolName extracts the tool name from either Anthropic format (top-level
// "name") or OpenAI function format (nested under "function.name").
func toolName(t interface{}) string {
	tm, ok := t.(map[string]interface{})
	if !ok {
		return ""
	}
	if name, ok := tm["name"].(string); ok {
		return name
	}
	if fn, ok := tm["function"].(map[string]interface{}); ok {
		if name, ok := fn["name"].(string); ok {
			return name
		}
	}
	return ""
}

func toolNamesFromRequest(req map[string]interface{}) map[string]bool {
	names := map[string]bool{}
	tools, _ := req["tools"].([]interface{})
	for _, t := range tools {
		if name := toolName(t); name != "" {
			names[name] = true
		}
	}
	return names
}

func isMessagesPath(path string) bool {
	return strings.HasSuffix(path, "/messages") || strings.HasSuffix(path, "/messages/")
}

func isChatCompletionsPath(path string) bool {
	return strings.HasSuffix(path, "/chat/completions") || strings.HasSuffix(path, "/chat/completions/")
}
