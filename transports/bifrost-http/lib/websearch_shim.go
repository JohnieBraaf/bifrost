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

// StreamingShimMiddleware injects web_search and web_fetch tool definitions
// (in Anthropic-native input_schema format) into the raw body of Anthropic
// Messages API requests so the tools reach Claude before passthrough forwards
// the body to kiro-gateway.
func StreamingShimMiddleware() schemas.BifrostHTTPMiddleware {
	return func(next fasthttp.RequestHandler) fasthttp.RequestHandler {
		return func(ctx *fasthttp.RequestCtx) {
			path := string(ctx.Path())
			if !isMessagesPath(path) {
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
			for _, t := range anthropicTools {
				tm := t.(map[string]interface{})
				if !existing[tm["name"].(string)] {
					toAdd = append(toAdd, t)
				}
			}

			if len(toAdd) > 0 {
				switch tools := req["tools"].(type) {
				case []interface{}:
					req["tools"] = append(tools, toAdd...)
				default:
					req["tools"] = toAdd
				}

				// Inject system prompt guidance so Claude doesn't dump raw results
				// in <web_search> XML blocks or stop without synthesising.
				const webSearchGuidance = "When you use web_search or web_fetch, integrate the results directly into your reply. Do not output raw result lists, do not wrap content in XML tags like <web_search>, and do not stop after showing results — respond as if you simply know the information."
				switch s := req["system"].(type) {
				case string:
					if s != "" {
						req["system"] = s + "\n\n" + webSearchGuidance
					} else {
						req["system"] = webSearchGuidance
					}
				case []interface{}:
					req["system"] = append(s, map[string]interface{}{"type": "text", "text": webSearchGuidance})
				default:
					req["system"] = webSearchGuidance
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

func toolNamesFromRequest(req map[string]interface{}) map[string]bool {
	names := map[string]bool{}
	tools, _ := req["tools"].([]interface{})
	for _, t := range tools {
		if tm, ok := t.(map[string]interface{}); ok {
			if name, ok := tm["name"].(string); ok {
				names[name] = true
			}
		}
	}
	return names
}

func isMessagesPath(path string) bool {
	return strings.HasSuffix(path, "/messages") || strings.HasSuffix(path, "/messages/")
}
