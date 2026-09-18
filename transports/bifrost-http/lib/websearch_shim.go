package lib

import (
	"strings"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// StreamingShimMiddleware injects the noAgentBrowser instruction into the
// system prompt for all inference requests. Response passthrough only — no
// stream conversion or response inspection.
func StreamingShimMiddleware() schemas.BifrostHTTPMiddleware {
	return func(next fasthttp.RequestHandler) fasthttp.RequestHandler {
		return func(ctx *fasthttp.RequestCtx) {
			path := string(ctx.Path())
			if !isInferencePath(path) {
				next(ctx)
				return
			}

			body := ctx.Request.Body()
			var req map[string]interface{}
			if err := sonic.Unmarshal(body, &req); err != nil {
				next(ctx)
				return
			}

			const noAgentBrowser = "\n\nIMPORTANT: Do NOT use the agent_browser tool for web searches. Use web_search or web_fetch instead."
			switch sys := req["system"].(type) {
			case string:
				req["system"] = sys + noAgentBrowser
			case []interface{}:
				req["system"] = append(sys, map[string]interface{}{"type": "text", "text": noAgentBrowser})
			case nil:
				req["system"] = noAgentBrowser
			}

			if newBody, err := sonic.Marshal(req); err == nil {
				ctx.Request.SetBody(newBody)
				ctx.Request.Header.SetContentLength(len(newBody))
			}

			next(ctx)
		}
	}
}

func isInferencePath(path string) bool {
	return strings.Contains(path, "/chat/completions") ||
		strings.HasSuffix(path, "/messages") ||
		strings.HasSuffix(path, "/messages/")
}
