package lib

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// StreamingShimMiddleware converts streaming inference requests to non-streaming
// so bifrost's native MCP agent loop can run. After the agent loop completes,
// it converts the final non-streaming JSON response back to Messages API SSE
// so the compat plugin can translate it to whatever format the client expects.
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

			origStream, _ := req["stream"].(bool)

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

			// Only convert if the original client wanted streaming and the
			// inner handler returned a non-streaming response.
			if !origStream || ctx.Response.StatusCode() != 200 {
				return
			}
			ct := string(ctx.Response.Header.Peek("Content-Type"))
			if strings.Contains(ct, "event-stream") {
				return // already SSE
			}

			var msg map[string]interface{}
			if err := sonic.Unmarshal(ctx.Response.Body(), &msg); err != nil {
				return
			}

			var sseBody []byte
			if _, hasChoices := msg["choices"]; hasChoices {
				sseBody = jsonToOpenAISSE(msg)
			} else {
				sseBody = jsonMsgToSSE(msg)
			}
			ctx.Response.Reset()
			ctx.Response.Header.Set("Content-Type", "text/event-stream; charset=utf-8")
			ctx.Response.Header.Set("Cache-Control", "no-cache")
			ctx.Response.SetBodyStream(bytes.NewReader(sseBody), -1)
		}
	}
}

// jsonMsgToSSE converts a complete Anthropic Messages API JSON response to
// a buffered Messages API SSE byte stream, including thinking blocks.
func jsonMsgToSSE(msg map[string]interface{}) []byte {
	var b strings.Builder

	id, _ := msg["id"].(string)
	model, _ := msg["model"].(string)
	stopReason, _ := msg["stop_reason"].(string)
	if stopReason == "" {
		stopReason = "end_turn"
	}
	usage, _ := msg["usage"].(map[string]interface{})
	if usage == nil {
		usage = map[string]interface{}{}
	}
	content, _ := msg["content"].([]interface{})

	emit := func(eventType string, data interface{}) {
		payload, _ := json.Marshal(data)
		b.WriteString("event: ")
		b.WriteString(eventType)
		b.WriteString("\ndata: ")
		b.Write(payload)
		b.WriteString("\n\n")
	}

	emit("message_start", map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id": id, "type": "message", "role": "assistant",
			"content": []interface{}{}, "model": model,
			"stop_reason": nil, "stop_sequence": nil, "usage": usage,
		},
	})
	emit("ping", map[string]interface{}{"type": "ping"})

	for i, rawBlock := range content {
		block, ok := rawBlock.(map[string]interface{})
		if !ok {
			continue
		}
		blockType, _ := block["type"].(string)

		startBlock := block
		if blockType == "text" {
			startBlock = map[string]interface{}{"type": "text", "text": ""}
		} else if blockType == "thinking" {
			startBlock = map[string]interface{}{"type": "thinking", "thinking": "", "signature": block["signature"]}
		} else if blockType == "tool_use" {
			startBlock = map[string]interface{}{
				"type": blockType, "id": block["id"], "name": block["name"],
			}
		}
		emit("content_block_start", map[string]interface{}{
			"type": "content_block_start", "index": i, "content_block": startBlock,
		})

		switch blockType {
		case "text":
			emit("content_block_delta", map[string]interface{}{
				"type": "content_block_delta", "index": i,
				"delta": map[string]interface{}{"type": "text_delta", "text": block["text"]},
			})
		case "thinking":
			if thinking, ok := block["thinking"].(string); ok && thinking != "" {
				emit("content_block_delta", map[string]interface{}{
					"type": "content_block_delta", "index": i,
					"delta": map[string]interface{}{"type": "thinking_delta", "thinking": thinking},
				})
			}
		case "tool_use":
			if input, ok := block["input"]; ok {
				if inputJSON, err := json.Marshal(input); err == nil {
					emit("content_block_delta", map[string]interface{}{
						"type": "content_block_delta", "index": i,
						"delta": map[string]interface{}{"type": "input_json_delta", "partial_json": string(inputJSON)},
					})
				}
			}
		}
		emit("content_block_stop", map[string]interface{}{"type": "content_block_stop", "index": i})
	}

	outputTokens, _ := usage["output_tokens"]
	emit("message_delta", map[string]interface{}{
		"type": "message_delta",
		"delta": map[string]interface{}{
			"stop_reason": stopReason, "stop_sequence": nil,
		},
		"usage": map[string]interface{}{"output_tokens": outputTokens},
	})
	emit("message_stop", map[string]interface{}{"type": "message_stop"})

	return []byte(b.String())
}

// jsonToOpenAISSE converts a complete OpenAI chat completion JSON response
// to a buffered OpenAI SSE byte stream.
func jsonToOpenAISSE(msg map[string]interface{}) []byte {
	var b strings.Builder

	model, _ := msg["model"].(string)
	choices, _ := msg["choices"].([]interface{})

	if len(choices) > 0 {
		choice, _ := choices[0].(map[string]interface{})
		message, _ := choice["message"].(map[string]interface{})
		finishReason := choice["finish_reason"]

		var text string
		var reasoning string
		if message != nil {
			text, _ = message["content"].(string)
			reasoning, _ = message["reasoning_content"].(string)
		}

		chunk := map[string]interface{}{
			"id":      msg["id"],
			"object":  "chat.completion.chunk",
			"created": msg["created"],
			"model":   model,
			"choices": []interface{}{map[string]interface{}{
				"index":         0,
				"delta":         map[string]interface{}{"role": "assistant", "content": ""},
				"finish_reason": nil,
			}},
		}
		payload, _ := json.Marshal(chunk)
		b.WriteString("data: ")
		b.Write(payload)
		b.WriteString("\n\n")

		if reasoning != "" {
			chunk["choices"] = []interface{}{map[string]interface{}{
				"index":         0,
				"delta":         map[string]interface{}{"reasoning_content": reasoning},
				"finish_reason": nil,
			}}
			payload, _ = json.Marshal(chunk)
			b.WriteString("data: ")
			b.Write(payload)
			b.WriteString("\n\n")
		}

		if text != "" {
			chunk["choices"] = []interface{}{map[string]interface{}{
				"index":         0,
				"delta":         map[string]interface{}{"content": text},
				"finish_reason": nil,
			}}
			payload, _ = json.Marshal(chunk)
			b.WriteString("data: ")
			b.Write(payload)
			b.WriteString("\n\n")
		}

		if toolCalls, ok := message["tool_calls"]; ok && toolCalls != nil {
			chunk["choices"] = []interface{}{map[string]interface{}{
				"index":         0,
				"delta":         map[string]interface{}{"tool_calls": toolCalls},
				"finish_reason": nil,
			}}
			payload, _ = json.Marshal(chunk)
			b.WriteString("data: ")
			b.Write(payload)
			b.WriteString("\n\n")
		}

		chunk["choices"] = []interface{}{map[string]interface{}{
			"index":         0,
			"delta":         map[string]interface{}{},
			"finish_reason": finishReason,
		}}
		if usage, ok := msg["usage"]; ok {
			chunk["usage"] = usage
		}
		payload, _ = json.Marshal(chunk)
		b.WriteString("data: ")
		b.Write(payload)
		b.WriteString("\n\n")
	}

	b.WriteString("data: [DONE]\n\n")
	return []byte(b.String())
}

func isInferencePath(path string) bool {
	return strings.HasSuffix(path, "/messages") ||
		strings.HasSuffix(path, "/messages/")
}
