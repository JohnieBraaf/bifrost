//go:build !tinygo && !wasm

package mcp

import (
	"strings"
	"sync"

	"github.com/google/uuid"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
)

// WrapResponsesStreamWithAgentLoop wraps a Responses API stream with transparent
// MCP tool-call execution. Each agent turn is buffered; if the turn ends with tool
// calls all auto-executable tools are run in parallel and a follow-up stream is
// requested. The final turn (no tool calls) is forwarded directly to the returned
// channel so the client receives it as a live stream.
func (m *MCPManager) WrapResponsesStreamWithAgentLoop(
	ctx *schemas.BifrostContext,
	originalReq *schemas.BifrostResponsesRequest,
	initialStream chan *schemas.BifrostStreamChunk,
	makeStream func(*schemas.BifrostContext, *schemas.BifrostResponsesRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError),
) chan *schemas.BifrostStreamChunk {
	maxDepth := int(m.toolsManager.maxAgentDepth.Load())
	out := make(chan *schemas.BifrostStreamChunk, 32)
	go func() {
		defer close(out)
		req := cloneResponsesRequest(originalReq)
		m.runResponsesStreamAgentLoop(ctx, req, initialStream, makeStream, out, 0, maxDepth)
	}()
	return out
}

// WrapChatStreamWithAgentLoop wraps a Chat Completions stream with transparent
// MCP tool-call execution. Same buffering strategy as WrapResponsesStreamWithAgentLoop.
func (m *MCPManager) WrapChatStreamWithAgentLoop(
	ctx *schemas.BifrostContext,
	originalReq *schemas.BifrostChatRequest,
	initialStream chan *schemas.BifrostStreamChunk,
	makeStream func(*schemas.BifrostContext, *schemas.BifrostChatRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError),
) chan *schemas.BifrostStreamChunk {
	maxDepth := int(m.toolsManager.maxAgentDepth.Load())
	out := make(chan *schemas.BifrostStreamChunk, 32)
	go func() {
		defer close(out)
		req := cloneChatRequest(originalReq)
		m.runChatStreamAgentLoop(ctx, req, initialStream, makeStream, out, 0, maxDepth)
	}()
	return out
}

// runResponsesStreamAgentLoop is the recursive inner loop for the Responses API path.
func (m *MCPManager) runResponsesStreamAgentLoop(
	ctx *schemas.BifrostContext,
	req *schemas.BifrostResponsesRequest,
	stream chan *schemas.BifrostStreamChunk,
	makeStream func(*schemas.BifrostContext, *schemas.BifrostResponsesRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError),
	out chan *schemas.BifrostStreamChunk,
	depth, maxDepth int,
) {
	// Drain the stream into a buffer.
	var buf []*schemas.BifrostStreamChunk
	for chunk := range stream {
		if chunk != nil {
			buf = append(buf, chunk)
		}
	}

	// Extract tool calls from the buffered chunks.
	toolCalls := extractToolCallsFromResponsesStream(buf)

	// No tool calls or depth limit reached — forward the buffer as a live stream and stop.
	if len(toolCalls) == 0 || depth >= maxDepth {
		for _, chunk := range buf {
			out <- chunk
		}
		return
	}

	// Separate auto-executable from non-auto-executable.
	var autoExec, nonAutoExec []schemas.ChatAssistantMessageToolCall
	for _, tc := range toolCalls {
		if tc.Function.Name == nil {
			nonAutoExec = append(nonAutoExec, tc)
			continue
		}
		client := m.GetClientForTool(*tc.Function.Name)
		if client != nil && canAutoExecuteTool(*tc.Function.Name, client.ExecutionConfig) {
			autoExec = append(autoExec, tc)
		} else {
			nonAutoExec = append(nonAutoExec, tc)
		}
	}

	// If nothing can be auto-executed just forward the buffer as-is.
	if len(autoExec) == 0 {
		for _, chunk := range buf {
			out <- chunk
		}
		return
	}

	// Execute auto-executable tools in parallel.
	toolResults := m.executeToolsParallel(ctx, autoExec)

	// Build follow-up request using the Responses adapter helpers.
	fakeResp := reconstructResponsesResponseFromStream(buf)
	adapter := &responsesAPIAdapter{
		originalReq:     req,
		initialResponse: fakeResp,
		makeReq: func(_ *schemas.BifrostContext, _ *schemas.BifrostResponsesRequest) (*schemas.BifrostResponsesResponse, *schemas.BifrostError) {
			return nil, nil
		},
	}
	history := adapter.getConversationHistory()
	history = adapter.addAssistantMessage(history, fakeResp)
	history = adapter.addToolResults(history, toolResults)
	newReq := adapter.createNewRequest(history).(*schemas.BifrostResponsesRequest)

	// Clear passthrough flags so follow-up requests use bifrost's native serialization,
	// which the route converter will then render into the appropriate wire format.
	prepareFollowUpContext(ctx)

	followUpStream, bifrostErr := makeStream(ctx, newReq)
	if bifrostErr != nil {
		out <- &schemas.BifrostStreamChunk{BifrostError: bifrostErr}
		return
	}

	// Non-auto-executable tools remain in the response — drain follow-up and return.
	if len(nonAutoExec) > 0 {
		for chunk := range followUpStream {
			if chunk != nil {
				out <- chunk
			}
		}
		return
	}

	m.runResponsesStreamAgentLoop(ctx, newReq, followUpStream, makeStream, out, depth+1, maxDepth)
}

// runChatStreamAgentLoop is the recursive inner loop for the Chat Completions path.
func (m *MCPManager) runChatStreamAgentLoop(
	ctx *schemas.BifrostContext,
	req *schemas.BifrostChatRequest,
	stream chan *schemas.BifrostStreamChunk,
	makeStream func(*schemas.BifrostContext, *schemas.BifrostChatRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError),
	out chan *schemas.BifrostStreamChunk,
	depth, maxDepth int,
) {
	var buf []*schemas.BifrostStreamChunk
	for chunk := range stream {
		if chunk == nil {
			continue
		}
		// Reasoning-only chunks can never contain tool calls — pass them
		// through immediately so the client sees thinking tokens in real time.
		if isChatReasoningOnlyChunk(chunk) {
			out <- chunk
			continue
		}
		buf = append(buf, chunk)
	}

	toolCalls := extractToolCallsFromChatStream(buf)

	if len(toolCalls) == 0 || depth >= maxDepth {
		for _, chunk := range buf {
			out <- chunk
		}
		return
	}

	var autoExec, nonAutoExec []schemas.ChatAssistantMessageToolCall
	for _, tc := range toolCalls {
		if tc.Function.Name == nil {
			nonAutoExec = append(nonAutoExec, tc)
			continue
		}
		client := m.GetClientForTool(*tc.Function.Name)
		if client != nil && canAutoExecuteTool(*tc.Function.Name, client.ExecutionConfig) {
			autoExec = append(autoExec, tc)
		} else {
			nonAutoExec = append(nonAutoExec, tc)
		}
	}

	if len(autoExec) == 0 {
		for _, chunk := range buf {
			out <- chunk
		}
		return
	}

	toolResults := m.executeToolsParallel(ctx, autoExec)

	fakeResp := reconstructChatResponseFromStream(buf)
	adapter := &chatAPIAdapter{
		originalReq:     req,
		initialResponse: fakeResp,
		makeReq: func(_ *schemas.BifrostContext, _ *schemas.BifrostChatRequest) (*schemas.BifrostChatResponse, *schemas.BifrostError) {
			return nil, nil
		},
	}
	history := adapter.getConversationHistory()
	history = adapter.addAssistantMessage(history, fakeResp)
	history = adapter.addToolResults(history, toolResults)
	newReq := adapter.createNewRequest(history).(*schemas.BifrostChatRequest)

	prepareFollowUpContext(ctx)

	followUpStream, bifrostErr := makeStream(ctx, newReq)
	if bifrostErr != nil {
		out <- &schemas.BifrostStreamChunk{BifrostError: bifrostErr}
		return
	}

	if len(nonAutoExec) > 0 {
		for chunk := range followUpStream {
			if chunk != nil {
				out <- chunk
			}
		}
		return
	}

	m.runChatStreamAgentLoop(ctx, newReq, followUpStream, makeStream, out, depth+1, maxDepth)
}

// executeToolsParallel runs a set of tool calls concurrently and returns their results
// as ChatMessage values, compatible with both Chat and Responses adapter helpers.
func (m *MCPManager) executeToolsParallel(
	ctx *schemas.BifrostContext,
	toolCalls []schemas.ChatAssistantMessageToolCall,
) []*schemas.ChatMessage {
	results := make([]*schemas.ChatMessage, len(toolCalls))
	var wg sync.WaitGroup
	wg.Add(len(toolCalls))
	for i, tc := range toolCalls {
		go func(i int, tc schemas.ChatAssistantMessageToolCall) {
			defer wg.Done()
			toolCtx := schemas.NewBifrostContext(ctx, schemas.NoDeadline)
			toolCtx.SetValue(schemas.BifrostContextKeyMCPLogID, uuid.New().String())
			req := &schemas.BifrostMCPRequest{
				RequestType:                  schemas.MCPRequestTypeChatToolCall,
				ChatAssistantMessageToolCall: &tc,
			}
			resp, err := m.executeToolForAgent(toolCtx, req)
			if err != nil || resp == nil || resp.ChatMessage == nil {
				results[i] = createToolResultMessage(tc, "", err)
			} else {
				results[i] = resp.ChatMessage
			}
		}(i, tc)
	}
	wg.Wait()
	return results
}

// prepareFollowUpContext clears only UseRawRequestBody so bifrost serializes
// the new request body, while preserving SendBackRawResponse + kiro OAuth headers.
// It also clears stale stream state from the previous request so the follow-up
// gets a clean HTTP connection instead of reusing the one vLLM just closed.
func prepareFollowUpContext(ctx *schemas.BifrostContext) {
	ctx.SetValue(schemas.BifrostContextKeyUseRawRequestBody, false)
	// Clear stream-lifecycle flags that fasthttp/bifrost set on the previous
	// connection. Without this, the follow-up sees ConnectionClosed=true and
	// fasthttp immediately returns ErrConnectionClosed ("stream closed").
	ctx.ClearValue(schemas.BifrostContextKeyConnectionClosed)
	ctx.ClearValue(schemas.BifrostContextKeyStreamEndIndicator)
	ctx.ClearValue(schemas.BifrostContextKeyStreamBodyExhausted)
}

// isChatReasoningOnlyChunk returns true for chunks that carry only reasoning
// tokens and no content, tool calls, or finish reason. These can never trigger
// tool execution and should be forwarded immediately rather than buffered.
func isChatReasoningOnlyChunk(chunk *schemas.BifrostStreamChunk) bool {
	if chunk.BifrostChatResponse == nil {
		return false
	}
	for _, choice := range chunk.BifrostChatResponse.Choices {
		if choice.FinishReason != nil {
			return false
		}
		if choice.ChatStreamResponseChoice == nil || choice.ChatStreamResponseChoice.Delta == nil {
			return false
		}
		delta := choice.ChatStreamResponseChoice.Delta
		hasReasoning := delta.Reasoning != nil && *delta.Reasoning != ""
		hasContent := delta.Content != nil && *delta.Content != ""
		hasToolCalls := len(delta.ToolCalls) > 0
		if !hasReasoning || hasContent || hasToolCalls {
			return false
		}
	}
	return len(chunk.BifrostChatResponse.Choices) > 0
}

// extractToolCallsFromResponsesStream extracts all tool calls from a buffered
// Responses API stream. Handles both passthrough (raw Anthropic SSE) and structured chunks.
func extractToolCallsFromResponsesStream(chunks []*schemas.BifrostStreamChunk) []schemas.ChatAssistantMessageToolCall {
	// Try passthrough first.
	if passthrough := extractToolCallsFromPassthrough(chunks); passthrough != nil {
		return passthrough
	}
	// Try structured Responses stream via the completed response.
	resp := reconstructResponsesResponseFromStream(chunks)
	if resp == nil || !hasToolCallsForResponsesResponse(resp) {
		return nil
	}
	chatResp := resp.ToBifrostChatResponse()
	return extractToolCalls(chatResp)
}

// extractToolCallsFromChatStream extracts tool calls from a buffered Chat Completions stream.
func extractToolCallsFromChatStream(chunks []*schemas.BifrostStreamChunk) []schemas.ChatAssistantMessageToolCall {
	resp := reconstructChatResponseFromStream(chunks)
	if resp == nil || !hasToolCallsForChatResponse(resp) {
		return nil
	}
	return extractToolCalls(resp)
}

// extractToolCallsFromPassthrough parses raw Anthropic Messages SSE from passthrough
// chunks and returns any tool_use blocks as bifrost ChatAssistantMessageToolCall values.
// Returns nil if the chunks are not in passthrough format.
func extractToolCallsFromPassthrough(chunks []*schemas.BifrostStreamChunk) []schemas.ChatAssistantMessageToolCall {
	hasPassthrough := false
	type ptToolCall struct {
		id      string
		name    string
		argsBuf strings.Builder
	}
	tools := map[int]*ptToolCall{}
	hasToolUseStop := false

	for _, chunk := range chunks {
		if chunk.BifrostPassthroughResponse == nil {
			continue
		}
		hasPassthrough = true

		for _, line := range strings.Split(string(chunk.BifrostPassthroughResponse.Body), "\n") {
			line = strings.TrimSpace(line)
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if !gjson.Valid(data) {
				continue
			}
			switch gjson.Get(data, "type").String() {
			case "content_block_start":
				if gjson.Get(data, "content_block.type").String() == "tool_use" {
					idx := int(gjson.Get(data, "index").Int())
					tools[idx] = &ptToolCall{
						id:   gjson.Get(data, "content_block.id").String(),
						name: gjson.Get(data, "content_block.name").String(),
					}
				}
			case "content_block_delta":
				idx := int(gjson.Get(data, "index").Int())
				if tc, ok := tools[idx]; ok {
					if gjson.Get(data, "delta.type").String() == "input_json_delta" {
						tc.argsBuf.WriteString(gjson.Get(data, "delta.partial_json").String())
					}
				}
			case "message_delta":
				if gjson.Get(data, "delta.stop_reason").String() == "tool_use" {
					hasToolUseStop = true
				}
			}
		}
	}

	if !hasPassthrough || !hasToolUseStop || len(tools) == 0 {
		return nil
	}

	result := make([]schemas.ChatAssistantMessageToolCall, 0, len(tools))
	for _, tc := range tools {
		name := tc.name
		id := tc.id
		args := tc.argsBuf.String()
		tcType := "function"
		result = append(result, schemas.ChatAssistantMessageToolCall{
			ID:   &id,
			Type: &tcType,
			Function: schemas.ChatAssistantMessageToolCallFunction{
				Name:      &name,
				Arguments: args,
			},
		})
	}
	return result
}

// reconstructResponsesResponseFromStream reassembles a BifrostResponsesResponse from
// a buffered stream. For structured chunks the last response.completed event is used.
// For passthrough chunks a minimal response is synthesised from parsed SSE events.
func reconstructResponsesResponseFromStream(chunks []*schemas.BifrostStreamChunk) *schemas.BifrostResponsesResponse {
	// Structured: use the last response.completed chunk which carries the full response.
	for i := len(chunks) - 1; i >= 0; i-- {
		chunk := chunks[i]
		if chunk.BifrostResponsesStreamResponse != nil &&
			chunk.BifrostResponsesStreamResponse.Type == schemas.ResponsesStreamResponseTypeCompleted &&
			chunk.BifrostResponsesStreamResponse.Response != nil {
			return chunk.BifrostResponsesStreamResponse.Response
		}
	}

	// Passthrough: synthesise a minimal response from parsed Anthropic Messages SSE.
	type ptToolCall struct {
		id      string
		name    string
		argsBuf strings.Builder
	}
	tools := map[int]*ptToolCall{}
	var textContent strings.Builder

	for _, chunk := range chunks {
		if chunk.BifrostPassthroughResponse == nil {
			continue
		}
		for _, line := range strings.Split(string(chunk.BifrostPassthroughResponse.Body), "\n") {
			line = strings.TrimSpace(line)
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if !gjson.Valid(data) {
				continue
			}
			switch gjson.Get(data, "type").String() {
			case "content_block_start":
				if gjson.Get(data, "content_block.type").String() == "tool_use" {
					idx := int(gjson.Get(data, "index").Int())
					tools[idx] = &ptToolCall{
						id:   gjson.Get(data, "content_block.id").String(),
						name: gjson.Get(data, "content_block.name").String(),
					}
				}
			case "content_block_delta":
				idx := int(gjson.Get(data, "index").Int())
				deltaType := gjson.Get(data, "delta.type").String()
				switch deltaType {
				case "text_delta":
					textContent.WriteString(gjson.Get(data, "delta.text").String())
				case "input_json_delta":
					if tc, ok := tools[idx]; ok {
						tc.argsBuf.WriteString(gjson.Get(data, "delta.partial_json").String())
					}
				}
			}
		}
	}

	if len(tools) == 0 && textContent.Len() == 0 {
		return nil
	}

	var output []schemas.ResponsesMessage

	if textContent.Len() > 0 {
		msgType := schemas.ResponsesMessageTypeMessage
		role := schemas.ResponsesInputMessageRoleAssistant
		text := textContent.String()
		contentType := schemas.ResponsesOutputMessageContentTypeText
		output = append(output, schemas.ResponsesMessage{
			Type: &msgType,
			Role: &role,
			Content: &schemas.ResponsesMessageContent{
				ContentBlocks: []schemas.ResponsesMessageContentBlock{
					{Type: contentType, Text: &text},
				},
			},
		})
	}

	for _, tc := range tools {
		name := tc.name
		callID := tc.id
		args := tc.argsBuf.String()
		msgType := schemas.ResponsesMessageTypeFunctionCall
		output = append(output, schemas.ResponsesMessage{
			Type: &msgType,
			ResponsesToolMessage: &schemas.ResponsesToolMessage{
				Name:      &name,
				CallID:    &callID,
				Arguments: &args,
			},
		})
	}

	return &schemas.BifrostResponsesResponse{Output: output}
}

// reconstructChatResponseFromStream reassembles a BifrostChatResponse from a buffered
// Chat Completions stream by merging all streaming deltas into a single non-stream response.
func reconstructChatResponseFromStream(chunks []*schemas.BifrostStreamChunk) *schemas.BifrostChatResponse {
	type toolAccum struct {
		id       string
		toolType string
		name     string
		argsBuf  strings.Builder
	}

	var textBuf strings.Builder
	tools := map[int]*toolAccum{}
	finishReason := ""

	for _, chunk := range chunks {
		if chunk.BifrostChatResponse == nil {
			continue
		}
		for _, choice := range chunk.BifrostChatResponse.Choices {
			if choice.FinishReason != nil {
				finishReason = *choice.FinishReason
			}
			if choice.ChatStreamResponseChoice == nil || choice.ChatStreamResponseChoice.Delta == nil {
				continue
			}
			delta := choice.ChatStreamResponseChoice.Delta
			if delta.Content != nil {
				textBuf.WriteString(*delta.Content)
			}
			for _, tc := range delta.ToolCalls {
				idx := int(tc.Index)
				if _, ok := tools[idx]; !ok {
					tools[idx] = &toolAccum{}
				}
				accum := tools[idx]
				if tc.ID != nil {
					accum.id = *tc.ID
				}
				if tc.Type != nil {
					accum.toolType = *tc.Type
				}
				if tc.Function.Name != nil {
					accum.name = *tc.Function.Name
				}
				accum.argsBuf.WriteString(tc.Function.Arguments)
			}
		}
	}

	if textBuf.Len() == 0 && len(tools) == 0 {
		return nil
	}

	tcs := make([]schemas.ChatAssistantMessageToolCall, 0, len(tools))
	for _, t := range tools {
		id := t.id
		tcType := t.toolType
		name := t.name
		args := t.argsBuf.String()
		tcs = append(tcs, schemas.ChatAssistantMessageToolCall{
			ID:   &id,
			Type: &tcType,
			Function: schemas.ChatAssistantMessageToolCallFunction{
				Name:      &name,
				Arguments: args,
			},
		})
	}

	// Only include text content when there are no tool_calls or when content is
	// non-empty. OpenAI/vLLM reject assistant messages with tool_calls that have
	// an empty-string content field — it must be null/omitted in that case.
	var msgContent *schemas.ChatMessageContent
	if textBuf.Len() > 0 {
		content := textBuf.String()
		msgContent = &schemas.ChatMessageContent{ContentStr: &content}
	}
	msg := &schemas.ChatMessage{
		Role:    schemas.ChatMessageRoleAssistant,
		Content: msgContent,
		ChatAssistantMessage: &schemas.ChatAssistantMessage{
			ToolCalls: tcs,
		},
	}
	fr := finishReason
	return &schemas.BifrostChatResponse{
		Choices: []schemas.BifrostResponseChoice{
			{
				FinishReason: &fr,
				ChatNonStreamResponseChoice: &schemas.ChatNonStreamResponseChoice{
					Message: msg,
				},
			},
		},
	}
}

// cloneResponsesRequest makes a shallow copy of req so the agent loop can mutate
// the Input slice without aliasing the caller's request.
func cloneResponsesRequest(req *schemas.BifrostResponsesRequest) *schemas.BifrostResponsesRequest {
	if req == nil {
		return nil
	}
	clone := *req
	if req.Input != nil {
		clone.Input = make([]schemas.ResponsesMessage, len(req.Input))
		copy(clone.Input, req.Input)
	}
	return &clone
}

// cloneChatRequest makes a shallow copy of req.
func cloneChatRequest(req *schemas.BifrostChatRequest) *schemas.BifrostChatRequest {
	if req == nil {
		return nil
	}
	clone := *req
	if req.Input != nil {
		clone.Input = make([]schemas.ChatMessage, len(req.Input))
		copy(clone.Input, req.Input)
	}
	return &clone
}
