package anthropicmsg

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/capsohq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

func TestPassthroughPreservesNativeMessagesBodyAndUsage(t *testing.T) {
	requestBody := []byte(`{"model":"grok-4.5","max_tokens":64,"system":[{"type":"text","text":"cached","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"use a tool"}]}`)
	responseBody := []byte(`{"id":"msg_xai","type":"message","role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"lookup","input":{"q":"native"}}],"usage":{"input_tokens":128,"cache_read_input_tokens":4736,"cache_creation_input_tokens":64,"output_tokens":12}}`)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/messages" {
			t.Errorf("request = %s %s, want POST /v1/messages", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("x-api-key"); got != "xai-secret" {
			t.Errorf("x-api-key = %q", got)
		}
		if got := r.Header.Get("anthropic-version"); got != DefaultAPIVersion {
			t.Errorf("anthropic-version = %q", got)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
		}
		if !bytes.Equal(body, requestBody) {
			t.Errorf("request body changed:\n got: %s\nwant: %s", body, requestBody)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(responseBody)
	}))
	defer server.Close()

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	resp, bifrostErr := Passthrough(ctx, &fasthttp.Client{}, testKey("xai-secret"), &schemas.BifrostPassthroughRequest{
		Method: http.MethodPost, Path: "/v1/messages", Body: requestBody,
	}, Config{Provider: schemas.XAI, Endpoint: server.URL})
	if bifrostErr != nil {
		t.Fatalf("Passthrough error: %+v", bifrostErr)
	}
	if !bytes.Equal(resp.Body, responseBody) {
		t.Fatalf("response body changed:\n got: %s\nwant: %s", resp.Body, responseBody)
	}
	usage := resp.PassthroughUsage
	if usage == nil || usage.LLMUsage == nil {
		t.Fatalf("missing passthrough usage: %+v", usage)
	}
	if usage.LLMUsage.PromptTokens != 4928 || usage.LLMUsage.CompletionTokens != 12 || usage.LLMUsage.TotalTokens != 4940 {
		t.Fatalf("usage = %+v", usage.LLMUsage)
	}
	details := usage.LLMUsage.PromptTokensDetails
	if details == nil || details.CachedReadTokens != 4736 || details.CachedWriteTokens != 64 {
		t.Fatalf("cache usage = %+v", details)
	}
}

func TestPassthroughStreamRelaysAnthropicSSEVerbatimAndMergesUsage(t *testing.T) {
	frames := []byte("event: message_start\n" +
		"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":128,\"cache_read_input_tokens\":4736,\"cache_creation_input_tokens\":64,\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\n" +
		"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"lookup\",\"input\":{}}}\n\n" +
		"event: message_delta\n" +
		"data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":12}}\n\n" +
		"event: message_stop\n" +
		"data: {\"type\":\"message_stop\"}\n\n")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(frames[:len(frames)/2])
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		_, _ = w.Write(frames[len(frames)/2:])
	}))
	defer server.Close()

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	postHook := func(_ *schemas.BifrostContext, response *schemas.BifrostResponse, err *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError) {
		return response, err
	}
	stream, bifrostErr := PassthroughStream(ctx, postHook, nil, &fasthttp.Client{}, testKey("xai-secret"), &schemas.BifrostPassthroughRequest{
		Method: http.MethodPost, Path: "/v1/messages", Body: []byte(`{"model":"grok-4.5","stream":true}`),
	}, Config{Provider: schemas.XAI, Endpoint: server.URL})
	if bifrostErr != nil {
		t.Fatalf("PassthroughStream error: %+v", bifrostErr)
	}

	var relayed []byte
	var usage *schemas.BifrostPassthroughUsage
	for chunk := range stream {
		if chunk == nil || chunk.BifrostPassthroughResponse == nil {
			continue
		}
		response := chunk.BifrostPassthroughResponse
		relayed = append(relayed, response.Body...)
		if response.PassthroughUsage != nil {
			usage = response.PassthroughUsage
		}
	}
	if !bytes.Equal(relayed, frames) {
		t.Fatalf("SSE bytes changed:\n got: %q\nwant: %q", relayed, frames)
	}
	if usage == nil || usage.LLMUsage == nil {
		t.Fatalf("missing stream usage: %+v", usage)
	}
	if usage.LLMUsage.PromptTokens != 4928 || usage.LLMUsage.CompletionTokens != 12 || usage.LLMUsage.TotalTokens != 4940 {
		t.Fatalf("stream usage = %+v", usage.LLMUsage)
	}
}

func testKey(value string) schemas.Key {
	return schemas.Key{Value: schemas.SecretVar{Val: value}}
}
