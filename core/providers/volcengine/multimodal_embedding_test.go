package volcengine

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	schemas "github.com/capsohq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

type testLogger struct{}

func (l *testLogger) Debug(msg string, args ...any)                     {}
func (l *testLogger) Info(msg string, args ...any)                      {}
func (l *testLogger) Warn(msg string, args ...any)                      {}
func (l *testLogger) Error(msg string, args ...any)                     {}
func (l *testLogger) Fatal(msg string, args ...any)                     {}
func (l *testLogger) SetLevel(level schemas.LogLevel)                   {}
func (l *testLogger) SetOutputType(outputType schemas.LoggerOutputType) {}
func (l *testLogger) LogHTTPRequest(level schemas.LogLevel, msg string) schemas.LogEventBuilder {
	return &noopLogEventBuilder{}
}

type noopLogEventBuilder struct{}

func (b *noopLogEventBuilder) Str(key, val string) schemas.LogEventBuilder         { return b }
func (b *noopLogEventBuilder) Int(key string, val int) schemas.LogEventBuilder     { return b }
func (b *noopLogEventBuilder) Int64(key string, val int64) schemas.LogEventBuilder { return b }
func (b *noopLogEventBuilder) Send()                                               {}

func newTestVolcengineProvider(baseURL string) *VolcengineProvider {
	return &VolcengineProvider{
		logger: &testLogger{},
		client: &fasthttp.Client{
			ReadTimeout:  5 * time.Second,
			WriteTimeout: 5 * time.Second,
		},
		networkConfig: schemas.NetworkConfig{
			BaseURL: baseURL,
		},
	}
}

func TestMultiModalEmbedding_TextAndImage(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings/multimodal" {
			t.Errorf("expected path /embeddings/multimodal, got %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("expected Authorization Bearer test-key, got %s", r.Header.Get("Authorization"))
		}
		var requestBody struct {
			Model           string `json:"model"`
			Instructions    string `json:"instructions"`
			EncodingFormat  string `json:"encoding_format"`
			Dimensions      int    `json:"dimensions"`
			SparseEmbedding struct {
				Type string `json:"type"`
			} `json:"sparse_embedding"`
		}
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}
		if requestBody.Model != "doubao-embedding-vision-250615" {
			t.Errorf("expected model doubao-embedding-vision-250615, got %s", requestBody.Model)
		}
		if requestBody.Instructions != "Target_modality: text and video.\nInstruction:Compress the text\\video into one word.\nQuery:" {
			t.Errorf("unexpected instructions: %s", requestBody.Instructions)
		}
		if requestBody.EncodingFormat != "float" {
			t.Errorf("expected encoding_format float, got %s", requestBody.EncodingFormat)
		}
		if requestBody.Dimensions != 2048 {
			t.Errorf("expected dimensions 2048, got %d", requestBody.Dimensions)
		}
		if requestBody.SparseEmbedding.Type != "enabled" {
			t.Errorf("expected sparse_embedding.type enabled, got %s", requestBody.SparseEmbedding.Type)
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{
			"data": {
				"embedding": [0.1, 0.2, 0.3, 0.4],
				"sparse_embedding": [
					{"index": 1, "value": 0.0887451171875},
					{"index": 13, "value": 0.0125}
				],
				"object": "embedding"
			},
			"id": "test-id",
			"created": 1752133360,
			"model": "doubao-embedding-vision-250615",
			"object": "list",
			"usage": {
				"prompt_tokens": 42,
				"prompt_tokens_details": {"text_tokens": 42, "image_tokens": 0},
				"total_tokens": 42
			}
		}`)
	}))
	defer server.Close()

	provider := newTestVolcengineProvider(server.URL)

	text := "a photo of a cat"
	instructions := "Target_modality: text and video.\nInstruction:Compress the text\\video into one word.\nQuery:"
	encodingFormat := "float"
	request := &schemas.BifrostEmbeddingRequest{
		Provider: schemas.Volcengine,
		Model:    "doubao-embedding-vision-250615",
		Input: &schemas.EmbeddingInput{
			MultiModalInputs: []schemas.MultiModalEmbeddingInput{
				{Type: schemas.MultiModalEmbeddingText, Text: &text},
				{Type: schemas.MultiModalEmbeddingImageURL, ImageURL: &schemas.MultiModalEmbeddingMediaURL{URL: "https://example.com/cat.jpg"}},
			},
		},
		Params: &schemas.EmbeddingParameters{
			Instructions:   &instructions,
			EncodingFormat: &encodingFormat,
			Dimensions:     intPtr(2048),
			SparseEmbedding: map[string]interface{}{
				"type": "enabled",
			},
		},
	}

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	resp, bifrostErr := provider.Embedding(ctx, schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, request)
	if bifrostErr != nil {
		t.Fatalf("Embedding returned error: %v", bifrostErr.Error)
	}

	if resp == nil {
		t.Fatal("expected non-nil response")
	}
	if len(resp.Data) != 1 {
		t.Fatalf("expected 1 embedding data, got %d", len(resp.Data))
	}
	if len(resp.Data[0].Embedding.EmbeddingArray) != 4 {
		t.Fatalf("expected 4-dim embedding, got %d", len(resp.Data[0].Embedding.EmbeddingArray))
	}
	if resp.Data[0].Embedding.EmbeddingArray[0] != 0.1 {
		t.Fatalf("expected first value 0.1, got %f", resp.Data[0].Embedding.EmbeddingArray[0])
	}
	if resp.Data[0].Object != "embedding" {
		t.Fatalf("expected object 'embedding', got '%s'", resp.Data[0].Object)
	}
	if len(resp.Data[0].SparseEmbedding) != 2 {
		t.Fatalf("expected 2 sparse embedding entries, got %d", len(resp.Data[0].SparseEmbedding))
	}
	if resp.Data[0].SparseEmbedding[0].Index != 1 {
		t.Fatalf("expected first sparse embedding index 1, got %d", resp.Data[0].SparseEmbedding[0].Index)
	}
	if resp.Data[0].SparseEmbedding[0].Value <= 0 {
		t.Fatalf("expected first sparse embedding value > 0, got %f", resp.Data[0].SparseEmbedding[0].Value)
	}
	if resp.Model != "doubao-embedding-vision-250615" {
		t.Fatalf("expected model 'doubao-embedding-vision-250615', got '%s'", resp.Model)
	}
	if resp.Object != "list" {
		t.Fatalf("expected object 'list', got '%s'", resp.Object)
	}
	if resp.Usage == nil || resp.Usage.TotalTokens != 42 {
		t.Fatalf("expected usage.total_tokens=42, got %v", resp.Usage)
	}
	if resp.Usage == nil || resp.Usage.PromptTokens != 42 {
		t.Fatalf("expected usage.prompt_tokens=42, got %v", resp.Usage)
	}
	if resp.Usage == nil || resp.Usage.PromptTokensDetails == nil || resp.Usage.PromptTokensDetails.TextTokens != 42 {
		t.Fatalf("expected usage.prompt_tokens_details.text_tokens=42, got %v", resp.Usage)
	}
	if resp.ExtraFields.Provider != schemas.Volcengine {
		t.Fatalf("expected provider volcengine, got %s", resp.ExtraFields.Provider)
	}
	if resp.ExtraFields.RequestType != schemas.EmbeddingRequest {
		t.Fatalf("expected request type embedding, got %s", resp.ExtraFields.RequestType)
	}
}

func TestMultiModalEmbedding_VideoInput(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings/multimodal" {
			t.Errorf("expected path /embeddings/multimodal, got %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{
			"data": {"embedding": [0.5, 0.6]},
			"model": "doubao-embedding-vision-250615"
		}`)
	}))
	defer server.Close()

	provider := newTestVolcengineProvider(server.URL)

	request := &schemas.BifrostEmbeddingRequest{
		Provider: schemas.Volcengine,
		Model:    "doubao-embedding-vision-250615",
		Input: &schemas.EmbeddingInput{
			MultiModalInputs: []schemas.MultiModalEmbeddingInput{
				{Type: schemas.MultiModalEmbeddingVideoURL, VideoURL: &schemas.MultiModalEmbeddingMediaURL{URL: "https://example.com/video.mp4"}},
			},
		},
	}

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	resp, bifrostErr := provider.Embedding(ctx, schemas.Key{Value: schemas.EnvVar{Val: "key"}}, request)
	if bifrostErr != nil {
		t.Fatalf("Embedding returned error: %v", bifrostErr.Error)
	}

	if len(resp.Data) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(resp.Data))
	}
	if len(resp.Data[0].Embedding.EmbeddingArray) != 2 {
		t.Fatalf("expected 2-dim embedding, got %d", len(resp.Data[0].Embedding.EmbeddingArray))
	}
	// Usage should be nil when not returned
	if resp.Usage != nil {
		t.Fatalf("expected nil usage when not in response, got %v", resp.Usage)
	}
}

func TestMultiModalEmbedding_FallsBackToTextEmbedding(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Text-only requests should go to /embeddings, not /embeddings/multimodal
		if r.URL.Path != "/embeddings" {
			t.Errorf("expected path /embeddings for text input, got %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{
			"data": [{"index": 0, "object": "embedding", "embedding": [0.1, 0.2]}],
			"model": "doubao-embedding-large-text-240915",
			"object": "list",
			"usage": {"prompt_tokens": 5, "total_tokens": 5}
		}`)
	}))
	defer server.Close()

	provider := newTestVolcengineProvider(server.URL)

	text := "hello world"
	request := &schemas.BifrostEmbeddingRequest{
		Provider: schemas.Volcengine,
		Model:    "doubao-embedding-large-text-240915",
		Input: &schemas.EmbeddingInput{
			Text: &text,
		},
	}

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	resp, bifrostErr := provider.Embedding(ctx, schemas.Key{Value: schemas.EnvVar{Val: "key"}}, request)
	if bifrostErr != nil {
		t.Fatalf("Embedding returned error: %v", bifrostErr.Error)
	}

	if len(resp.Data) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(resp.Data))
	}
}

func TestMultiModalEmbedding_ServerError(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":{"message":"invalid model","type":"invalid_request_error"}}`)
	}))
	defer server.Close()

	provider := newTestVolcengineProvider(server.URL)

	text := "test"
	request := &schemas.BifrostEmbeddingRequest{
		Provider: schemas.Volcengine,
		Model:    "nonexistent-model",
		Input: &schemas.EmbeddingInput{
			MultiModalInputs: []schemas.MultiModalEmbeddingInput{
				{Type: schemas.MultiModalEmbeddingText, Text: &text},
			},
		},
	}

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	_, bifrostErr := provider.Embedding(ctx, schemas.Key{Value: schemas.EnvVar{Val: "key"}}, request)
	if bifrostErr == nil {
		t.Fatal("expected error for server error response")
	}
}

func TestMultiModalEmbedding_MixedInputs(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings/multimodal" {
			t.Errorf("expected path /embeddings/multimodal, got %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{
			"data": {"embedding": [0.1, 0.2, 0.3]},
			"model": "doubao-embedding-vision-250615",
			"usage": {"total_tokens": 100}
		}`)
	}))
	defer server.Close()

	provider := newTestVolcengineProvider(server.URL)

	text := "a cat playing"
	request := &schemas.BifrostEmbeddingRequest{
		Provider: schemas.Volcengine,
		Model:    "doubao-embedding-vision-250615",
		Input: &schemas.EmbeddingInput{
			MultiModalInputs: []schemas.MultiModalEmbeddingInput{
				{Type: schemas.MultiModalEmbeddingText, Text: &text},
				{Type: schemas.MultiModalEmbeddingImageURL, ImageURL: &schemas.MultiModalEmbeddingMediaURL{URL: "https://example.com/cat.jpg"}},
				{Type: schemas.MultiModalEmbeddingVideoURL, VideoURL: &schemas.MultiModalEmbeddingMediaURL{URL: "https://example.com/cat.mp4"}},
			},
		},
	}

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	resp, bifrostErr := provider.Embedding(ctx, schemas.Key{Value: schemas.EnvVar{Val: "key"}}, request)
	if bifrostErr != nil {
		t.Fatalf("Embedding returned error: %v", bifrostErr.Error)
	}

	if len(resp.Data) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(resp.Data))
	}
	if len(resp.Data[0].Embedding.EmbeddingArray) != 3 {
		t.Fatalf("expected 3-dim embedding, got %d", len(resp.Data[0].Embedding.EmbeddingArray))
	}
}

func TestMultiModalEmbedding_RawRequestResponse(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"data": {"embedding": [0.1]}, "model": "test"}`)
	}))
	defer server.Close()

	provider := &VolcengineProvider{
		logger: &testLogger{},
		client: &fasthttp.Client{
			ReadTimeout:  5 * time.Second,
			WriteTimeout: 5 * time.Second,
		},
		networkConfig: schemas.NetworkConfig{
			BaseURL: server.URL,
		},
		sendBackRawRequest:  true,
		sendBackRawResponse: true,
	}

	text := "test"
	request := &schemas.BifrostEmbeddingRequest{
		Provider: schemas.Volcengine,
		Model:    "test",
		Input: &schemas.EmbeddingInput{
			MultiModalInputs: []schemas.MultiModalEmbeddingInput{
				{Type: schemas.MultiModalEmbeddingText, Text: &text},
			},
		},
	}

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	resp, bifrostErr := provider.Embedding(ctx, schemas.Key{Value: schemas.EnvVar{Val: "key"}}, request)
	if bifrostErr != nil {
		t.Fatalf("Embedding returned error: %v", bifrostErr.Error)
	}

	if resp.ExtraFields.RawRequest == nil {
		t.Fatal("expected RawRequest to be set")
	}
	if resp.ExtraFields.RawResponse == nil {
		t.Fatal("expected RawResponse to be set")
	}
}

func intPtr(v int) *int {
	return &v
}
