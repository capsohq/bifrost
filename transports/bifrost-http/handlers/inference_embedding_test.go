package handlers

import (
	"testing"

	"github.com/capsohq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

func TestPrepareEmbeddingRequest_MultimodalVolcengineFields(t *testing.T) {
	t.Parallel()

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.SetBodyString(`{
		"model": "volcengine/doubao-embedding-vision-251215",
		"input": [
			{
				"type": "text",
				"text": "the sky is blue"
			}
		],
		"instructions": "Target_modality: text and video.\nInstruction:Compress the text\\video into one word.\nQuery:",
		"encoding_format": "float",
		"dimensions": 1024,
		"sparse_embedding": {
			"type": "enabled"
		}
	}`)

	req, bifrostReq, err := prepareEmbeddingRequest(ctx)
	if err != nil {
		t.Fatalf("prepareEmbeddingRequest returned error: %v", err)
	}
	if req == nil || bifrostReq == nil || bifrostReq.Params == nil {
		t.Fatal("expected non-nil request and params")
	}

	if bifrostReq.Model != "doubao-embedding-vision-251215" {
		t.Fatalf("expected model doubao-embedding-vision-251215, got %s", bifrostReq.Model)
	}

	if bifrostReq.Params.Instructions == nil || *bifrostReq.Params.Instructions == "" {
		t.Fatal("expected instructions to be parsed")
	}
	if bifrostReq.Params.EncodingFormat == nil || *bifrostReq.Params.EncodingFormat != "float" {
		t.Fatalf("expected encoding_format float, got %v", bifrostReq.Params.EncodingFormat)
	}
	if bifrostReq.Params.Dimensions == nil || *bifrostReq.Params.Dimensions != 1024 {
		t.Fatalf("expected dimensions 1024, got %v", bifrostReq.Params.Dimensions)
	}
	if bifrostReq.Params.SparseEmbedding == nil {
		t.Fatal("expected sparse_embedding to be parsed")
	}
	if sparseType, ok := bifrostReq.Params.SparseEmbedding["type"]; !ok || sparseType != "enabled" {
		t.Fatalf("expected sparse_embedding.type enabled, got %v", bifrostReq.Params.SparseEmbedding)
	}
}

func TestPrepareEmbeddingRequest_ExtractsVolcengineInstructionsConfig(t *testing.T) {
	t.Parallel()

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.SetBodyString(`{
		"model": "volcengine/doubao-embedding-vision-251215",
		"input": "hello",
		"encoding_format": "float",
		"volcengine_instructions_config": {
			"task_type": "retrieval/ranking",
			"role": "corpus",
			"target_modality": "text",
			"validate_template": true
		}
	}`)

	_, bifrostReq, err := prepareEmbeddingRequest(ctx)
	if err != nil {
		t.Fatalf("prepareEmbeddingRequest returned error: %v", err)
	}
	if bifrostReq == nil || bifrostReq.Params == nil {
		t.Fatal("expected non-nil bifrost request and params")
	}
	if bifrostReq.Input == nil || bifrostReq.Input.Text == nil || *bifrostReq.Input.Text != "hello" {
		t.Fatalf("expected string input to be parsed, got %+v", bifrostReq.Input)
	}
	if bifrostReq.Params.ExtraParams == nil {
		t.Fatal("expected extra params to be populated")
	}
	rawConfig, ok := bifrostReq.Params.ExtraParams["volcengine_instructions_config"]
	if !ok {
		t.Fatalf("expected volcengine_instructions_config in extra params, got %v", bifrostReq.Params.ExtraParams)
	}
	configMap, ok := rawConfig.(map[string]any)
	if !ok {
		t.Fatalf("expected volcengine_instructions_config to be a map, got %T", rawConfig)
	}
	if taskType, ok := configMap["task_type"].(string); !ok || taskType != "retrieval/ranking" {
		t.Fatalf("expected task_type retrieval/ranking, got %#v", configMap["task_type"])
	}
	if role, ok := configMap["role"].(string); !ok || role != "corpus" {
		t.Fatalf("expected role corpus, got %#v", configMap["role"])
	}
	if validateTemplate, ok := configMap["validate_template"].(bool); !ok || !validateTemplate {
		t.Fatalf("expected validate_template true, got %#v", configMap["validate_template"])
	}
	if bifrostReq.Provider != schemas.Volcengine {
		t.Fatalf("expected provider volcengine, got %s", bifrostReq.Provider)
	}
}
