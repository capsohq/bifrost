package schemas

import (
	"testing"
)

func TestEmbeddingInput_UnmarshalJSON_String(t *testing.T) {
	input := &EmbeddingInput{}
	if err := input.UnmarshalJSON([]byte(`"hello world"`)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if input.Text == nil || *input.Text != "hello world" {
		t.Fatalf("expected Text='hello world', got %v", input.Text)
	}
	if input.IsMultiModal() {
		t.Fatal("expected IsMultiModal()=false for text input")
	}
}

func TestEmbeddingInput_UnmarshalJSON_Texts(t *testing.T) {
	input := &EmbeddingInput{}
	if err := input.UnmarshalJSON([]byte(`["hello","world"]`)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(input.Texts) != 2 || input.Texts[0] != "hello" || input.Texts[1] != "world" {
		t.Fatalf("expected Texts=['hello','world'], got %v", input.Texts)
	}
	if input.IsMultiModal() {
		t.Fatal("expected IsMultiModal()=false for texts input")
	}
}

func TestEmbeddingInput_UnmarshalJSON_Embedding(t *testing.T) {
	input := &EmbeddingInput{}
	if err := input.UnmarshalJSON([]byte(`[1,2,3]`)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(input.Embedding) != 3 {
		t.Fatalf("expected Embedding len=3, got %d", len(input.Embedding))
	}
}

func TestEmbeddingInput_UnmarshalJSON_MultiModal(t *testing.T) {
	data := []byte(`[
		{"type":"text","text":"a photo of a cat"},
		{"type":"image_url","image_url":{"url":"https://example.com/cat.jpg"}},
		{"type":"video_url","video_url":{"url":"https://example.com/cat.mp4"}}
	]`)
	input := &EmbeddingInput{}
	if err := input.UnmarshalJSON(data); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !input.IsMultiModal() {
		t.Fatal("expected IsMultiModal()=true")
	}
	if len(input.MultiModalInputs) != 3 {
		t.Fatalf("expected 3 multimodal inputs, got %d", len(input.MultiModalInputs))
	}

	// Verify text input
	if input.MultiModalInputs[0].Type != MultiModalEmbeddingText {
		t.Fatalf("expected type 'text', got '%s'", input.MultiModalInputs[0].Type)
	}
	if input.MultiModalInputs[0].Text == nil || *input.MultiModalInputs[0].Text != "a photo of a cat" {
		t.Fatal("expected text content 'a photo of a cat'")
	}

	// Verify image input
	if input.MultiModalInputs[1].Type != MultiModalEmbeddingImageURL {
		t.Fatalf("expected type 'image_url', got '%s'", input.MultiModalInputs[1].Type)
	}
	if input.MultiModalInputs[1].ImageURL == nil || input.MultiModalInputs[1].ImageURL.URL != "https://example.com/cat.jpg" {
		t.Fatal("expected image URL 'https://example.com/cat.jpg'")
	}

	// Verify video input
	if input.MultiModalInputs[2].Type != MultiModalEmbeddingVideoURL {
		t.Fatalf("expected type 'video_url', got '%s'", input.MultiModalInputs[2].Type)
	}
	if input.MultiModalInputs[2].VideoURL == nil || input.MultiModalInputs[2].VideoURL.URL != "https://example.com/cat.mp4" {
		t.Fatal("expected video URL 'https://example.com/cat.mp4'")
	}

	// Verify other fields are nil
	if input.Text != nil || input.Texts != nil || input.Embedding != nil || input.Embeddings != nil {
		t.Fatal("expected non-multimodal fields to be nil")
	}
}

func TestEmbeddingInput_UnmarshalJSON_MultiModal_TextOnly(t *testing.T) {
	data := []byte(`[{"type":"text","text":"hello"}]`)
	input := &EmbeddingInput{}
	if err := input.UnmarshalJSON(data); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !input.IsMultiModal() {
		t.Fatal("expected IsMultiModal()=true for typed text input")
	}
	if len(input.MultiModalInputs) != 1 {
		t.Fatalf("expected 1 multimodal input, got %d", len(input.MultiModalInputs))
	}
}

func TestEmbeddingInput_UnmarshalJSON_MultiModal_ImageOnly(t *testing.T) {
	data := []byte(`[{"type":"image_url","image_url":{"url":"https://example.com/img.png"}}]`)
	input := &EmbeddingInput{}
	if err := input.UnmarshalJSON(data); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !input.IsMultiModal() {
		t.Fatal("expected IsMultiModal()=true")
	}
	if input.MultiModalInputs[0].ImageURL.URL != "https://example.com/img.png" {
		t.Fatal("expected image URL to match")
	}
}

func TestEmbeddingInput_UnmarshalJSON_MultiModal_VideoOnly(t *testing.T) {
	data := []byte(`[{"type":"video_url","video_url":{"url":"https://example.com/video.mp4"}}]`)
	input := &EmbeddingInput{}
	if err := input.UnmarshalJSON(data); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !input.IsMultiModal() {
		t.Fatal("expected IsMultiModal()=true")
	}
	if input.MultiModalInputs[0].VideoURL.URL != "https://example.com/video.mp4" {
		t.Fatal("expected video URL to match")
	}
}

func TestEmbeddingInput_MarshalJSON_MultiModal(t *testing.T) {
	text := "a cat"
	input := &EmbeddingInput{
		MultiModalInputs: []MultiModalEmbeddingInput{
			{Type: MultiModalEmbeddingText, Text: &text},
			{Type: MultiModalEmbeddingImageURL, ImageURL: &MultiModalEmbeddingMediaURL{URL: "https://example.com/cat.jpg"}},
		},
	}
	data, err := input.MarshalJSON()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Round-trip: unmarshal the marshalled data
	input2 := &EmbeddingInput{}
	if err := input2.UnmarshalJSON(data); err != nil {
		t.Fatalf("round-trip unmarshal failed: %v", err)
	}
	if !input2.IsMultiModal() {
		t.Fatal("round-trip: expected IsMultiModal()=true")
	}
	if len(input2.MultiModalInputs) != 2 {
		t.Fatalf("round-trip: expected 2 multimodal inputs, got %d", len(input2.MultiModalInputs))
	}
	if input2.MultiModalInputs[0].Type != MultiModalEmbeddingText {
		t.Fatal("round-trip: expected first input type 'text'")
	}
	if input2.MultiModalInputs[1].Type != MultiModalEmbeddingImageURL {
		t.Fatal("round-trip: expected second input type 'image_url'")
	}
}

func TestEmbeddingInput_MarshalJSON_Empty(t *testing.T) {
	input := &EmbeddingInput{}
	_, err := input.MarshalJSON()
	if err == nil {
		t.Fatal("expected error for empty input")
	}
}

func TestEmbeddingInput_MarshalJSON_MultipleSet(t *testing.T) {
	text := "hello"
	input := &EmbeddingInput{
		Text:  &text,
		Texts: []string{"a", "b"},
	}
	_, err := input.MarshalJSON()
	if err == nil {
		t.Fatal("expected error when multiple fields are set")
	}
}

func TestEmbeddingInput_MarshalJSON_MultiModalAndText(t *testing.T) {
	text := "hello"
	input := &EmbeddingInput{
		Text: &text,
		MultiModalInputs: []MultiModalEmbeddingInput{
			{Type: MultiModalEmbeddingText, Text: &text},
		},
	}
	_, err := input.MarshalJSON()
	if err == nil {
		t.Fatal("expected error when both Text and MultiModalInputs are set")
	}
}

func TestEmbeddingInput_IsMultiModal(t *testing.T) {
	text := "hello"

	tests := []struct {
		name     string
		input    EmbeddingInput
		expected bool
	}{
		{"nil multimodal", EmbeddingInput{Text: &text}, false},
		{"empty multimodal", EmbeddingInput{MultiModalInputs: []MultiModalEmbeddingInput{}}, false},
		{"with multimodal", EmbeddingInput{MultiModalInputs: []MultiModalEmbeddingInput{{Type: MultiModalEmbeddingText, Text: &text}}}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.input.IsMultiModal(); got != tt.expected {
				t.Fatalf("IsMultiModal()=%v, want %v", got, tt.expected)
			}
		})
	}
}

func TestEmbeddingInput_UnmarshalJSON_InvalidInput(t *testing.T) {
	input := &EmbeddingInput{}
	err := input.UnmarshalJSON([]byte(`{"invalid": true}`))
	if err == nil {
		t.Fatal("expected error for invalid input shape")
	}
}

func TestMultiModalEmbeddingInput_Types(t *testing.T) {
	if MultiModalEmbeddingText != "text" {
		t.Fatalf("expected 'text', got '%s'", MultiModalEmbeddingText)
	}
	if MultiModalEmbeddingImageURL != "image_url" {
		t.Fatalf("expected 'image_url', got '%s'", MultiModalEmbeddingImageURL)
	}
	if MultiModalEmbeddingVideoURL != "video_url" {
		t.Fatalf("expected 'video_url', got '%s'", MultiModalEmbeddingVideoURL)
	}
}
