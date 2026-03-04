package schemas

import (
	"fmt"
)

type BifrostEmbeddingRequest struct {
	Provider       ModelProvider        `json:"provider"`
	Model          string               `json:"model"`
	Input          *EmbeddingInput      `json:"input,omitempty"`
	Params         *EmbeddingParameters `json:"params,omitempty"`
	Fallbacks      []Fallback           `json:"fallbacks,omitempty"`
	RawRequestBody []byte               `json:"-"` // set bifrost-use-raw-request-body to true in ctx to use the raw request body. Bifrost will directly send this to the downstream provider.
}

func (r *BifrostEmbeddingRequest) GetRawRequestBody() []byte {
	return r.RawRequestBody
}

type BifrostEmbeddingResponse struct {
	Data        []EmbeddingData            `json:"data"` // Maps to "data" field in provider responses (e.g., OpenAI embedding format)
	Model       string                     `json:"model"`
	Object      string                     `json:"object"` // "list"
	Usage       *BifrostLLMUsage           `json:"usage"`
	ExtraFields BifrostResponseExtraFields `json:"extra_fields"`
}

// MultiModalEmbeddingInputType represents the type of multimodal embedding input.
type MultiModalEmbeddingInputType string

const (
	MultiModalEmbeddingText     MultiModalEmbeddingInputType = "text"
	MultiModalEmbeddingImageURL MultiModalEmbeddingInputType = "image_url"
	MultiModalEmbeddingVideoURL MultiModalEmbeddingInputType = "video_url"
)

// MultiModalEmbeddingInput represents a single input item for multimodal embedding.
type MultiModalEmbeddingInput struct {
	Type     MultiModalEmbeddingInputType `json:"type"`
	Text     *string                      `json:"text,omitempty"`
	ImageURL *MultiModalEmbeddingMediaURL `json:"image_url,omitempty"`
	VideoURL *MultiModalEmbeddingMediaURL `json:"video_url,omitempty"`
}

// MultiModalEmbeddingMediaURL holds the URL for an image or video input.
type MultiModalEmbeddingMediaURL struct {
	URL string `json:"url"`
}

// EmbeddingInput represents the input for an embedding request.
type EmbeddingInput struct {
	Text             *string
	Texts            []string
	Embedding        []int
	Embeddings       [][]int
	MultiModalInputs []MultiModalEmbeddingInput // For multimodal embedding (text + image_url + video_url)
}

// IsMultiModal returns true if the input contains multimodal inputs.
func (e *EmbeddingInput) IsMultiModal() bool {
	return len(e.MultiModalInputs) > 0
}

func (e *EmbeddingInput) MarshalJSON() ([]byte, error) {
	// enforce one-of
	set := 0
	if e.Text != nil {
		set++
	}
	if e.Texts != nil {
		set++
	}
	if e.Embedding != nil {
		set++
	}
	if e.Embeddings != nil {
		set++
	}
	if e.MultiModalInputs != nil {
		set++
	}
	if set == 0 {
		return nil, fmt.Errorf("embedding input is empty")
	}
	if set > 1 {
		return nil, fmt.Errorf("embedding input must set exactly one of: text, texts, embedding, embeddings, multimodal")
	}

	if e.Text != nil {
		return Marshal(*e.Text)
	}
	if e.Texts != nil {
		return Marshal(e.Texts)
	}
	if e.Embedding != nil {
		return Marshal(e.Embedding)
	}
	if e.Embeddings != nil {
		return Marshal(e.Embeddings)
	}
	if e.MultiModalInputs != nil {
		return Marshal(e.MultiModalInputs)
	}

	return nil, fmt.Errorf("invalid embedding input")
}

func (e *EmbeddingInput) UnmarshalJSON(data []byte) error {
	e.Text = nil
	e.Texts = nil
	e.Embedding = nil
	e.Embeddings = nil
	e.MultiModalInputs = nil
	// Try string
	var s string
	if err := Unmarshal(data, &s); err == nil {
		e.Text = &s
		return nil
	}
	// Try multimodal (array of objects with "type" field) — must try before []string
	var mm []MultiModalEmbeddingInput
	if err := Unmarshal(data, &mm); err == nil && len(mm) > 0 && mm[0].Type != "" {
		e.MultiModalInputs = mm
		return nil
	}
	// Try []string
	var ss []string
	if err := Unmarshal(data, &ss); err == nil {
		e.Texts = ss
		return nil
	}
	// Try []int
	var i []int
	if err := Unmarshal(data, &i); err == nil {
		e.Embedding = i
		return nil
	}
	// Try [][]int
	var i2 [][]int
	if err := Unmarshal(data, &i2); err == nil {
		e.Embeddings = i2
		return nil
	}

	return fmt.Errorf("unsupported embedding input shape")
}

type EmbeddingParameters struct {
	EncodingFormat *string `json:"encoding_format,omitempty"` // Format for embedding output (e.g., "float", "base64")
	Dimensions     *int    `json:"dimensions,omitempty"`      // Number of dimensions for embedding output
	Instructions   *string `json:"instructions,omitempty"`    // Optional provider-specific embedding instruction/prompt

	// Optional provider-specific sparse embedding configuration.
	// Example: {"type":"enabled"}
	SparseEmbedding map[string]interface{} `json:"sparse_embedding,omitempty"`

	// Dynamic parameters that can be provider-specific, they are directly
	// added to the request as is.
	ExtraParams map[string]interface{} `json:"-"`
}

type EmbeddingData struct {
	Index           int                    `json:"index"`
	Object          string                 `json:"object"`                     // "embedding"
	Embedding       EmbeddingStruct        `json:"embedding"`                  // can be string, []float32 or [][]float32
	SparseEmbedding []EmbeddingSparseValue `json:"sparse_embedding,omitempty"` // optional sparse vector entries
}

type EmbeddingSparseValue struct {
	Index int     `json:"index"`
	Value float32 `json:"value"`
}

type EmbeddingStruct struct {
	EmbeddingStr     *string
	EmbeddingArray   []float32
	Embedding2DArray [][]float32
}

func (be EmbeddingStruct) MarshalJSON() ([]byte, error) {
	if be.EmbeddingStr != nil {
		return Marshal(be.EmbeddingStr)
	}
	if be.EmbeddingArray != nil {
		return Marshal(be.EmbeddingArray)
	}
	if be.Embedding2DArray != nil {
		return Marshal(be.Embedding2DArray)
	}
	return nil, fmt.Errorf("no embedding found")
}

func (be *EmbeddingStruct) UnmarshalJSON(data []byte) error {
	// First, try to unmarshal as a direct string
	var stringContent string
	if err := Unmarshal(data, &stringContent); err == nil {
		be.EmbeddingStr = &stringContent
		return nil
	}

	// Try to unmarshal as a direct array of float32
	var arrayContent []float32
	if err := Unmarshal(data, &arrayContent); err == nil {
		be.EmbeddingArray = arrayContent
		return nil
	}

	// Try to unmarshal as a direct 2D array of float32
	var arrayContent2D [][]float32
	if err := Unmarshal(data, &arrayContent2D); err == nil {
		be.Embedding2DArray = arrayContent2D
		return nil
	}

	return fmt.Errorf("embedding field is neither a string nor an array of float32 nor a 2D array of float32")
}
