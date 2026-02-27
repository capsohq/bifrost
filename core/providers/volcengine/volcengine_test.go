package volcengine_test

import (
	"os"
	"strings"
	"testing"

	"github.com/capsohq/bifrost/core/internal/llmtests"
	"github.com/capsohq/bifrost/core/schemas"
)

func envOrDefault(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func TestVolcengine(t *testing.T) {
	t.Parallel()

	if strings.TrimSpace(os.Getenv("VOLCENGINE_API_KEY")) == "" {
		t.Skip("Skipping Volcengine tests because VOLCENGINE_API_KEY is not set")
	}

	client, ctx, cancel, err := llmtests.SetupTest()
	if err != nil {
		t.Fatalf("Error initializing test setup: %v", err)
	}
	defer cancel()

	testConfig := llmtests.ComprehensiveTestConfig{
		Provider:             schemas.Volcengine,
		ChatModel:            envOrDefault("VOLCENGINE_CHAT_MODEL", "doubao-seed-1-6-250615"),
		TextModel:            envOrDefault("VOLCENGINE_TEXT_MODEL", "doubao-seed-1-6-250615"),
		VisionModel:          envOrDefault("VOLCENGINE_VISION_MODEL", "doubao-1.5-vision-pro-250328"),
		EmbeddingModel:       envOrDefault("VOLCENGINE_EMBEDDING_MODEL", "doubao-embedding-large-text-240915"),
		ImageGenerationModel: envOrDefault("VOLCENGINE_IMAGE_MODEL", "doubao-seedream-4-5-251128"),
		VideoGenerationModel: envOrDefault("VOLCENGINE_VIDEO_MODEL", "doubao-seedance-1-0-lite-i2v-250428"),
		Scenarios: llmtests.TestScenarios{
			TextCompletion:        true,
			TextCompletionStream:  true,
			SimpleChat:            true,
			CompletionStream:      true,
			MultiTurnConversation: true,
			ToolCalls:             true,
			ToolCallsStreaming:    true,
			MultipleToolCalls:     true,
			End2EndToolCalling:    true,
			AutomaticFunctionCall: true,
			ImageURL:              true,
			ImageBase64:           true,
			MultipleImages:        true,
			Embedding:             true,
			ListModels:            true,
			ImageGeneration:       true,
			FileUpload:            true,
			FileList:              true,
			FileRetrieve:          true,
			FileDelete:            true,
			FileContent:           true,
			VideoGeneration:       true,
			VideoRetrieve:         true,
			VideoDownload:         true,
			VideoList:             true,
			VideoDelete:           true,
		},
		DisableParallelFor: []string{"VideoGeneration", "VideoRetrieve", "VideoDownload", "VideoList", "VideoDelete"},
	}

	t.Run("VolcengineTests", func(t *testing.T) {
		llmtests.RunAllComprehensiveTests(t, client, ctx, testConfig)
	})
	client.Shutdown()
}
