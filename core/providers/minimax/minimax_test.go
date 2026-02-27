package minimax_test

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

func TestMinimax(t *testing.T) {
	t.Parallel()

	if strings.TrimSpace(os.Getenv("MINIMAX_API_KEY")) == "" {
		t.Skip("Skipping Minimax tests because MINIMAX_API_KEY is not set")
	}

	client, ctx, cancel, err := llmtests.SetupTest()
	if err != nil {
		t.Fatalf("Error initializing test setup: %v", err)
	}
	defer cancel()

	testConfig := llmtests.ComprehensiveTestConfig{
		Provider:             schemas.Minimax,
		TextModel:            envOrDefault("MINIMAX_TEXT_MODEL", "MiniMax-M2.5"),
		ChatModel:            envOrDefault("MINIMAX_CHAT_MODEL", "M2-her"),
		PromptCachingModel:   envOrDefault("MINIMAX_PROMPT_CACHING_MODEL", "MiniMax-M2.5"),
		ImageGenerationModel: envOrDefault("MINIMAX_IMAGE_MODEL", "image-01"),
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
			PromptCaching:         true,
			ListModels:            true,
			ImageGeneration:       true,
		},
		DisableParallelFor: []string{"PromptCaching"},
	}

	t.Run("MinimaxTests", func(t *testing.T) {
		llmtests.RunAllComprehensiveTests(t, client, ctx, testConfig)
	})
	client.Shutdown()
}
